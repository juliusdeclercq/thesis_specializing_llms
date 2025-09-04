# -*- coding: utf-8 -*-
"""
Created on Tue Apr 15 11:44:39 2025

@author: Julius de Clercq
"""

import random
import os
import tempfile
import shutil
import time
import argparse
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
from multiprocessing import Manager
import math
import traceback # Keep for printing tracebacks on error


# --- Configuration ---
NUM_BUCKETS = 128
MAX_WORKERS_PASS1 = int(os.environ["SLURM_CPUS_PER_TASK"])
MAX_WORKERS_PASS2 = 4
RANDOM_SEED = 98765432123456789
# --- End Configuration ---

# Global variable for file size (set in run_parallel_partition)
file_size = 0

def parse_arguments():
    """Parses command-line arguments."""
    parser = argparse.ArgumentParser(description="Shuffle large JSONL file.")
    parser.add_argument('scratch_dir', type=str, help="Path to the scratch directory for input and active processing.")
    parser.add_argument('project_dir', type=str, help="Path to the project directory for temp files and outputs.")
    parser.add_argument('--input_file', type=str, help="Path to the input data.")
    parser.add_argument('--train_size', type=str, help="Maximum number of lines to write to output (for truncation).")
    parser.add_argument('--eval_size', type=str, help="Number of examples to use as evaluation data.")
    args = parser.parse_args()
    args.eval_size = int(float(args.eval_size))
    args.train_size = int(float(args.train_size))
    return args.scratch_dir, args.project_dir, args.input_file, args.train_size, args.eval_size

def calculate_optimal_buffer_size(num_workers, num_buckets, available_memory_gb=336):
    """
    Calculate optimal buffer size based on workers and available memory.
    
    Reserve memory for:
    - OS and other processes: 10 GB
    - Python interpreter per worker: ~200 MB
    - Safety margin: 20% of remaining
    """
    os_reserve_gb = 10
    python_overhead_per_worker_gb = 0.2
    safety_factor = 0.8  # Use only 80% of available
    
    # Calculate available memory for buffers
    total_python_overhead_gb = num_workers * python_overhead_per_worker_gb
    usable_memory_gb = (available_memory_gb - os_reserve_gb - total_python_overhead_gb) * safety_factor
    
    # Each worker opens num_buckets files
    total_file_handles = num_workers * num_buckets
    
    # Calculate buffer size in bytes
    buffer_size_bytes = int((usable_memory_gb * 1024**3) / total_file_handles)
    
    # Ensure minimum buffer size of 1 MB and maximum of 64 MB
    buffer_size_bytes = max(1024 * 1024, min(buffer_size_bytes, 64 * 1024 * 1024))
    
    return buffer_size_bytes

def find_newline_offset(file_handle, start_offset):
    """
    Finds the byte offset of the first newline character at or after start_offset.
    Crucial for starting reads at line boundaries.
    """
    if start_offset == 0:
        return 0 # Start of file is always a valid boundary

    file_handle.seek(start_offset)
    chunk = file_handle.read(1024) # Read a small chunk to find the newline
    try:
        newline_pos = chunk.index('\n')
        # Return position *after* the newline
        return start_offset + newline_pos + 1
    except ValueError:
        # If no newline found in the chunk, try reading line by line cautiously
        # This case might happen with very long lines or near EOF
        file_handle.seek(start_offset) # Go back to original position
        try:
            # Read potentially partial line and discard it to align to next line
            _ = file_handle.readline()
            # Return the position of the start of the *next* line
            return file_handle.tell()
        except Exception as e:
             # If readline fails (e.g., EOF), returning original offset might be
             # the only option, though this scenario should be rare if chunking is correct.
             print(f"Warning: find_newline_offset encountered issue near offset {start_offset}: {e}. Returning original offset.")
             return start_offset

def partition_chunk(worker_id, input_file, temp_dir, num_buckets, start_byte, end_byte, total_lines_counter, IO_BUFFER_SIZE):
    """
    Pass 1 Worker: Reads a byte range [start_byte, end_byte) of the input file,
    hashes each line to a bucket index, and writes the line to a
    worker-specific temporary bucket file.
    """
    print(f"  [Pass 1 - Worker {worker_id:02d}] Started. Range: {start_byte}-{end_byte}")
    worker_bucket_filenames = [os.path.join(temp_dir, f"w{worker_id:02d}_bucket_{i:04d}.tmp") for i in range(num_buckets)]
    worker_bucket_handles = []
    lines_processed = 0
    
    
    try:
        # Open all bucket files for writing
        worker_bucket_handles = [open(fname, "w", encoding="utf-8", buffering=IO_BUFFER_SIZE) for fname in worker_bucket_filenames]

        with open(input_file, "r", encoding="utf-8", buffering=IO_BUFFER_SIZE) as infile:
            # Ensure reading starts on a line boundary (unless it's the first chunk)
            start_byte = find_newline_offset(infile, start_byte)
            infile.seek(start_byte)
            current_pos = start_byte

            # Read lines until the read position reaches or exceeds the end_byte
            while current_pos < end_byte:
                line = infile.readline()
                if not line:
                    break # End of file reached within the chunk

                # Calculate position *after* reading the line
                next_pos = infile.tell()

                # Only process lines that *start* within this chunk's range
                # This check prevents processing lines that belong to the next chunk
                if current_pos < end_byte:
                    target_bucket_index = random.randint(0, num_buckets - 1)
                    worker_bucket_handles[target_bucket_index].write(line)
                    lines_processed += 1

                current_pos = next_pos
                # Stop if we've read up to or past the end boundary
                if current_pos >= end_byte:
                    break

    finally:
        # CRITICAL: Ensure all file handles are closed to flush buffers
        for handle in worker_bucket_handles:
            if handle and not handle.closed:
                handle.close()

    # Update shared counter (Manager handles atomicity)
    total_lines_counter.value += lines_processed
    print(f"  [Pass 1 - Worker {worker_id:02d}] Finished. Processed {lines_processed:,} lines.")
    return worker_bucket_filenames


def run_parallel_partition(input_file, temp_dir, num_buckets, num_workers):
    """
    Orchestrates Pass 1: Calculates byte ranges for workers and runs
    partition_chunk tasks in parallel using ProcessPoolExecutor.
    """
    print(f"[Pass 1] Starting parallel partitioning with {num_workers} workers...")
    start_time = time.time()
    global file_size # Access the global variable
    file_size = os.path.getsize(input_file)
    if file_size == 0:
        raise ValueError("Input file is empty.")

    # Calculate roughly equal byte chunks
    chunk_size = math.ceil(file_size / num_workers)
    all_bucket_files = []
    manager = Manager()
    total_lines_processed = manager.Value('L', 0) # Shared counter
    
    IO_BUFFER_SIZE = calculate_optimal_buffer_size(num_workers, num_buckets)
    
    print(f"[Pass 1] Input size: {file_size / 1024**4:.2f} TB. Approx chunk size: {chunk_size / 1024**3:.2f} GB.")

    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        futures = []
        for i in range(num_workers):
            start_byte = i * chunk_size
            end_byte = min((i + 1) * chunk_size, file_size) # Ensure end_byte doesn't exceed file size

            # Avoid submitting tasks for empty ranges if file_size < chunk_size * i
            if start_byte < file_size:
                futures.append(executor.submit(partition_chunk, i, input_file, temp_dir, num_buckets, start_byte, end_byte, total_lines_processed, IO_BUFFER_SIZE))

        # Collect results; worker errors will propagate here and halt execution
        for future in as_completed(futures):
            worker_files = future.result() # Raises exception if worker failed
            all_bucket_files.extend(worker_files)

    elapsed = time.time() - start_time
    final_total_lines = total_lines_processed.value
    print(f"[Pass 1] Finished partitioning. Found {final_total_lines:,} total lines. [{elapsed:.2f} s]")

    if not all_bucket_files and file_size > 0:
         # This check might be redundant if worker errors propagate, but harmless
         raise RuntimeError("Pass 1 did not produce any bucket files despite non-empty input.")
    return all_bucket_files, final_total_lines # Return line count too

def process_logical_bucket(logical_bucket_index, input_filenames, train_filename, eval_filename, lock, total_lines_written_shared, eval_lines_written_shared, max_output_lines, eval_size):
    """
    Pass 2 Worker: Reads all temporary files for a single logical bucket,
    shuffles the combined lines in memory, and appends to train and eval files under a lock.
    Stops writing to eval after eval_size is reached and stops overall if max_output_lines is reached.
    Deletes input files after processing.
    """
    pid = os.getpid()
    print(f"  [Pass 2 - Worker {pid}] Processing logical bucket {logical_bucket_index:04d} ({len(input_filenames)} files)...")
    start_time = time.time()
    all_lines = []
    total_size_gb = 0

    # Read all input bucket files
    for fname in input_filenames:
        fsize = os.path.getsize(fname)
        total_size_gb += fsize / 1024**3
        with open(fname, "r", encoding="utf-8") as f:
            all_lines.extend(f.readlines())

    # Delete input files after reading to free space
    for fname in input_filenames:
        try:
            os.remove(fname)
            print(f"  [Pass 2 - Worker {pid}] Deleted temp file {fname}")
        except Exception as e:
            print(f"  [Pass 2 - Worker {pid}] Warning: Failed to delete {fname}: {e}")

    if not all_lines:
        print(f"  [Pass 2 - Worker {pid}] Logical bucket {logical_bucket_index:04d} has no lines. Skipping.")
        return 0, 0  # Return lines written to train and eval

    num_lines = len(all_lines)
    print(f"  [Pass 2 - Worker {pid}] Read {num_lines:,} lines ({total_size_gb:.2f} GB) for bucket {logical_bucket_index:04d}. Shuffling...")

    # Shuffle combined lines in memory
    random.shuffle(all_lines)

    # Write to train and eval files based on current counts, under lock
    train_lines_written = 0
    eval_lines_written = 0
    with lock:
        current_total = total_lines_written_shared.value
        current_eval = eval_lines_written_shared.value
        remaining_total_quota = max(0, max_output_lines - current_total)
        
        lines_to_write = all_lines[:remaining_total_quota]
        if lines_to_write:
            with open(eval_filename, "a", encoding="utf-8") as eval_out, \
                 open(train_filename, "a", encoding="utf-8") as train_out:
                for line in lines_to_write:
                    if current_eval < eval_size:
                        eval_out.write(line)
                        current_eval += 1
                        eval_lines_written += 1
                    else:
                        train_out.write(line)
                        train_lines_written += 1
            total_lines_written_shared.value += (train_lines_written + eval_lines_written)
            eval_lines_written_shared.value += eval_lines_written

    elapsed = time.time() - start_time
    print(f"  [Pass 2 - Worker {pid}] Finished logical bucket {logical_bucket_index:04d}. Wrote {train_lines_written:,} train lines, {eval_lines_written:,} eval lines. [{elapsed:.2f} s]")
    return train_lines_written, eval_lines_written


def shuffle_and_merge_buckets(all_bucket_filenames, train_filename, eval_filename, num_logical_buckets, max_workers, max_output_lines, eval_size):
    """
    Orchestrates Pass 2: Groups temporary bucket files by logical bucket index,
    then processes each logical bucket in parallel with truncation and train-eval split.
    """
    print("\n[Pass 2] Starting shuffling and merging of logical buckets...")
    start_time = time.time()

    # Group filenames by logical bucket index
    grouped_buckets = [[] for _ in range(num_logical_buckets)]
    for fname in all_bucket_filenames:
        basename = os.path.basename(fname)
        parts = basename.split('_')  # e.g., "w01_bucket_0042.tmp"
        logical_index = int(parts[2].split('.')[0])  # "0042" -> 42
        if 0 <= logical_index < num_logical_buckets:
            grouped_buckets[logical_index].append(fname)
        else:
            raise ValueError(f"Parsed invalid logical bucket index {logical_index} from filename {fname}")

    tasks = [(idx, files) for idx, files in enumerate(grouped_buckets) if files]
    num_tasks = len(tasks)
    if num_tasks == 0 and len(all_bucket_filenames) > 0:
        raise RuntimeError("Pass 1 produced files, but grouping yielded no tasks for Pass 2.")

    print(f"[Pass 2] Grouped files into {num_tasks} non-empty logical buckets.")

    # Ensure train and eval output files are empty/created before workers append
    with open(train_filename, "w", encoding="utf-8"):
        pass
    with open(eval_filename, "w", encoding="utf-8"):
        pass

    print(f"[Pass 2] Processing {num_tasks} logical buckets using up to {max_workers} workers...")
    manager = Manager()
    lock = manager.Lock()
    total_lines_written_shared = manager.Value('L', 0)  # Shared counter for total lines
    eval_lines_written_shared = manager.Value('L', 0)  # Shared counter for eval lines
    total_train_lines_written = 0
    total_eval_lines_written = 0
    processed_bucket_count = 0

    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(process_logical_bucket, idx, files, train_filename, eval_filename, lock, total_lines_written_shared, eval_lines_written_shared, max_output_lines, eval_size): idx for idx, files in tasks}

        for future in as_completed(futures):
            train_lines, eval_lines = future.result()
            total_train_lines_written += train_lines
            total_eval_lines_written += eval_lines
            processed_bucket_count += 1
            if processed_bucket_count % (num_tasks // 20 or 1) == 0 or processed_bucket_count == num_tasks:
                elapsed_pass2 = time.time() - start_time
                percent_done = (processed_bucket_count / num_tasks) * 100
                print(f"  [Pass 2] Progress: {processed_bucket_count}/{num_tasks} buckets merged ({percent_done:.1f}%) [{elapsed_pass2:.2f} s]")

    elapsed = time.time() - start_time
    print(f"\n[Pass 2] Finished merging {processed_bucket_count}/{num_tasks} logical buckets.")
    print(f"[Pass 2] Time taken: {elapsed:.2f} seconds.")
    print(f"[Pass 2] Total train lines written to '{os.path.basename(train_filename)}': {total_train_lines_written:,}")
    print(f"[Pass 2] Total eval lines written to '{os.path.basename(eval_filename)}': {total_eval_lines_written:,}")
    return total_train_lines_written, total_eval_lines_written

# --- Main Execution Logic ---
def main():
    # Set random seed once at the start for replicability
    random.seed(RANDOM_SEED)

    overall_start_time = time.time()
    temp_dir_path = None  # Initialize to None

    try:
        SCRATCH_DIR, PROJECT_DIR, INPUT_FILE, TRAIN_SIZE, EVAL_SIZE = parse_arguments()
        OUTPUT_DIR = os.path.join(PROJECT_DIR, "output")
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        
        data_stem = Path(INPUT_FILE).stem
        TRAIN_OUTPUT_FILE = os.path.join(OUTPUT_DIR, f"{data_stem}_train.jsonl")
        EVAL_OUTPUT_FILE  = os.path.join(OUTPUT_DIR, f"{data_stem}_eval.jsonl")
        TEMP_DIR_BASE = PROJECT_DIR  # Base directory for temp folder in project space
        EVAL_SIZE = 20000  # Define eval set size

        print("="*60)
        print("Starting Bucket Shuffling Process")
        print(f"Input File:  {INPUT_FILE}")
        print(f"Train Output File: {TRAIN_OUTPUT_FILE}")
        print(f"Eval Output File: {EVAL_OUTPUT_FILE}")
        print(f"Temp Base:   {TEMP_DIR_BASE}")
        print(f"Buckets:     {NUM_BUCKETS}")
        print(f"Pass 1 Wkrs: {MAX_WORKERS_PASS1}")
        print(f"Pass 2 Wkrs: {MAX_WORKERS_PASS2}")
        print(f"Train Set Size: {TRAIN_SIZE:,}")
        print(f"Eval Set Size: {EVAL_SIZE:,}")
        print(f"Random Seed: {RANDOM_SEED}")
        print("="*60)

        # Create a unique temporary directory for this run's buckets in project space
        temp_dir_path = tempfile.mkdtemp(prefix="shuffle_buckets_", dir=TEMP_DIR_BASE)
        print(f"Created temporary directory for buckets: {temp_dir_path}")
        print("Note: This directory will be automatically cleaned up after processing.")

        # === Pass 1: Parallel Partitioning ===
        all_bucket_filenames, pass1_lines = run_parallel_partition(
            INPUT_FILE, temp_dir_path, NUM_BUCKETS, MAX_WORKERS_PASS1
        )

        # === Pass 2: Shuffling & Merging with Train-Eval Split ===
        train_lines, eval_lines = shuffle_and_merge_buckets(
            all_bucket_filenames, TRAIN_OUTPUT_FILE, EVAL_OUTPUT_FILE, NUM_BUCKETS, MAX_WORKERS_PASS2, TRAIN_SIZE, EVAL_SIZE
        )

        print("\n--- Shuffle and split process completed ---")

        # Final Verification
        print("Verification:")
        print(f"  Lines found in Pass 1: {pass1_lines:,}")
        print(f"  Lines written in Pass 2 (total): {train_lines + eval_lines:,}")
        print(f"  Lines written to train: {train_lines:,}")
        print(f"  Lines written to eval: {eval_lines:,}")
        if pass1_lines != (train_lines + eval_lines):
            print("!! WARNING: Line count mismatch between Pass 1 and Pass 2!")
            print("!! This could indicate data loss or duplication. Please investigate.")
        else:
            print("  Line counts match between passes.")

    except Exception as e:
        # Catch any exception that wasn't handled within the functions
        print("\n--- Shuffle process FAILED ---")
        print(f"Error: {e}")
        print("\n--- Traceback ---")
        traceback.print_exc()
        print("----------------")
        # Indicate failure with non-zero exit code
        exit_code = 1
    else:
        exit_code = 0
    
    finally:
        # Clean up temporary directory
        if temp_dir_path and os.path.exists(temp_dir_path):
            print(f"\nCleaning up temporary directory: {temp_dir_path}")
            try:
                shutil.rmtree(temp_dir_path)
                print("Temporary directory removed successfully.")
            except Exception as cleanup_error:
                print(f"Warning: Failed to remove temporary directory: {cleanup_error}")
                print(f"You may need to manually remove: {temp_dir_path}")

        overall_elapsed = time.time() - overall_start_time
        hours, rem = divmod(overall_elapsed, 3600)
        minutes, seconds = divmod(rem, 60)
        print(f"\nTotal execution time: {int(hours)} hours, {int(minutes)} minutes, {int(seconds)} seconds.")

        # Exit with appropriate code
        exit(exit_code)

if __name__ == "__main__":
    main()
    