# -*- coding: utf-8 -*-
"""
Created on Tue Apr 15 11:44:39 2025

@author: Julius de Clercq
"""

import random
import os
import tempfile
import time
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from multiprocessing import Manager
import math
import traceback # Keep for printing tracebacks on error


# --- Configuration ---
NUM_BUCKETS = 64
MAX_WORKERS_PASS1 = 32
MAX_WORKERS_PASS2 = 6
RANDOM_SEED = 98765432123456789
IO_BUFFER_SIZE = 64 * 1024 * 1024 # 64 MiB buffer for reads/writes
# --- End Configuration ---

# Global variable for file size (set in run_parallel_partition)
file_size = 0

def parse_arguments():
    """Parses command-line arguments."""
    parser = argparse.ArgumentParser(description="Shuffle large JSONL file.")
    parser.add_argument('scratch_dir', type=str, help="Path to the scratch directory containing input/output.")
    args = parser.parse_args()
    return args.scratch_dir

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

def partition_chunk(worker_id, input_file, temp_dir, num_buckets, start_byte, end_byte, total_lines_counter):
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

    print(f"[Pass 1] Input size: {file_size / 1024**4:.2f} TB. Approx chunk size: {chunk_size / 1024**3:.2f} GB.")

    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        futures = []
        for i in range(num_workers):
            start_byte = i * chunk_size
            end_byte = min((i + 1) * chunk_size, file_size) # Ensure end_byte doesn't exceed file size

            # Avoid submitting tasks for empty ranges if file_size < chunk_size * i
            if start_byte < file_size:
                futures.append(executor.submit(partition_chunk, i, input_file, temp_dir, num_buckets, start_byte, end_byte, total_lines_processed))

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

def process_logical_bucket(logical_bucket_index, input_filenames, output_filename, lock):
    """
    Pass 2 Worker: Reads all temporary files for a single logical bucket,
    shuffles the combined lines in memory, and appends them to the final
    output file under a lock. Does NOT delete input bucket files.
    """
    pid = os.getpid()
    print(f"  [Pass 2 - Worker {pid}] Processing logical bucket {logical_bucket_index:04d} ({len(input_filenames)} files)...")
    start_time = time.time()
    all_lines = []
    total_size_gb = 0

    # Read all input bucket files; errors (e.g., FileNotFoundError) will halt the worker
    for fname in input_filenames:
        fsize = os.path.getsize(fname) # Raises FileNotFoundError if missing
        total_size_gb += fsize / 1024**3
        with open(fname, "r", encoding="utf-8", buffering=IO_BUFFER_SIZE) as f:
            all_lines.extend(f.readlines()) # Raises MemoryError if too large

    if not all_lines:
        print(f"  [Pass 2 - Worker {pid}] Logical bucket {logical_bucket_index:04d} has no lines. Skipping.")
        return 0 # Return 0 lines written

    num_lines = len(all_lines)
    print(f"  [Pass 2 - Worker {pid}] Read {num_lines:,} lines ({total_size_gb:.2f} GB) for bucket {logical_bucket_index:04d}. Shuffling...")

    # Shuffle combined lines in memory
    random.shuffle(all_lines)

    print(f"  [Pass 2 - Worker {pid}] Writing {num_lines:,} lines for bucket {logical_bucket_index:04d}...")
    # Append shuffled lines to the single output file (lock ensures serial writes)
    with lock:
        with open(output_filename, "a", encoding="utf-8", buffering=IO_BUFFER_SIZE) as outfile:
            outfile.writelines(all_lines)

    elapsed = time.time() - start_time
    print(f"  [Pass 2 - Worker {pid}] Finished logical bucket {logical_bucket_index:04d}. Wrote {num_lines:,} lines. [{elapsed:.2f} s]")
    return num_lines


def shuffle_and_merge_buckets(all_bucket_filenames, output_filename, num_logical_buckets, max_workers):
    """
    Orchestrates Pass 2: Groups temporary bucket files by logical bucket index,
    then processes each logical bucket (read, shuffle, write) in parallel
    using ProcessPoolExecutor.
    """
    print("\n[Pass 2] Starting shuffling and merging of logical buckets...")
    start_time = time.time()

    # Group filenames by logical bucket index based on filename convention
    # Errors in parsing (ValueError, IndexError) will halt execution here
    grouped_buckets = [[] for _ in range(num_logical_buckets)]
    for fname in all_bucket_filenames:
        basename = os.path.basename(fname)
        parts = basename.split('_') # e.g., "w01_bucket_0042.tmp"
        logical_index = int(parts[2].split('.')[0]) # "0042" -> 42
        if 0 <= logical_index < num_logical_buckets:
            grouped_buckets[logical_index].append(fname)
        else:
             # This indicates a logic error or unexpected file
             raise ValueError(f"Parsed invalid logical bucket index {logical_index} from filename {fname}")

    # Create list of tasks (non-empty buckets)
    tasks = [(idx, files) for idx, files in enumerate(grouped_buckets) if files]
    num_tasks = len(tasks)
    if num_tasks == 0 and len(all_bucket_filenames) > 0:
        raise RuntimeError("Pass 1 produced files, but grouping yielded no tasks for Pass 2.")

    print(f"[Pass 2] Grouped files into {num_tasks} non-empty logical buckets.")

    # Ensure output file is empty/created before workers append to it
    with open(output_filename, "w", encoding="utf-8") as f:
        pass # Truncates the file if it exists

    print(f"[Pass 2] Processing {num_tasks} logical buckets using up to {max_workers} workers...")
    manager = Manager()
    lock = manager.Lock() # Lock for append writes to the single output file
    total_lines_written = 0
    processed_bucket_count = 0

    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        # Submit tasks: process_logical_bucket(logical_idx, list_of_filenames, output, lock)
        futures = {executor.submit(process_logical_bucket, idx, files, output_filename, lock): idx for idx, files in tasks}

        # Collect results; worker errors propagate here
        for future in as_completed(futures):
            logical_idx = futures[future] # Get associated task index
            lines_written = future.result() # Raises exception if worker failed
            total_lines_written += lines_written
            processed_bucket_count += 1

            # Progress reporting
            if processed_bucket_count % (num_tasks // 20 or 1) == 0 or processed_bucket_count == num_tasks:
                 elapsed_pass2 = time.time() - start_time
                 percent_done = (processed_bucket_count / num_tasks) * 100
                 print(f"  [Pass 2] Progress: {processed_bucket_count}/{num_tasks} buckets merged ({percent_done:.1f}%) [{elapsed_pass2:.2f} s]")

    elapsed = time.time() - start_time
    print(f"\n[Pass 2] Finished merging {processed_bucket_count}/{num_tasks} logical buckets.")
    print(f"[Pass 2] Time taken: {elapsed:.2f} seconds.")
    print(f"[Pass 2] Total lines written to '{os.path.basename(output_filename)}': {total_lines_written:,}")
    return total_lines_written

# --- Main Execution Logic ---
if __name__ == "__main__":
    # Set random seed once at the start for replicability
    random.seed(RANDOM_SEED)

    overall_start_time = time.time()
    temp_dir_path = None # Initialize to None

    try:
        SCRATCH_DIR = parse_arguments()
        INPUT_FILE = os.path.join(SCRATCH_DIR, "output", "domain_adaptation_data.jsonl")
        OUTPUT_FILE = os.path.join(SCRATCH_DIR, "output", "shuffled_domain_adaptation_data.jsonl")
        TEMP_DIR_BASE = SCRATCH_DIR # Base directory for temp folder

        print("="*60)
        print("Starting Simplified Large File Shuffle Process")
        print(f"Input File:  {INPUT_FILE}")
        print(f"Output File: {OUTPUT_FILE}")
        print(f"Temp Base:   {TEMP_DIR_BASE}")
        print(f"Buckets:     {NUM_BUCKETS}")
        print(f"Pass 1 Wkrs: {MAX_WORKERS_PASS1}")
        print(f"Pass 2 Wkrs: {MAX_WORKERS_PASS2}")
        print(f"Random Seed: {RANDOM_SEED}")
        print("="*60)

        # Create a unique temporary directory for this run's buckets
        temp_dir_path = tempfile.mkdtemp(prefix="shuffle_buckets_", dir=TEMP_DIR_BASE)
        print(f"Created temporary directory for buckets: {temp_dir_path}")
        print("IMPORTANT: This directory will NOT be automatically cleaned up.")

        # === Pass 1: Parallel Partitioning ===
        all_bucket_filenames, pass1_lines = run_parallel_partition(
            INPUT_FILE, temp_dir_path, NUM_BUCKETS, MAX_WORKERS_PASS1
        )

        # === Pass 2: Shuffling & Merging ===
        pass2_lines = shuffle_and_merge_buckets(
            all_bucket_filenames, OUTPUT_FILE, NUM_BUCKETS, MAX_WORKERS_PASS2
        )

        print("\n--- Shuffle process completed ---")

        # Final Verification (Optional but recommended)
        print("Verification:")
        print(f"  Lines found in Pass 1: {pass1_lines:,}")
        print(f"  Lines written in Pass 2: {pass2_lines:,}")
        if pass1_lines != pass2_lines:
            print("!! WARNING: Line count mismatch between Pass 1 and Pass 2!")
            print("!! This could indicate data loss or duplication. Please investigate.")
        else:
            print("  Line counts match between passes.")

    except Exception as e:
        # Catch any exception that wasn't handled within the functions
        print(f"\n--- Shuffle process FAILED ---")
        print(f"Error: {e}")
        print("\n--- Traceback ---")
        traceback.print_exc() # Print detailed traceback
        print("----------------")
        # Indicate failure with non-zero exit code
        exit_code = 1
    else:
        # Runs only if the try block completes without exceptions
        exit_code = 0
    finally:
        # This block always runs, regardless of success or failure

        # --- No Automatic Cleanup ---
        if temp_dir_path and os.path.exists(temp_dir_path):
             print(f"\nTemporary bucket files remain in: {temp_dir_path}")
             print(f"Please verify the output file '{OUTPUT_FILE}' and manually delete the temporary directory when done.")
        elif temp_dir_path:
             print(f"\nTemporary directory {temp_dir_path} was expected but not found (or not created successfully).")


        overall_elapsed = time.time() - overall_start_time
        hours, rem = divmod(overall_elapsed, 3600)
        minutes, seconds = divmod(rem, 60)
        print(f"\nTotal execution time: {int(hours)} hours, {int(minutes)} minutes, {int(seconds)} seconds.")

        # Exit with appropriate code
        exit(exit_code)