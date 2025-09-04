# -*- coding: utf-8 -*-
"""
Created on Mon Apr  7 12:55:50 2025

@author: Julius de Clercq

The goal of this script is to subset the textual data, which is stored in ~3300
.tar-files, using only whitelisted form types (e.g. 10-Q, 10-K). The subset is
to be stored in one .jsonl file, with one line for each filing. This will contain
about 3.6 million filings, thus 3.6M lines in the file. The filings must have a
randomized permutation, which is to be achieved using the `shuf` Unix-command in
the SLURM terminal of the Snellius compute cluster, so that does not have to be
done here.

Simplified version focusing on core multiprocessing logic with batching.
Assumes input data integrity and path validity. Minimal error handling.
"""

# Imports
import pandas as pd
from pathlib import Path
from time import time as t
from multiprocessing import Manager
from concurrent.futures import ProcessPoolExecutor, as_completed
import argparse
import tarfile
import json
import re
import os


#%%             Argument parser
def parse_arguments():
    """
    Handles input year and scratch directory arguments for cluster execution,
    with a fallback for local testing.
    """
    parser = argparse.ArgumentParser(description="Subset SEC data based on form type.")
    parser.add_argument('input_year', nargs='?', type=str, help="Input directory year.")
    parser.add_argument('scratch_dir', nargs='?', type=str, help="Path to the scratch-shared directory.")
    args = parser.parse_args()

    input_year_str = ""
    scratch_dir = ""

    try:
        # Attempt to use provided arguments
        if args.input_year and args.scratch_dir:
            input_year_str = args.input_year
            int(input_year_str) # Validate year format early
            scratch_dir = args.scratch_dir
        else:
            raise ValueError("Incomplete arguments") # Trigger local mode
    except Exception:
        print("\nArguments not passed or invalid. Running in local test mode.\n")
        # Keep input_year_str and scratch_dir as "" for local mode detection in main()

    return input_year_str, scratch_dir


#%% Helper function for writing batches safely
def write_batch(output_jsonl_path, lock, lines_batch):
    """ Safely appends a batch of lines using a lock. Minimal error handling. """
    if not lines_batch:
        return 0
    try:
        with lock:
            with open(output_jsonl_path, "a", encoding="utf-8") as jsonl_file:
                jsonl_file.writelines(lines_batch)
        return len(lines_batch)
    except Exception as e:
        # Log critical write errors potentially related to locking or filesystem issues
        print(f"CRITICAL WRITE ERROR: {e} while writing batch to {output_jsonl_path}. Data loss possible.")
        # Depending on severity, might want os._exit(1) here if recovery isn't possible
        return 0 # Indicate failure

#%% Worker Function
def subset_tar_data(input_tar_path, output_jsonl_path, lock, whitelist, filename_pattern, batch_char_limit):
    """
    Processes a single tar file, extracts whitelisted filings, and appends them
    to the output JSONL file in batches based on character count.
    Minimal error handling, assumes data validity.
    """
    lines_to_write = []
    current_batch_chars = 0
    total_lines_written = 0

    
    with tarfile.open(input_tar_path, "r") as tar:
       for member in tar.getmembers():
           if member.isfile():
               pattern_match = filename_pattern.search(member.name)
               form_type = pattern_match.group(1) if pattern_match else None

               if form_type in whitelist:
                   # Assume member is extractable and content decodable/serializable
                   with tar.extractfile(member) as file:
                       if file is not None:
                           content = file.read().decode("utf-8", errors="ignore")
                           record = {"filing": content}
                           json_line = json.dumps(record) + "\n"
                           line_len = len(json_line)

                           lines_to_write.append(json_line)
                           current_batch_chars += line_len

                           if current_batch_chars >= batch_char_limit:
                               written_count = write_batch(output_jsonl_path, lock, lines_to_write)
                               total_lines_written += written_count
                               lines_to_write = []
                               current_batch_chars = 0

    # Write final batch
    if lines_to_write:
        written_count = write_batch(output_jsonl_path, lock, lines_to_write)
        total_lines_written += written_count

    return total_lines_written

#%% Orchestration Function
def compile_domain_adaptation_dataset(tar_files_paths, output_jsonl_path, whitelist_path, active_cores, batch_char_limit):
    """
    Orchestrates the parallel processing using ProcessPoolExecutor.
    Minimal error reporting.
    """
    total_lines_output = 0
    # Assume whitelist exists and is readable
    whitelist = set(pd.read_pickle(whitelist_path))
    filename_pattern = re.compile(r"^\d{8}_(.*?)_\d+_\d+\.txt$")

    lock = Manager().Lock()

    # Ensure output directory exists and clear/create output file
    if not os.path.exists(output_jsonl_path): # Initialize output file if it does not yet exist.
        os.makedirs(os.path.dirname(output_jsonl_path), exist_ok=True)
        with open(output_jsonl_path, "w", encoding="utf-8"):
             pass # Create/truncate

    print(f"Starting processing of {len(tar_files_paths)} tar files using {active_cores} cores.")
    batch_limit_mib = batch_char_limit / (1024 * 1024)
    print(f"Using batch character limit: ~{batch_limit_mib:.1f} MiB")

    processed_files = 0
    with ProcessPoolExecutor(max_workers=active_cores) as executor:
        futures = {executor.submit(subset_tar_data, tar_path, output_jsonl_path, lock, whitelist, filename_pattern, batch_char_limit): tar_path
                   for tar_path in tar_files_paths}

        for future in as_completed(futures):
            tar_path = futures[future]
            try:
                lines_written = future.result()
                total_lines_output += lines_written
                processed_files += 1
                # Minimal progress update
                if processed_files % 50 == 0: # Print every 50 files
                     print(f"  Processed {processed_files}/{len(tar_files_paths)} files...")
            except Exception as e:
                # Catch errors raised *from* the worker process (e.g., unhandled exceptions there)
                 print(f"ERROR: Worker process for {os.path.basename(tar_path)} failed: {e}")
                 # Continue processing other files

    print("\n--- Processing Summary ---")
    print(f"Attempted processing {len(tar_files_paths)} tar files.")
    print(f"Total lines written to output file: {total_lines_output}")
    print("-------------------------\n")


#%% Main Execution Block
def main():
    start = t()

    # --- Configuration ---
    input_limit = None # Set to integer for testing, None for all files
    default_year = 2012
    write_batch_char_limit = 650 * 1024 * 1024
    # --- End Configuration ---

    paths = {}
    year_str, scratch_dir = parse_arguments()
    

    # Determine execution mode and set paths
    if not year_str:  # Local test mode
        year = default_year
        print(f"Running in local mode for year {year}.")
        # Assume script is in 'scripts' folder, parent is project root
        project_root = Path(os.getcwd()).parent
        paths["whitelist_path"] = project_root / "subsetting domain adaptation data" / "whitelist.pkl"
        paths["input_path"] = project_root / "intermediate" / str(year) / "filings"
        paths["output_jsonl_path"] = project_root / "output" / "domain_adaptation_data.jsonl"
        max_cores = os.cpu_count()
        active_cores = max_cores - 1 if max_cores > 1 else 1
        
    else: # Cluster mode
        year = int(year_str)
        print(f"Running in cluster mode for year {year}.")
        # Assume scratch_dir is valid
        paths["whitelist_path"] = os.path.join(scratch_dir, "whitelist.pkl")
        paths["input_path"] = os.path.join(os.environ["PROJECT_DIR"], "intermediate", str(year), "filings")
        paths["output_jsonl_path"] = os.path.join(scratch_dir, "output", "domain_adaptation_data.jsonl")
        active_cores = int(os.environ["SLURM_CPUS_PER_TASK"])

    if input_limit:
        active_cores = min(active_cores, input_limit) # Don't use more cores than files if limited

    # --- Prepare list of tar files ---
    # Assume input path exists and contains .tar files
    all_tar_files = [f for f in os.listdir(paths["input_path"]) if f.endswith(".tar")]
    if input_limit:
        tar_files_to_process = all_tar_files[:input_limit]
        print(f"Limiting processing to the first {len(tar_files_to_process)} tar files.")
    else:
        tar_files_to_process = all_tar_files

    tar_files_full_paths = [os.path.join(paths["input_path"], f) for f in tar_files_to_process]

    if not tar_files_full_paths:
        print("No tar files found or selected for processing.")
        return

    # --- Execution ---
    print(f"Output will be written to: {paths['output_jsonl_path']}")
    print(f"Using {active_cores} cores.")

    compile_domain_adaptation_dataset(
        tar_files_paths=tar_files_full_paths,
        output_jsonl_path=str(paths["output_jsonl_path"]), # Ensure string path
        whitelist_path=str(paths["whitelist_path"]),   # Ensure string path
        active_cores=active_cores,
        batch_char_limit=write_batch_char_limit
    )

    # --- Timing ---
    end = t()
    running_time = end - start
    num_processed = len(tar_files_full_paths)
    print(f"\nProcessing {num_processed} tar-files took {running_time // 60:.0f} min {running_time % 60:.2f} seconds.\n\n")


if __name__ == "__main__":
    main()