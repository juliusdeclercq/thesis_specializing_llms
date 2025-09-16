# -*- coding: utf-8 -*-
"""
Created on Thu Jul 10 01:13:23 2025

@author: Julius de Clercq
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
from tqdm import tqdm
# import fcntl      # Package with a different lock that is multi-node safe


#%% Argument parser
def parse_arguments():
    """
    Handles input year argument for cluster execution.
    """
    parser = argparse.ArgumentParser(description="Subset SEC data based on form type.")
    parser.add_argument('input_year', type=str, help="Input directory year.")
    input_year = parser.parse_args().input_year
    
    return int(input_year)


#%% Helper function for writing batches safely with file locking
def write_batch(output_jsonl_path, lines_batch, lock):
    """ 
    Safely appends a batch of lines using file-level locking.
    This ensures multiple processes writing to the same file don't corrupt it.
    """
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
        raise e


#%% Worker Function
def subset_tar_data(input_tar_path, output_jsonl_path, whitelist, filename_pattern, batch_char_limit, lock):
    """
    Processes a single tar file, extracts whitelisted filings, and appends them
    to the output JSONL file in batches based on character count.
    Note: lock parameter removed as we use file-level locking now.
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
                    with tar.extractfile(member) as file:
                        if file is not None:
                            content = file.read().decode("utf-8", errors="ignore")
                            record = {"filing": content}
                            json_line = json.dumps(record) + "\n"
                            line_len = len(json_line)

                            lines_to_write.append(json_line)
                            current_batch_chars += line_len

                            if current_batch_chars >= batch_char_limit:
                                written_count = write_batch(output_jsonl_path, lines_to_write, lock)
                                total_lines_written += written_count
                                lines_to_write = []
                                current_batch_chars = 0

    # Write final batch
    if lines_to_write:
        written_count = write_batch(output_jsonl_path, lines_to_write, lock)
        total_lines_written += written_count

    return total_lines_written


#%% Orchestration Function
def compile_domain_adaptation_dataset(tar_files_paths, output_jsonl_path, whitelist_path, active_cores, batch_char_limit):
    """
    Orchestrates the parallel processing using ProcessPoolExecutor.
    No Manager().Lock() needed - using file-level locking instead.
    """
    total_lines_output = 0
    whitelist = set(pd.read_pickle(whitelist_path))
    filename_pattern = re.compile(r"^\d{8}_(.*?)_\d+_\d+\.txt$")

    # Ensure output directory exists
    os.makedirs(os.path.dirname(output_jsonl_path), exist_ok=True)
    
    # Initialize multiprocessing lock 
    lock = Manager().Lock()
    
    # Create output file if it doesn't exist (don't truncate for parallel runs)
    if not os.path.exists(output_jsonl_path):
        with open(output_jsonl_path, "w", encoding="utf-8"):
            pass

    print(f"Starting processing of {len(tar_files_paths)} tar files using {active_cores} cores.")
    batch_limit_mib = batch_char_limit / (1024 * 1024)
    print(f"Using batch character limit: ~{batch_limit_mib:.1f} MiB")

    with ProcessPoolExecutor(max_workers=active_cores) as executor:
        futures = {
            executor.submit(
                subset_tar_data, 
                tar_path, 
                output_jsonl_path, 
                whitelist, 
                filename_pattern, 
                batch_char_limit,
                lock
            ): tar_path for tar_path in tar_files_paths
        }

        # Use tqdm to show progress
        with tqdm(total=len(tar_files_paths), desc="Processing tar files", unit="files") as pbar:
            for future in as_completed(futures):
                tar_path = futures[future]
                try:
                    lines_written = future.result()
                    total_lines_output += lines_written
                    pbar.update(1)
                except Exception as e:
                    print(f"\nERROR: Worker process for {os.path.basename(tar_path)} failed: {e}")
                    pbar.update(1)  # Still update progress even on error

    print("\n--- Processing Summary ---")
    print(f"Attempted processing {len(tar_files_paths)} tar files.")
    print(f"Total lines written to output file: {total_lines_output}")
    print("-------------------------\n")


#%% Main Execution Block
def main():
    start = t()

    # --- Configuration ---
    write_batch_char_limit = 650 * 1024 * 1024 # limiting to avoid OOM errors
    # --- End Configuration ---

    # Get input year
    year = parse_arguments()
    
    # Get environment variables
    scratch_dir = os.environ["SCRATCH_DIR"]
    # project_dir = os.environ["PROJECT_DIR"]
    active_cores = int(os.environ["SLURM_CPUS_PER_TASK"])
    
    print(f"Running for year {year}.")
    
    # Set paths
    paths = {
        "whitelist_path": os.path.join(scratch_dir, "whitelist.pkl"),
        "input_path": os.path.join(scratch_dir, "intermediate", str(year), "filings"),
        "output_jsonl_path": os.path.join(scratch_dir, "output", f"subsetted_data_{year}.jsonl")
    }

    # Get list of tar files
    all_tar_files = [f for f in os.listdir(paths["input_path"]) if f.endswith(".tar")]
    tar_files_full_paths = [os.path.join(paths["input_path"], f) for f in all_tar_files]

    if not tar_files_full_paths:
        print("No tar files found for processing.")
        return

    # --- Execution ---
    print(f"Output will be written to: {paths['output_jsonl_path']}")
    print(f"Using {active_cores} cores.")

    compile_domain_adaptation_dataset(
        tar_files_paths=tar_files_full_paths,
        output_jsonl_path=paths["output_jsonl_path"],
        whitelist_path=paths["whitelist_path"],
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