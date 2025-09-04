# -*- coding: utf-8 -*-
"""
Created on Thu Jun 19 11:35:35 2025

@author: Julius de Clercq
"""

import argparse
from pathlib import Path
from tqdm import tqdm
import os

def main():
    # Parse arguments
    parser = argparse.ArgumentParser(description="Append worker chunk files to an existing JSONL file.")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory containing remaining worker chunk files.")
    parser.add_argument("--num_workers", type=int, default=96, help="Number of worker files to check (default: 96 for 0000-0095).")
    parser.add_argument("--final_output", type=str, default="tokenized_domain_adapt_data.jsonl", help="Name of the existing merged output file to append to.")
    args = parser.parse_args()

    # Define output path for merged file
    output_path = Path(args.output_dir) / args.final_output
    print(f"Appending chunk files to {output_path}...")

    # Build list of worker file paths (only include existing files)
    worker_results = []
    for idx in range(args.num_workers):
        worker_idx = f"{idx:04d}"
        scratch_file = Path(args.output_dir) / f"worker_{worker_idx}_chunks.jsonl"
        if scratch_file.exists():
            worker_results.append((idx, scratch_file))

    # Sort by worker_idx to maintain order
    worker_results.append((0, Path(args.output_dir) / "tokenized_domain_adapt.jsonl")) # !!!!!!!!!!!
    worker_results.sort(key=lambda x: x[0])

    if not worker_results:
        print("Error: No remaining worker files found to append!")
        exit(1)

    # Append remaining JSONL files to the existing output file
    total_size = 0
    
    with open(output_path, "ab") as outfile:  # 'ab' mode to append binary
        for worker_idx, file_path in tqdm(worker_results, desc="Appending files"):
            file_size = file_path.stat().st_size
            total_size += file_size
            with open(file_path, "rb") as infile:
                while True:
                    buffer = infile.read(1024 ** 3)  # Read 1 MB at a time
                    if not buffer:
                        break
                    outfile.write(buffer)
            # Delete intermediate file to free space
            file_path.unlink()
            print(f"Deleted intermediate file: {file_path}")

    print(f"\nAppended data to: {output_path}")
    print(f"Total file size appended: {total_size / (1024**3):.2f} GB")

if __name__ == "__main__":
    main()