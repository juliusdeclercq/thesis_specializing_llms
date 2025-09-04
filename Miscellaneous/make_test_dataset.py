# -*- coding: utf-8 -*-
"""
Created on Thu Apr 17 14:18:52 2025

@author: Julius de Clercq
"""

import os
import argparse 

def parse_arguments():
    parser = argparse.ArgumentParser(description="Make a small test sample .")
    parser.add_argument('--scratch_dir', type=str, required=True, help="Path to the scratch directory.")
    parser.add_argument('--input_path', type=str, required=True, help="Path to the scratch directory.")
    parser.add_argument('--lines', type=int, required=True, help="Number of lines to sample.")
    args = parser.parse_args()
    return args

# --- Configuration ---
args = parse_arguments()
scratch_dir = args.scratch_dir
input_path = args.input_path
lines = args.lines
input_file_path = os.path.join(input_path)
output_file_path = os.path.join(scratch_dir, "data", f"test_{lines}.jsonl")

# --- Configuration ---

print(f"Reading first {lines} lines from: {input_file_path}")
print(f"Writing to: {output_file_path}")

# Use 'with open' to ensure files are closed automatically
with open(input_file_path, 'r', encoding='utf-8') as infile, \
     open(output_file_path, 'w', encoding='utf-8') as outfile:

    for i, line in enumerate(infile):
        if i >= lines:
            break # Stop after reading the desired number of lines
        outfile.write(line)

print(f"Finished creating subset file: {output_file_path}")