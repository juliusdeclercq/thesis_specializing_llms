# -*- coding: utf-8 -*-
"""
Created on Fri Jun 20 17:03:01 2025

@author: Julius de Clercq


This script separates a validation set from the training data for domain adaptation.
Note that eval is a misnomer here, as it actually means validation set. 
"""

import argparse
import json
import os
from pathlib import Path

def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Split eval dataset and append to train.")
    parser.add_argument("--eval_file", type=str, required=True, help="Path to the eval JSONL file.")
    parser.add_argument("--train_file", type=str, required=True, help="Path to the train JSONL file.")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory for output files.")
    parser.add_argument("--eval_keep_size", type=int, default=10000, help="Number of examples to keep in eval set.")
    return parser.parse_args()

def split_and_append(args):
    """Read eval file, split it, and append to train file."""
    eval_keep_size = args.eval_keep_size
    eval_file = args.eval_file
    train_file = args.train_file
    output_dir = args.output_dir

    # Ensure output directory exists
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    # Temporary output paths
    new_eval_path = os.path.join(output_dir, "new_eval_domain_adaptation_data.jsonl")
    temp_append_path = os.path.join(output_dir, "temp_append_to_train.jsonl")

    print(f"Processing eval file: {eval_file}")
    print(f"Keeping {eval_keep_size} examples in eval, moving rest to train.")

    # Step 1: Split eval file into new eval (keep) and temp append file
    eval_total = 0
    with open(eval_file, 'r', encoding='utf-8') as f_in, \
         open(new_eval_path, 'w', encoding='utf-8') as f_eval, \
         open(temp_append_path, 'w', encoding='utf-8') as f_append:
        for line in f_in:
            eval_total += 1
            if eval_total <= eval_keep_size:
                f_eval.write(line)
            else:
                f_append.write(line)
    print(f"Processed {eval_total} eval examples. Kept {eval_keep_size}, moved {eval_total - eval_keep_size} to temp append file.")

    # Step 2: Append temp file to existing train file
    new_train_path = os.path.join(output_dir, "new_train_domain_adaptation_data.jsonl")
    print(f"Appending to train file: {train_file}")
    lines_appended = 0
    with open(train_file, 'r', encoding='utf-8') as f_train, \
         open(temp_append_path, 'r', encoding='utf-8') as f_append, \
         open(new_train_path, 'w', encoding='utf-8') as f_out:
        # Write all original train lines
        for line in f_train:
            f_out.write(line)
        # Append the moved eval lines
        for line in f_append:
            f_out.write(line)
            lines_appended += 1
    print(f"Appended {lines_appended} examples to new train file.")

    # Step 3: Cleanup temporary file
    os.remove(temp_append_path)
    print(f"Cleaned up temporary file: {temp_append_path}")
    print(f"New eval file saved to: {new_eval_path}")
    print(f"New train file saved to: {new_train_path}")

if __name__ == "__main__":
    args = parse_args()
    split_and_append(args)