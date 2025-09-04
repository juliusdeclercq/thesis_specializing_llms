# -*- coding: utf-8 -*-
"""
Created on Mon Jun 16 15:58:38 2025

@author: Julius de Clercq
"""

import argparse

def count_lines(path):
    with open(path, "rb") as f:
        return sum(1 for _ in f)

def parse_args():
    parser = argparse.ArgumentParser(description="Efficiently count lines in a file.")
    parser.add_argument('--file', type=str, required=True, help="Path to the JSONL file.")
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    total_lines = count_lines(args.file)
    print(f"\n\nTotal lines: {total_lines}", 30*"=", "\n\n")