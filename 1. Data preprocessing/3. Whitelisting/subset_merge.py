# -*- coding: utf-8 -*-
"""
Created on Tue Jul 15 12:32:57 2025

@author: Julius de Clercq


Script aiming to merge the whitelisted data per year, which is needed because 
the years were whitelisted on seaparate nodes for the sake of parallelization 
and mitigating damage in case of failure.
"""
import os 
from pathlib import Path


data_dir = Path(os.environ["SCRATCH_OUTPUT"])  

# Output file
output_file = Path(os.environ["SCRATCH_DIR"]) / "data" / os.environ["OUTPUT_FILE"]

# Years to include
years = range(2012, 2025)

with output_file.open("w", encoding="utf-8") as outfile:
    for year in years:
        input_file = data_dir / f"subsetted_data_{year}.jsonl"
        if not input_file.exists():
            print(f"Warning: {input_file} does not exist. Skipping.")
            continue
        print(f"Merging {input_file}")
        with input_file.open("r", encoding="utf-8") as infile:
            for line in infile:
                outfile.write(line)

print(f"\nMerged files into {output_file}")