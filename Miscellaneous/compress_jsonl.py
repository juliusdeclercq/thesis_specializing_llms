# -*- coding: utf-8 -*-
"""
Created on Fri Jul 18 13:37:31 2025

@author: Julius de Clercq
"""

import subprocess
import time
import psutil
import os
from pathlib import Path

input_file = os.environ["INPUT_FILE"]
data_dir = Path(input_file).parent
file_stem = Path(input_file).stem 
output_file = Path(data_dir) / f"{file_stem}.zst"


proc = subprocess.Popen([
    "zstd", 
    "--threads=0", 
    os.environ["COMPRESSION_LEVEL"], 
    "--rm",
    input_file, 
    "-o", str(output_file)
])

time.sleep(10)  # wait a bit for zstd to start threads

p = psutil.Process(proc.pid)
print("zstd thread count:", p.num_threads())

proc.wait()