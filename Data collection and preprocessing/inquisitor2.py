# -*- coding: utf-8 -*-
"""
Created on Wed Feb 12 13:54:04 2025

@author: Julius de Clercq
"""

import os
import io
import gzip
import tarfile
import hashlib
import zlib
import concurrent.futures

def check_gzip_integrity(file_path):
    """Check if the gzip file can be fully decompressed without errors."""
    try:
        with gzip.open(file_path, "rb") as f:
            for _ in f:  # Read line-by-line to force full decompression
                pass
    except zlib.error as e:
        print(f"Zlib Error in GZIP: {file_path} - {e}")
        return file_path
    except Exception as e:
        print(f"Corrupt GZIP: {file_path} - {e}")
        return file_path
    return None

def check_tar_integrity(file_path):
    """Check if the .tar file inside the .tar.gz is valid by attempting extraction."""
    try:
        with gzip.open(file_path, "rb") as f:
            with tarfile.open(fileobj=f, mode="r") as tar:
                tar.extractall(path="/dev/null")  # Extract to a dummy location
    except (tarfile.TarError, OSError, zlib.error) as e:
        print(f"Corrupt TAR: {file_path} - {e}")
        return file_path
    return None

def check_data_integrity(file_path):
    """Check if the file data has been corrupted (e.g., truncated)."""
    try:
        with gzip.open(file_path, "rb") as f:
            content = f.read()  # Fully read the file to detect corruption
            file_hash = hashlib.sha256(content).hexdigest()
            # If you have expected hashes, compare here
    except Exception as e:
        print(f"Data corruption detected in {file_path}: {e}")
        return file_path
    return None

def check_tar_gz(file_path):
    """Run all checks on a .tar.gz file."""
    for check in [check_gzip_integrity, check_tar_integrity, check_data_integrity]:
        result = check(file_path)
        if result:
            return result
    return None

def process_year(year, base_dir):
    """Check all .tar.gz files in a given year's directory in parallel."""
    dir_path = os.path.join(base_dir, str(year))
    
    if not os.path.isdir(dir_path):
        print(f"Skipping missing directory: {dir_path}")
        return []

    print(f"Processing year: {year}")
    corrupt_files = []
    files = [os.path.join(dir_path, f) for f in os.listdir(dir_path) if f.endswith(".tar.gz")]

    with concurrent.futures.ProcessPoolExecutor(max_workers=os.cpu_count() - 1) as executor:
        future_to_file = {executor.submit(check_tar_gz, file): file for file in files}
        
        for future in concurrent.futures.as_completed(future_to_file):
            try:
                result = future.result()
                if result:
                    print(f"Corrupt file: {result}")
                    corrupt_files.append(result)
            except Exception as e:
                print(f"Error in worker process: {e}")
    
    return corrupt_files

def main():
    """Main execution function."""
    base_dir = "/projects/prjs1109/data/raw"
    YEARS = [2016]      # range(2012, 2025)
    LOG_FILE = "corrupt_files.log"
    
    all_corrupt_files = []
    
    for year in YEARS:
        corrupt_list = process_year(year, base_dir)
        all_corrupt_files.extend(corrupt_list)
    
    with open(LOG_FILE, "w") as log:
        for file in all_corrupt_files:
            log.write(file + "\n")
    print(f"Check complete. Corrupt files (if any) are listed in {LOG_FILE}.")
    
if __name__ == "__main__":
    main()
