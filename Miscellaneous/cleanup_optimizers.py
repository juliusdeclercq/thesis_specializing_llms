# -*- coding: utf-8 -*-
"""
Created on Mon Jun 30 13:14:28 2025

@author: Julius de Clercq
"""

import os
from concurrent.futures import ProcessPoolExecutor, as_completed


#%%
def delete_optimizer(path):
    """Deletes the given optimizer.pt file."""
    try:
        os.remove(path)
        return f"Deleted: {path}"
    except Exception as e:
        return f"Could not delete {path}: {e}"


#%%
if __name__ == "__main__":
    # Collect all optimizer.pt file paths
    optimizer_files = []
    BASE_DIR = os.environ["BASE_DIR"]
    for root, dirs, files in os.walk(BASE_DIR):
        if "optimizer.pt" in files:
            optimizer_files.append(os.path.join(root, "optimizer.pt"))

    print(f"Found {len(optimizer_files)} optimizer.pt files to delete.")

    
    with ProcessPoolExecutor(max_workers = int(os.environ["SLURM_CPUS_PER_TASK"])) as executor:
        # Submit all delete jobs
        futures = [executor.submit(delete_optimizer, path) for path in optimizer_files]

        # As each completes, print the result
        for future in as_completed(futures):
            print(future.result())
            
            
            
            