# -*- coding: utf-8 -*-
"""
Created on Mon Jun  9 15:21:05 2025

@author: Julius de Clercq
"""

import os
import shutil
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from collections import deque

def remove_item(path):
    """Remove a single file or directory."""
    try:
        if os.path.isfile(path) or os.path.islink(path):
            os.unlink(path)
        elif os.path.isdir(path):
            shutil.rmtree(path)
        return True, path
    except Exception as e:
        return False, f"{path}: {e}"

def clear_directory_parallel(directory_path, num_workers):
    """Clear directory using ProcessPoolExecutor with progress tracking."""
    
    if not os.path.exists(directory_path):
        print(f"Directory {directory_path} does not exist.")
        return
    
    print(f"Scanning directory: {directory_path}")
    start_time = time.time()
    
    # Get list of all items in directory
    items = []
    for item in os.listdir(directory_path):
        items.append(os.path.join(directory_path, item))
    
    total_items = len(items)
    
    if total_items == 0:
        print("Directory is already empty.")
        return
    
    print(f"Found {total_items:,} items to remove")
    print(f"Using {num_workers} workers\n")
    
    # Progress tracking
    completed = 0
    failed = 0
    last_percentage = 0
    failures = deque(maxlen=100)  # Keep last 100 failures
    
    # Remove items in parallel with progress tracking
    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        # Submit all tasks
        futures = {executor.submit(remove_item, item): item for item in items}
        
        # Process completed tasks
        for future in as_completed(futures):
            success, result = future.result()
            completed += 1
            
            if not success:
                failed += 1
                failures.append(result)
            
            # Update progress every 5%
            percentage = int((completed / total_items) * 100)
            if percentage >= last_percentage + 5:
                elapsed = time.time() - start_time
                rate = completed / elapsed
                eta = (total_items - completed) / rate if rate > 0 else 0
                
                print(f"Progress: {percentage}% ({completed:,}/{total_items:,}) - "
                      f"Rate: {rate:.0f} items/sec - "
                      f"Elapsed: {elapsed:.0f}s - "
                      f"ETA: {eta:.0f}s")
                last_percentage = percentage
    
    # Final summary
    elapsed_total = time.time() - start_time
    successes = completed - failed
    
    print(f"\n{'='*60}")
    print(f"Completed in {elapsed_total:.1f} seconds")
    print(f"Total items processed: {completed:,}")
    print(f"Successes: {successes:,}")
    print(f"Failures: {failed:,}")
    print(f"Average rate: {completed/elapsed_total:.0f} items/sec")
    
    if failures:
        print(f"\nLast {len(failures)} failures:")
        for failure in failures:
            print(f"  {failure}")

if __name__ == "__main__":
    directory_to_clear = os.environ["DIRECTORY_TO_CLEAR"]
    clear_directory_parallel(directory_to_clear, num_workers=int(os.environ["SLURM_CPUS_PER_TASK"]))