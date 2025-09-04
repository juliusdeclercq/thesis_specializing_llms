# -*- coding: utf-8 -*-
"""
Created on Wed Jul  9 14:25:57 2025

@author: Julius de Clercq
"""


import os
import shutil
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from collections import deque, defaultdict
import heapq

def scan_directory_tree(directory_path, max_depth=None):
    """Scan directory tree and return all directories with their file info."""
    all_dirs = []
    dir_info = {}
    
    for root, dirs, files in os.walk(directory_path):
        # Calculate depth
        depth = root[len(directory_path):].count(os.sep)
        if max_depth is not None and depth > max_depth:
            dirs[:] = []  # Don't descend further
            continue
            
        # Calculate size and file count for this directory
        local_size = 0
        file_count = len(files)
        
        for file in files:
            try:
                file_path = os.path.join(root, file)
                local_size += os.path.getsize(file_path)
            except:
                pass
        
        # Store directory info
        dir_info[root] = {
            'size': local_size,
            'file_count': file_count,
            'has_subdirs': len(dirs) > 0,
            'depth': depth,
            'is_leaf': len(dirs) == 0
        }
        
        # Add ALL directories with files, not just leaf directories
        if file_count > 0:
            all_dirs.append((root, local_size, file_count))
    
    return all_dirs, dir_info

def remove_file(path):
    """Remove a single file."""
    try:
        if os.path.isfile(path) or os.path.islink(path):
            os.unlink(path)
            return True, path
        return False, f"{path}: Not a file"
    except Exception as e:
        return False, f"{path}: {e}"

def remove_empty_directory(path):
    """Remove a directory if it's empty."""
    try:
        if os.path.isdir(path) and not os.listdir(path):
            os.rmdir(path)
            return True, path
        return False, f"{path}: Directory not empty or doesn't exist"
    except Exception as e:
        return False, f"{path}: {e}"

def get_all_files_in_directory(directory):
    """Get all files in a directory (non-recursive)."""
    files = []
    try:
        for item in os.listdir(directory):
            item_path = os.path.join(directory, item)
            if os.path.isfile(item_path) or os.path.islink(item_path):
                files.append(item_path)
    except Exception as e:
        print(f"Error listing directory {directory}: {e}", flush=True)
    return files

def clear_directory_balanced(directory_path, num_workers):
    """Clear directory using balanced worker assignment."""
    
    if not os.path.exists(directory_path):
        print(f"Directory {directory_path} does not exist.", flush=True)
        return
    
    print(f"Scanning directory tree: {directory_path}", flush=True)
    scan_start = time.time()
    
    # Scan directory tree to find ALL directories with files
    dirs_with_files, dir_info = scan_directory_tree(directory_path)
    
    scan_time = time.time() - scan_start
    print(f"Scan completed in {scan_time:.1f} seconds", flush=True)
    print(f"Found {len(dirs_with_files)} directories containing files", flush=True)
    print(f"Total directories: {len(dir_info)}", flush=True)
    
    # Sort directories by size (largest first) for better work distribution
    dirs_with_files.sort(key=lambda x: x[1], reverse=True)
    
    # Calculate total work
    total_files = sum(info['file_count'] for info in dir_info.values())
    total_size = sum(info['size'] for info in dir_info.values())
    
    print(f"Total files to remove: {total_files:,}", flush=True)
    print(f"Total size: {total_size/1024/1024/1024:.2f} GB", flush=True)
    print(f"Using {num_workers} workers\n", flush=True)
    
    if total_files == 0:
        print("No files to remove.", flush=True)
        # Still try to remove empty directories
        if dir_info:
            print("Checking for empty directories to remove...", flush=True)
    else:
        start_time = time.time()
        completed_files = 0
        failed = 0
        last_percentage = 0
        failures = deque(maxlen=100)
        
        # Phase 1: Delete all files from ALL directories
        print("Phase 1: Deleting files from all directories...", flush=True)
        
        with ProcessPoolExecutor(max_workers=num_workers) as executor:
            # Create work queue - process all directories with files
            futures = {}
            
            # Process each directory that contains files
            for dir_path, size, file_count in dirs_with_files:
                if file_count > 0:
                    files = get_all_files_in_directory(dir_path)
                    for file_path in files:
                        future = executor.submit(remove_file, file_path)
                        futures[future] = file_path
            
            print(f"Total files queued: {len(futures)}", flush=True)
            
            # Process completed tasks
            for future in as_completed(futures):
                success, result = future.result()
                completed_files += 1
                
                if not success:
                    failed += 1
                    failures.append(result)
                
                # Update progress
                percentage = int((completed_files / total_files) * 100) if total_files > 0 else 100
                if percentage >= last_percentage + 5 or completed_files == total_files:
                    elapsed = time.time() - start_time
                    rate = completed_files / elapsed if elapsed > 0 else 0
                    eta = (total_files - completed_files) / rate if rate > 0 else 0
                    
                    print(f"Progress: {percentage}% ({completed_files:,}/{total_files:,}) - "
                          f"Rate: {rate:.0f} files/sec - "
                          f"Elapsed: {elapsed:.1f}s - "
                          f"ETA: {eta:.0f}s", flush=True)
                    last_percentage = percentage
    
    # Phase 2: Delete empty directories bottom-up
    print("\nPhase 2: Removing empty directories...", flush=True)
    
    # Get all directories sorted by depth (deepest first)
    all_dirs = [(path, info['depth']) for path, info in dir_info.items()]
    all_dirs.sort(key=lambda x: x[1], reverse=True)
    
    dirs_removed = 0
    dirs_failed = 0
    
    if all_dirs:
        with ProcessPoolExecutor(max_workers=num_workers) as executor:
            # Process directories in batches by depth
            current_depth = all_dirs[0][1]
            batch = []
            
            for dir_path, depth in all_dirs:
                if depth < current_depth:
                    # Process current batch
                    if batch:
                        print(f"Processing {len(batch)} directories at depth {current_depth}", flush=True)
                        futures = {executor.submit(remove_empty_directory, path): path 
                                  for path in batch}
                        for future in as_completed(futures):
                            success, result = future.result()
                            if success:
                                dirs_removed += 1
                            else:
                                dirs_failed += 1
                    
                    # Start new batch
                    current_depth = depth
                    batch = [dir_path]
                else:
                    batch.append(dir_path)
            
            # Process final batch
            if batch:
                print(f"Processing {len(batch)} directories at depth {current_depth}", flush=True)
                futures = {executor.submit(remove_empty_directory, path): path 
                          for path in batch}
                for future in as_completed(futures):
                    success, result = future.result()
                    if success:
                        dirs_removed += 1
                    else:
                        dirs_failed += 1
    
    # Try to remove the base directory itself if it's empty
    try:
        if os.path.exists(directory_path) and not os.listdir(directory_path):
            os.rmdir(directory_path)
            dirs_removed += 1
            print(f"Removed base directory: {directory_path}", flush=True)
    except:
        pass
    
    # Final summary
    if total_files > 0:
        elapsed_total = time.time() - start_time
        print(f"\n{'='*60}", flush=True)
        print(f"Completed in {elapsed_total:.1f} seconds (+ {scan_time:.1f}s scanning)", flush=True)
        print(f"Files processed: {completed_files:,}", flush=True)
        print(f"Files removed: {completed_files - failed:,}", flush=True)
        print(f"File failures: {failed:,}", flush=True)
        print(f"Directories removed: {dirs_removed:,}", flush=True)
        print(f"Directory failures: {dirs_failed:,}", flush=True)
        print(f"Average rate: {completed_files/elapsed_total:.0f} files/sec", flush=True)
    else:
        print(f"\n{'='*60}", flush=True)
        print(f"No files to remove", flush=True)
        print(f"Directories removed: {dirs_removed:,}", flush=True)
        print(f"Directory failures: {dirs_failed:,}", flush=True)
    
    if failures:
        print(f"\nLast {len(failures)} failures:", flush=True)
        for failure in failures:
            print(f"  {failure}", flush=True)
    
    # Final check: list any remaining items
    if os.path.exists(directory_path):
        try:
            remaining = os.listdir(directory_path)
            if remaining:
                print(f"\nWarning: {len(remaining)} items still remain in {directory_path}:", flush=True)
                for item in remaining[:10]:  # Show first 10
                    print(f"  - {item}", flush=True)
                if len(remaining) > 10:
                    print(f"  ... and {len(remaining) - 10} more", flush=True)
        except:
            pass

if __name__ == "__main__":
    directory_to_clear = os.environ["DIRECTORY_TO_CLEAR"]
    clear_directory_parallel = clear_directory_balanced  # For backward compatibility
    clear_directory_balanced(directory_to_clear, num_workers=int(os.environ["SLURM_CPUS_PER_TASK"]))