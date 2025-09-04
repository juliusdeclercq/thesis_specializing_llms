# -*- coding: utf-8 -*-
"""
Created on Wed May 14 15:30:25 2025

@author: Julius de Clercq
"""
#%%                 Imports
import os
import json
import argparse
import concurrent.futures
from transformers import AutoTokenizer
import torch
import pickle
import psutil
from pathlib import Path
from tqdm import tqdm

# import builtins
# Override print to always flush output instead of using a buffer (useful for multiprocessing).
# print = lambda *args, **kwargs: builtins.print(*args, **{**kwargs, 'flush': True})

# Set verbosity (True means a lot of print statements)
VERBOSE = False

# Force offline mode
# os.environ["HF_HUB_OFFLINE"] = "1"
# os.environ["TRANSFORMERS_OFFLINE"] = "1"


#%%                 Globals

worker_globals = {}

#%%                 Functions


def build_line_index(input_path, output_dir):
    """
    Build an index of byte offsets for each line in the file.
    """
    index_path = Path(output_dir) / "line_offsets.pkl"
    
    # Check if index already exists
    if index_path.exists():
        print("Loading existing line index...", flush=True)
        with open(index_path, "rb") as f:
            return pickle.load(f)
    
    print("Building line index (this will take a few minutes but only needs to be done once)...", flush=True)
    
    file_size = os.path.getsize(input_path)
    update_interval = int(file_size * 0.02)  # 2% of file size in bytes
    
    offsets = []
    with open(input_path, "rb") as f:
        offset = 0
        last_reported = 0
        
        # Create progress bar based on file size
        pbar = tqdm(total=file_size, desc="Indexing lines", unit="B", unit_scale=True)
        
        for line_num, line in enumerate(f):
            offsets.append(offset)
            offset += len(line)
            
            # Update progress bar every 2%
            if offset - last_reported >= update_interval:
                pbar.update(offset - last_reported)
                last_reported = offset
        
        # Final update for any remaining bytes
        if offset > last_reported:
            pbar.update(offset - last_reported)
        pbar.close()
    
    print(f"Indexed {len(offsets)} lines", flush=True)
    
    # Save index for future runs
    with open(index_path, "wb") as f:
        pickle.dump(offsets, f)
    
    print(f"Line index saved to {index_path}", flush=True)
    return offsets


def worker_init(args):
    """
    Worker initializer: sets up all per-process state (globals).
    """
    global worker_globals
    worker_globals["pid"] = os.getpid()
    worker_globals["input_path"] = args.input_path
    worker_globals["output_dir"] = args.output_dir
    worker_globals["tqdm_step"] = args.tqdm_step
    cache_kwargs = dict(cache_dir=args.cache_dir, local_files_only=True)
    worker_globals["tokenizer"] = AutoTokenizer.from_pretrained(
        args.model_name_or_path,
        use_fast=True,
        **cache_kwargs
    )
    worker_globals["max_seq_length"] = int(os.environ["MAX_SEQ_LENGTH"])
    worker_globals["stride"] = int(os.environ["STRIDE"])
    
    # Use eos_token_id as pad_token_id for Llama 3.1 model. Llama 3.3 does have a separate pad token. 
    if worker_globals["tokenizer"].pad_token is None:
        worker_globals["tokenizer"].pad_token = worker_globals["tokenizer"].eos_token
    

def tokenize_and_chunk(tokenizer, line, max_length=8192, stride=512):
    filing = json.loads(line)["filing"]
    if not filing or len(filing.strip()) == 0:
        print(f"Warning: Empty filing content on line {line}")
        return []
    
    tokens = tokenizer(
        filing,
        return_attention_mask=True,
        truncation=False,
        return_overflowing_tokens=True,     # This does nothing!
        stride=stride,                      # Neither does this! This is why I need to chunk manually....
        max_length=max_length,
        padding="max_length",  # Ensures consistent length. 
        return_tensors=None    # Ensure we get lists, not tensors
        )

    total_tokens = len(tokens['input_ids'][0])

    # Manual chunking
    chunks = []
    start = 0
    while start < total_tokens:
        end = min(start + max_length, total_tokens)
        chunk_input_ids = tokens['input_ids'][0][start:end]
        chunk_attn_mask = tokens['attention_mask'][0][start:end]
        if len(chunk_input_ids) < max_length:
            pad_len = max_length - len(chunk_input_ids)
            chunk_input_ids += [tokenizer.pad_token_id] * pad_len
            chunk_attn_mask += [0] * pad_len
        chunks.append({
            "input_ids": chunk_input_ids,
            "attention_mask": chunk_attn_mask,
            "labels": chunk_input_ids
        })
        start += max_length - stride
    return chunks


def worker_function(worker_idx, start_idx, end_idx, start_offset):
    
    print(f"Worker {worker_idx} started processing lines {start_idx} to {end_idx}", flush=False)
    print(f"Worker {worker_idx} memory usage: {psutil.Process().memory_info().rss / 1024**2:.2f} MB", flush=True)
    global worker_globals
    pid = worker_globals["pid"]
    tokenizer = worker_globals["tokenizer"]
    input_path = worker_globals["input_path"]
    output_dir = Path(worker_globals["output_dir"])
    tqdm_step = worker_globals["tqdm_step"]
    output_dir.mkdir(parents=True, exist_ok=True)

    if VERBOSE:
        print(f"\n\n-------- Worker Process {pid} ---------")
        print(f"Worker {worker_idx}: Processing lines {start_idx} to {end_idx}")
        print(f"Worker {worker_idx}: Starting at byte offset {start_offset}")
        print("----------------- MAIN ------------------\n\n", flush=True)

    # Use JSONL format for streaming append
    output_file = output_dir / f"worker_{worker_idx:04d}_chunks.jsonl"
    chunks_written = 0
    
    with open(input_path, "rb") as f_in, open(output_file, "w") as f_out:
        # Seek directly to the starting byte offset
        f_in.seek(start_offset)
        
        # Optionally, tqdm for worker 0
        pbar = tqdm(total=(end_idx - start_idx), desc=f"Processing filings W{worker_idx}", unit="filings") if worker_idx == 0 else None

        for idx in range(start_idx, end_idx):
            line = f_in.readline().decode('utf-8')

            chunks = tokenize_and_chunk(tokenizer, line, 
                                        max_length = worker_globals["max_seq_length"], 
                                        stride = worker_globals["stride"]
                                        )
            
            # Stream each chunk to disk immediately in one big JSONL file
            for chunk in chunks:
                json.dump(chunk, f_out)
                f_out.write('\n')
                chunks_written += 1
            
            # Explicitly delete chunks after writing
            del chunks
            
            # Optional tqdm update
            if pbar and ((idx - start_idx + 1) % tqdm_step == 0):
                pbar.update(tqdm_step)

        # Progress bar finalize
        if pbar:
            remain = (end_idx - start_idx) % tqdm_step
            if remain:
                pbar.update(remain)
            pbar.close()

    if VERBOSE:
        print(f"Worker {worker_idx}: Wrote {chunks_written} chunks to {output_file}")
        print("---------------- END MAIN ---------------\n\n\n", flush=True)

    return worker_idx, output_file, chunks_written

#%%                     Main
def main():
    # ------- Parse arguments -------
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_workers", type=int, required=True)
    parser.add_argument("--input_path", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--model_name_or_path", type=str, required=True)
    parser.add_argument("--cache_dir", type=str, required=True)
    parser.add_argument("--tqdm_step", type=int, required=True)  # New: pass progress step through CLI

    args = parser.parse_args()

    # Loading tokenizer once to check behaviour.
    print("\n\nInitializing tokenizer once to populate cache directory.\n\n", flush=True)
    print(f"Cache dir: {args.cache_dir}", flush=True)
    print(f"Model: {args.model_name_or_path}", flush=True)
    print(f"HF_HUB_OFFLINE: {os.environ.get('HF_HUB_OFFLINE', 'not set')}", flush=True)
    print("Loading tokenizer...", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name_or_path,
        cache_dir=args.cache_dir,
        local_files_only=False,
    )
    # Set pad_token if needed
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    print("Tokenizer loaded successfully!", flush=True)

    print(f"\n\nMain process: Starting tokenization with {args.num_workers} worker processes.\n\n")

    line_offsets = build_line_index(args.input_path, args.output_dir)
    total_lines = len(line_offsets)
    lines_per_worker = total_lines // args.num_workers

    work_ranges = []
    for i in range(args.num_workers):
        start_idx = i * lines_per_worker
        end_idx = (i + 1) * lines_per_worker if i < args.num_workers - 1 else total_lines
        start_offset = line_offsets[start_idx]
        work_ranges.append((start_idx, end_idx, start_offset))

    print(f"Total filings to process: {total_lines}")
    print(f"Lines per worker: {lines_per_worker}", flush = True)

    # ---------- Execution ----------
    worker_results = []
    total_chunks = 0
    with concurrent.futures.ProcessPoolExecutor(
            max_workers=args.num_workers,
            initializer=worker_init,
            initargs=(args,)
            ) as executor:
            futures = {executor.submit(worker_function, i, start, end, offset): i
                      for i, (start, end, offset) in enumerate(work_ranges)}
            for future in tqdm(concurrent.futures.as_completed(futures), total=args.num_workers, desc="Workers completed"):
                worker_idx, output_file, chunks_written = future.result()
                worker_results.append((worker_idx, output_file))
                total_chunks += chunks_written
    # -------------------------------

    print(f"\n\nAll workers completed. Total chunks created: {total_chunks}")
    print("Merging chunk files...")

    # Sort by worker_idx to maintain order
    worker_results.sort(key=lambda x: x[0])

    # Merge all JSONL files into one
    output_path = Path(args.output_dir) / os.environ["OUTPUT_FILE"]
    with open(output_path, "wb") as outfile:
        for worker_idx, file_path in tqdm(worker_results, desc="Merging files"):
            with open(file_path, "rb") as infile:
                outfile.write(infile.read())
            # Delete intermediate file
            file_path.unlink()

    # Count total lines in final file for verification
    with open(output_path, "r") as f:
        final_count = sum(1 for _ in f)

    print(f"\nFinal tokenized data saved to: {output_path}")
    print(f"Total chunks in final file: {final_count}")
    print(f"File size: {output_path.stat().st_size / (1024**3):.2f} GB")
    
    if final_count != total_chunks:
        print(f"WARNING: Chunk count mismatch! Expected {total_chunks}, got {final_count}")

if __name__ == "__main__":
    main()