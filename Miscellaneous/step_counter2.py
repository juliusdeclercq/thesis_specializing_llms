# -*- coding: utf-8 -*-
"""
Created on Tue Apr 22 23:19:39 2025

@author: Julius de Clercq
"""

import argparse
import math
import json
import multiprocessing
import os
from tqdm import tqdm
from transformers import AutoTokenizer, logging as hf_logging

# Suppress excessive warnings from tokenizers when used in multiprocessing
# (often related to fork safety, which is handled here by re-initializing)
hf_logging.set_verbosity_error()
# Also recommended by HF tokenizers for multiprocessing
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# --- Global Tokenizer Storage (per process) ---
# We initialize this once per worker process to avoid repeated loading
process_tokenizer = None

def init_tokenizer(model_name_or_path, trust_remote_code_flag):
    """Initializer function for each worker process."""
    global process_tokenizer
    # print(f"Initializing tokenizer in process {os.getpid()}...")
    try:
        process_tokenizer = AutoTokenizer.from_pretrained(
            model_name_or_path,
            trust_remote_code=trust_remote_code_flag
        )
        # print(f"Tokenizer initialized successfully in process {os.getpid()}.")
    except Exception as e:
        print(f"ERROR initializing tokenizer in process {os.getpid()}: {e}")
        # Worker will likely fail if tokenizer isn't loaded, handled in process_line
    # Set parallelism again just in case
    os.environ["TOKENIZERS_PARALLELISM"] = "false"


def process_line(line):
    """Processes a single line to count tokens."""
    global process_tokenizer
    # Ensure tokenizer was initialized
    if process_tokenizer is None:
        return 0, "tokenizer_error"

    try:
        data = json.loads(line)
        text = data.get("filing", "") # Use the correct key "filing"
        if text:
            # Tokenize using the process-local tokenizer
            # Disable warnings for sequence length here, we just need the count
            tokens = process_tokenizer(text, add_special_tokens=True)["input_ids"]
            return len(tokens), "success"
        else:
            return 0, "empty_filing" # Count lines with empty/missing "filing"
    except json.JSONDecodeError:
        return 0, "invalid_json"
    except Exception:
        # Capture generic errors during processing for this line
        return 0, "other_error"

def calculate_steps_parallel(args):
    """Calculates max_steps using multiprocessing."""

    # Get number of processes
    num_processes = args.num_workers if args.num_workers else os.cpu_count()
    print(f"Using {num_processes} worker processes.")

    # Check data file existence early
    if not os.path.exists(args.data_path):
         print(f"ERROR: Data file not found at {args.data_path}")
         return

    # --- Estimate Total Lines for Progress Bar (Optional but helpful) ---
    total_lines = 0
    if args.show_progress:
        try:
            print("Counting total lines for progress bar (this might take a moment)...")
            with open(args.data_path, 'r', encoding='utf-8') as f:
                for _ in f:
                    total_lines += 1
            print(f"Found {total_lines} lines.")
        except Exception as e:
            print(f"Warning: Could not count lines for progress bar - {e}")
            total_lines = None # Progress bar won't show total
    # --- End Line Count ---

    total_tokens = 0
    num_examples_processed = 0 # Lines successfully read and parsed (even if empty filing)
    num_valid_examples = 0 # Lines with non-empty filing and successful tokenization
    error_counts = {
        "tokenizer_error": 0,
        "empty_filing": 0,
        "invalid_json": 0,
        "other_error": 0,
    }

    # Create the pool AFTER checking file and potentially counting lines
    # Pass tokenizer info to the initializer for each worker
    pool = multiprocessing.Pool(processes=num_processes,
                                initializer=init_tokenizer,
                                initargs=(args.model_name_or_path, args.trust_remote_code))

    print(f"Starting parallel processing of {args.data_path}...")
    try:
        with open(args.data_path, 'r', encoding='utf-8') as f:
            # Use imap_unordered for potentially better performance
            # chunksize helps reduce overhead of passing small items
            map_results = pool.imap_unordered(process_line, f, chunksize=args.chunksize)

            # Wrap with tqdm if we have total_lines and want progress
            if args.show_progress and total_lines is not None:
                 pbar = tqdm(map_results, total=total_lines, desc="Processing lines")
            else:
                 pbar = map_results # No progress bar

            for token_count, status in pbar:
                num_examples_processed += 1
                if status == "success":
                    total_tokens += token_count
                    num_valid_examples += 1
                else:
                    error_counts[status] += 1

    except Exception as e:
        print(f"\nERROR during parallel processing: {e}")
    finally:
        # Important to close and join the pool
        print("\nClosing worker pool...")
        pool.close()
        pool.join()
        print("Worker pool closed.")


    # --- Reporting ---
    print(f"\n--- Processing Summary ---")
    print(f"Total lines read/processed: {num_examples_processed}")
    print(f"Lines with invalid JSON: {error_counts['invalid_json']}")
    print(f"Lines with missing/empty 'filing': {error_counts['empty_filing']}")
    print(f"Lines with tokenizer errors: {error_counts['tokenizer_error']}")
    print(f"Lines with other processing errors: {error_counts['other_error']}")

    if num_valid_examples == 0:
        print("\nERROR: No valid examples with non-empty 'filing' field were successfully tokenized.")
        return

    print(f"\n--- Calculation Inputs ---")
    print(f"Total valid examples tokenized: {num_valid_examples}")
    print(f"Total tokens counted: {total_tokens}")
    print(f"Max Sequence Length: {args.max_seq_length}")
    print(f"Num GPUs: {args.num_gpus}")
    print(f"Per Device Batch Size: {args.per_device_batch_size}")
    print(f"Gradient Accumulation Steps: {args.gradient_accumulation_steps}")

    total_chunks = math.ceil(total_tokens / args.max_seq_length)
    chunks_per_step = (
        args.num_gpus
        * args.per_device_batch_size
        * args.gradient_accumulation_steps
    )

    if chunks_per_step == 0:
        print("ERROR: Chunks per step is zero. Check GPU count, batch size, and grad accum.")
        return

    max_steps = math.ceil(total_chunks / chunks_per_step)

    print(f"\n--- Results ---")
    print(f"Approximate total chunks (size {args.max_seq_length}): {total_chunks}")
    print(f"Effective chunks processed per optimizer step: {chunks_per_step}")
    print(f"Calculated max_steps for one approximate epoch: {max_steps}")
    print("\nNOTE: This is an approximation based on token count and chunking.")

    return max_steps

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Calculate max_steps for one epoch using parallel processing.")

    parser.add_argument("--data_path", type=str, required=True, help="Path to the .jsonl data file.")
    parser.add_argument("--model_name_or_path", type=str, required=True, help="Model identifier for tokenizer loading.")
    parser.add_argument("--max_seq_length", type=int, required=True, help="Maximum sequence length used during training.")
    parser.add_argument("--num_gpus", type=int, default=1, help="Number of GPUs used for training.")
    parser.add_argument("--per_device_batch_size", type=int, required=True, help="Per device train batch size used during training.")
    parser.add_argument("--gradient_accumulation_steps", type=int, required=True, help="Gradient accumulation steps used during training.")
    parser.add_argument("--num_workers", type=int, default=None, help="Number of worker processes. Defaults to os.cpu_count().")
    parser.add_argument("--chunksize", type=int, default=100, help="Number of lines processed by a worker at a time.")
    parser.add_argument("--trust_remote_code", action='store_true', help="Pass true if tokenizer requires remote code.")
    parser.add_argument("--show_progress", action='store_true', help="Show a tqdm progress bar (requires counting lines first).")


    args = parser.parse_args()
    calculate_steps_parallel(args)