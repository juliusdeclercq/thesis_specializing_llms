# -*- coding: utf-8 -*-
"""
Created on Mon Jul 28 21:05:07 2025

@author: Julius de Clercq
"""
from transformers import AutoTokenizer

# Load both tokenizers
base_tokenizer = AutoTokenizer.from_pretrained("meta-llama/Meta-Llama-3.1-8B")
instruct_tokenizer = AutoTokenizer.from_pretrained("/scratch-shared/jdclercq/models/instruct_13216005/checkpoint-830")

# Find the difference
print(f"Base tokenizer size: {len(base_tokenizer)}")
print(f"Instruction tokenizer size: {len(instruct_tokenizer)}")

# Get the new tokens
all_tokens_base = set(base_tokenizer.get_vocab().keys())
all_tokens_instruct = set(instruct_tokenizer.get_vocab().keys())
new_tokens = all_tokens_instruct - all_tokens_base
print(f"New tokens added: {new_tokens}")

# Check if they're special tokens
print(f"Special tokens in instruct tokenizer: {instruct_tokenizer.special_tokens_map}")