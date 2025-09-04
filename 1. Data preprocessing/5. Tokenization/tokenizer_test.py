# -*- coding: utf-8 -*-
"""
Created on Wed Jun 18 16:30:09 2025

@author: Julius de Clercq
"""

from unsloth import FastLanguageModel

import os
import json
from torch import bfloat16
from transformers import AutoTokenizer
import huggingface_hub
import pathlib as pl


#%%
# Authenticate with Hugging Face
with open("API_keys.json", "r") as f:
    hf_token = json.load(f)["HF_token"]

huggingface_hub.login(hf_token)

# Load Llama 3 tokenizer
model_name = "meta-llama/Meta-Llama-3.1-8B"
# model_name = "meta-llama/Llama-3.3-70B-Instruct"
tokenizer = AutoTokenizer.from_pretrained(model_name)

if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token


# tokenizer = AutoTokenizer.from_pretrained(
#     model_name,
#     max_seq_length=8192,
#     device_map=None, # "auto"
#     dtype = bfloat16,  # None for auto detection. Float16 for Tesla T4, V100, Bfloat16 for Ampere+
#     cache_dir=os.environ["CACHE_DIR"],
# )

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name,
    max_seq_length=8192,
    device_map=None, # "auto"
    dtype = bfloat16,  # None for auto detection. Float16 for Tesla T4, V100, Bfloat16 for Ampere+
    cache_dir=os.environ["CACHE_DIR"],
    load_in_4bit=True,
    )

if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token


#%%


# filing_path = pl.Path("test_sequence.txt") 
# with open(filing_path, "r", encoding="utf-8", errors="ignore") as f:
#     filing = f.read() 

filing = "".join(f"{i} " for i in range(5))


#%% 

stride = 5 
max_length = 100


encoding = tokenizer(
    filing,
    return_attention_mask=True,
    truncation=False,
    return_overflowing_tokens=True,
    stride=stride,
    max_length=max_length,
    padding="max_length",  # Add this to ensure consistent length
    return_tensors=None    # Ensure we get lists, not tensors
    )

chunks = []
if isinstance(encoding["input_ids"][0], list):
    for input_ids, attn_mask in zip(encoding["input_ids"], encoding["attention_mask"]):
        if len(input_ids) > max_length:
            input_ids = input_ids[:max_length]
            attn_mask = attn_mask[:max_length]
        elif len(input_ids) < max_length:
            pad_len = max_length - len(input_ids)
            input_ids += [tokenizer.pad_token_id] * pad_len
            attn_mask += [0] * pad_len
        chunks.append({
            "input_ids": input_ids,
            "attention_mask": attn_mask,
            "labels": input_ids
        })
        
        
        
asdf = encoding["input_ids"] [0] 
token_list = [tokenizer.decode([tid]) for tid in asdf[:500]]
# print(token_list)    





asdf = encoding.data 



#%%



full_encoding = tokenizer(filing, truncation=False, return_attention_mask=True, return_tensors=None)
total_tokens = len(full_encoding['input_ids'])
print(f"Total tokens: {total_tokens}")

# Manual chunking
chunks = []
start = 0
while start < total_tokens:
    end = min(start + max_length, total_tokens)
    chunk_input_ids = full_encoding['input_ids'][start:end]
    chunk_attn_mask = full_encoding['attention_mask'][start:end]
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







#%%

print("Tokenizer class:", type(tokenizer))
print("Tokenizer name_or_path:", tokenizer.name_or_path)
print("BOS token ID:", tokenizer.bos_token_id)
print("EOS token ID:", tokenizer.eos_token_id)
print("PAD token ID:", tokenizer.pad_token_id)
print("Special tokens:", tokenizer.special_tokens_map)



#%%


sum([98, 20547, 970, 235])








































