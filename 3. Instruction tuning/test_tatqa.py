# -*- coding: utf-8 -*-
"""
Created on Mon Jul  7 14:19:42 2025

@author: Julius de Clercq
"""

import json
from tqdm import tqdm
from transformers import AutoTokenizer
from huggingface_hub import login
from pprint import pprint

def get_tokenizer():
    with open("API_keys.json", "r") as f:
        hf_token = json.load(f)["HF_token"]
    login(hf_token)
    model_name = "meta-llama/Meta-Llama-3.1-8B"
    # model_name = "meta-llama/Llama-3.3-70B-Instruct"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer

def load_data(filepath):
    if filepath.endswith(".jsonl"):
        with open(filepath, "r", encoding="utf-8") as f:
            return [json.loads(line) for line in f]
    elif filepath.endswith(".json"):
        with open(filepath, "r", encoding="utf-8") as f:
            return json.load(f)
    else:
        raise ValueError("Unsupported file format. Use .json or .jsonl")

def make_prompt(instruction, input_):
    # Adjust this function if your actual prompt format is different
    if input_:
        prompt = f"### Instruction:\n{instruction}\n\n### Input:\n{input_}\n\n### Response:"
    else:
        prompt = f"### Instruction:\n{instruction}\n\n### Response:"
    return prompt

def main():
    # filepath = "tatqa_original_train.json"
    # filepath = "tatqa_alpaca_train.jsonl"
    # filepath = "tatqa_alpaca_eval.jsonl"
    filepath = "fin_instruct_train.jsonl"
    # filepath = "fin_instruct_eval.jsonl"

    tokenizer = get_tokenizer()
    data = load_data(filepath)

    max_prompt_len = 0
    max_output_len = 0
    max_total_len = 0

    idx = 13
    for i, ex in enumerate(data):
        if i!=idx:
            continue
        instruction = ex.get("instruction", "")
        input_ = ex.get("input", "")
        output = ex.get("output", "")
        def make_prompt(instruction, input_):
            # Adjust this function if your actual prompt format is different
            if input_:
                prompt = f"### Instruction:\n{instruction}\n\n### Input:\n{input_}\n\n### Response:"
            else:
                prompt = f"### Instruction:\n{instruction}\n\n### Response:"
            return prompt
        
        prompt = make_prompt(instruction, input_)
        
        print(f"i = {i}\n\n\n")
        print(prompt)
        break
        
    #     prompt_len = len(tokenizer.encode(prompt, add_special_tokens=False))
    #     output_len = len(tokenizer.encode(output, add_special_tokens=False))
    #     total_len = len(tokenizer.encode(prompt + output, add_special_tokens=False))

    #     max_prompt_len = max(max_prompt_len, prompt_len)
    #     max_output_len = max(max_output_len, output_len)
    #     max_total_len = max(max_total_len, total_len)

    # print(f"MAX prompt (instruction+input) length: {max_prompt_len} tokens")
    # print(f"MAX output length: {max_output_len} tokens")
    # print(f"MAX total (prompt+output) length: {max_total_len} tokens\n")
    # print("Set your training max_seq_length to at least the max total length above for full coverage (or truncate as preferred).")

if __name__ == "__main__":
    main()