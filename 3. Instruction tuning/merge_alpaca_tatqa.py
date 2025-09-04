# -*- coding: utf-8 -*-
"""
Created on Mon Jul  7 14:05:27 2025

@author: Julius de Clercq
"""

import json
import random
import pathlib as pl
from datasets import load_dataset


#%%
def load_jsonl(path):
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f]

def save_jsonl(examples, path):
    with open(path, "w", encoding="utf-8") as f:
        for ex in examples:
            json.dump(ex, f, ensure_ascii=False)
            f.write("\n")


#%%

def main():
    base_dir = pl.Path(__file__).parent.resolve()

    tatqa_files = {
        "train": base_dir / "tatqa_alpaca_train.jsonl",
        "eval":  base_dir / "tatqa_alpaca_eval.jsonl"
    }

    # Load TAT-QA examples
    tatqa_data = {
        split: load_jsonl(path) for split, path in tatqa_files.items()
    }

    # Load Alpaca dataset from HF
    alpaca_dataset = load_dataset("tatsu-lab/alpaca")

    # For reproducibility
    random.seed(543212345)

    for split in ["train", "eval"]:
        tatqa_examples = tatqa_data[split]
        n_needed = len(tatqa_examples)

        # Always sample from Alpaca "train" as there is no "eval" set. 
        alpaca_examples = random.sample(list(alpaca_dataset["train"]), n_needed)


        # Double check the keys.
        required_keys = {"instruction", "input", "output"}
        for ex in alpaca_examples:
            if not required_keys <= ex.keys():
                raise ValueError(f"Alpaca example missing required keys: {ex}")

        # Combine TAT-QA & Alpaca and shuffle 
        combined = tatqa_examples + alpaca_examples
        random.shuffle(combined)

        # Write out combined JSONL
        out_path = base_dir / f"fin_instruct_{split}.jsonl"
        save_jsonl(combined, out_path)

        print(f"Wrote {len(combined)} examples to {out_path}")
        
        
if __name__ == "__main__":
    main()
        
