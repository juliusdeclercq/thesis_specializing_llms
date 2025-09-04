# -*- coding: utf-8 -*-
"""
Created on Mon Jul  7 14:07:31 2025

@author: Julius de Clercq
"""


import json
import requests
import pathlib as pl


def download_tatqa_json(url, local_filename):
    print(f"Downloading TAT-QA dataset from {url}")
    response = requests.get(url)
    response.raise_for_status()
    with open(local_filename, "wb") as f:
        f.write(response.content)
    print(f"Saved dataset to {local_filename}")

def table_to_text(table):
    """Convert a 2D table array to a markdown-like text table."""
    lines = []
    header = table[0]
    lines.append(" | ".join(header))
    lines.append(" | ".join(["---"] * len(header)))
    for row in table[1:]:
        lines.append(" | ".join(row))
    return "\n".join(lines)

def convert_tatqa_to_alpaca(input_json_path, output_jsonl_path):
    print(f"Loading dataset from {input_json_path}")
    with open(input_json_path, "r", encoding="utf-8") as f:
        tatqa_data = json.load(f)

    print(f"Processing {len(tatqa_data)} records...")
    examples = []
    for record in tatqa_data:
        table_text = table_to_text(record["table"]["table"])
        paragraphs = " ".join([p["text"].strip() for p in record["paragraphs"]])

        for question in record["questions"]:
            input_text = f"Table:\n{table_text}\n\nParagraphs:\n{paragraphs}\n\nQuestion:\n{question['question']}"
            instruction = "Answer the financial question based on the provided table and text."
            answer = str(question["answer"])

            example = {
                "instruction": instruction,
                "input": input_text,
                "output": answer
            }
            examples.append(example)

    print(f"Writing {len(examples)} Alpaca-formatted examples to {output_jsonl_path}")
    with open(output_jsonl_path, "w", encoding="utf-8") as out_f:
        for ex in examples:
            json.dump(ex, out_f, ensure_ascii=False)
            out_f.write("\n")

    print("Done!")



if __name__ == "__main__":
    base_dir = pl.Path(__file__).parent.resolve()
    # URLs of the raw JSON files
    tatqa_urls = {"train": "https://huggingface.co/datasets/next-tat/TAT-QA/resolve/main/tatqa_dataset_train.json",
                  "eval":  "https://huggingface.co/datasets/next-tat/TAT-QA/resolve/main/tatqa_dataset_dev.json"
                  }
    
    # Looping over the train and eval sets 
    for split in tatqa_urls.keys():
        json_path = base_dir / f"tatqa_original_{split}.json"
        jsonl_path = base_dir / f"tatqa_alpaca_{split}.jsonl"
        download_tatqa_json(tatqa_urls[split], json_path)           # Download
        convert_tatqa_to_alpaca(json_path, jsonl_path)              # Convert
    
    
    
    
    
    