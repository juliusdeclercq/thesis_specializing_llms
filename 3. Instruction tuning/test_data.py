# -*- coding: utf-8 -*-
"""
Created on Wed Jul 16 19:19:09 2025

@author: Julius de Clercq
"""


from unsloth import FastLanguageModel

import os
import json
import math
from dataclasses import dataclass, field
from datasets import load_dataset
from torch import bfloat16
from trl import SFTConfig, SFTTrainer
from transformers import (
    HfArgumentParser,
    AutoTokenizer,
    # TrainingArguments,
    # Trainer,
    default_data_collator
)

@dataclass
class ExperimentArguments:
    """
    Arguments corresponding to the experiments of the user
    """

    model_name_or_path: str = field(
        default=None,
        metadata={
            "help": "The model checkpoint or HuggingFace repo id to load the model from"},
    )
    data_dir: str = field(
        default=None,
        metadata={"help": "The directory or HuggingFace repo id to load the data from"},
    )
    train_data_path: str = field(
        default=None,
        metadata={"help": "Path to the training data."},
    )
    eval_data_path: str = field(
        default=None,
        metadata={"help": "Path to the evaluation data."},
    )
    from_foundation_model: str = field(
        default=True,
        metadata={
            "help": "Flag to specify whether the finetuning starts from a foundation model or a instruct-finetuned model"
        },
    )
    cache_dir: str = field(
        default=None,
        metadata={
            "help": "Where the model and tokenizer is or should be cached."
        },
    )
    
    def __post_init__(self):
        if self.model_name_or_path is None or self.data_dir is None:
            raise ValueError(
                f"Please specify the model and data! Received model: {self.model_name_or_path} and data: {self.data_dir}"
            )

    
#%%
def prepare_dataset(dataset, tokenizer, from_foundation_model=False):
    def make_prompt(instruction, input_):
        # Adjust this function if your actual prompt format is different
        if input_:
            prompt = f"### Instruction:\n{instruction}\n\n### Input:\n{input_}\n\n### Response:"
        else:
            prompt = f"### Instruction:\n{instruction}\n\n### Response:"
        return prompt
    
    # Dataset-specific function to convert the samples to the desired format
    def formatting_prompts_func(examples):
        texts = []
        for i in range(len(examples["instruction"])):
            instruction = examples["instruction"][i]
            input_ = examples.get("input", [""] * len(examples["instruction"]))[i]
            output = examples["output"][i]
            
            prompt = make_prompt(instruction, input_)
            # Combine prompt and output
            full_text = f"{prompt}\n{output}"
            texts.append(full_text)
        
        return {"text": texts}
    
    # Apply formatting to the dataset
    dataset = dataset.map(
        formatting_prompts_func, batched=True, num_proc=int(os.environ["SLURM_CPUS_PER_TASK"]) - 1
    )
    
    return dataset

#%%
def main(user_config, sft_config):
    # Load datasets
    train_dataset = load_dataset(
        "json",
        data_files={"train": user_config.train_data_path},
        split="train",
        streaming=False
    )
    eval_dataset = load_dataset(
        "json",
        data_files={"eval": user_config.eval_data_path},
        split="eval",
        streaming=False
    )
    
    
    
    print(tokenizer.encode("hello world!"))
    print(tokenizer.convert_ids_to_tokens([1,2,3]))
    
if __name__ == "__main__":
    # Parse both SFTConfig arguments and the extended model/training arguments
    parser = HfArgumentParser((ExperimentArguments, SFTConfig))
    user_config, sft_config = parser.parse_args_into_dataclasses()
    print(user_config, sft_config)
    main(user_config, sft_config)