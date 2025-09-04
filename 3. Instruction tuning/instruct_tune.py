# -*- coding: utf-8 -*-
"""
Created on Tue Jul  8 09:55:06 2025

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
    DataCollatorForLanguageModeling,
    # AutoTokenizer,
    # TrainingArguments,
    # Trainer,
)
# from unsloth.chat_templates import get_chat_template


@dataclass
class ExperimentArguments:
    """
    Arguments corresponding to the experiments of the user
    """
    use_model_cache: bool = field(
        metadata={"help": "Specifying whether to use the cached models or not."}
    )
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

class CustomTrainer(SFTTrainer):
    """Custom Trainer to track and save data stream position at checkpoints."""
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.processed_examples = 0

    def _save_checkpoint(self, model, trial=None):
        super()._save_checkpoint(model, trial)
        checkpoint_dir = os.path.join(self.args.output_dir, f"checkpoint-{self.state.global_step}")
        position_data = {"processed_examples": self.processed_examples}
        with open(os.path.join(checkpoint_dir, "data_stream_position.json"), "w") as f:
            json.dump(position_data, f)
        print(f"Saved data stream position: {self.processed_examples} examples at {checkpoint_dir}")

    def training_step(self, model, inputs, *args, **kwargs):
        batch_size = inputs["input_ids"].shape[0] if "input_ids" in inputs else 1
        self.processed_examples += batch_size
        return super().training_step(model, inputs)

    def evaluate(self, *args, **kwargs):  # Added custom evaluation metrics
        """Override evaluate to compute perplexity and cross-entropy."""
        metrics = super().evaluate(*args, **kwargs)
        if "eval_loss" in metrics:
            metrics["eval_perplexity"] = math.exp(metrics["eval_loss"])
            metrics["eval_cross_entropy"] = metrics["eval_loss"]
        return metrics
    
#%%
def prepare_dataset(dataset, tokenizer, max_seq_length):
    def make_prompt(instruction, input_):
        if input_:
            return f"### Instruction:\n{instruction}\n\n### Input:\n{input_}\n\n### Response:"
        else:
            return f"### Instruction:\n{instruction}\n\n### Response:"

    def preprocess(example):
        instruction = example["instruction"]
        input_ = example.get("input", "")
        output = example["output"]

        prompt = make_prompt(instruction, input_)
        # Tokenize prompt and prompt+output, NO special tokens to ensure alignment
        prompt_ids = tokenizer(prompt, add_special_tokens=False).input_ids
        output_ids = tokenizer("\n" + output, add_special_tokens=False).input_ids

        # Optionally, add BOS token if model expects it (often True for Llama-3)
        bos = [tokenizer.bos_token_id] if tokenizer.bos_token_id is not None else []
        input_ids = bos + prompt_ids + output_ids
        labels    = [-100]*(len(bos) + len(prompt_ids)) + output_ids

        # Now, crop to max_seq_length
        input_ids = input_ids[:max_seq_length]
        labels    = labels[:max_seq_length]
        attn_mask = [1] * len(input_ids)
        # Pad if necessary
        pad_len = max_seq_length - len(input_ids)
        if pad_len > 0:
            input_ids += [tokenizer.pad_token_id] * pad_len
            labels    += [-100] * pad_len
            attn_mask += [0] * pad_len

        return {
            "input_ids": input_ids,
            "attention_mask": attn_mask,
            "labels": labels,
        }

    processed_dataset = dataset.map(
        preprocess,
        remove_columns=dataset.column_names,
        num_proc=1,
        desc="Tokenizing and masking dataset"
    )
    return processed_dataset


def apply_qlora(model, max_seq_length):
    # Do model patching and add fast LoRA weights
    
    lora_rank = int(os.environ["LORA_RANK"])
    model = FastLanguageModel.get_peft_model(
        model,
        r = lora_rank,  # rank of parameters. Higher R means more parameters. Setting this as env variable in job script. 
        target_modules=[        # 
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ],
        lora_alpha = lora_rank * 2,  # Scaling of the weights, proporational to the rank. 
        lora_dropout=0,  # Dropout = 0 is currently optimized
        bias="none",  # Bias = "none" is currently optimized
        use_gradient_checkpointing="unsloth",
        max_seq_length=max_seq_length,
        random_state=47,
    )

    return model

#%%
def main(exp_args, train_args):
    # Load datasets
    train_dataset = load_dataset(
        "json",
        data_files={"train": exp_args.train_data_path},
        split="train",
        streaming=False
    )
    eval_dataset = load_dataset(
        "json",
        data_files={"eval": exp_args.eval_data_path},
        split="eval",
        streaming=False
    )

    # Load model
    cache_kwargs = dict(cache_dir=exp_args.model_cache, local_files_only=True) if exp_args.use_model_cache else {}
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=exp_args.model_name_or_path,
        max_seq_length=train_args.max_seq_length,
        dtype=bfloat16,
        load_in_4bit=False,
        attn_implementation="flash_attention",
        use_gradient_checkpointing="unsloth",
        **cache_kwargs
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model.gradient_checkpointing_enable()
    
    
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    

    # Map the datasets to the right template
    train_dataset = prepare_dataset(train_dataset, tokenizer, train_args.max_seq_length)
    eval_dataset  = prepare_dataset(eval_dataset, tokenizer, train_args.max_seq_length)
    # Pass the key of the dataset object
    train_args.dataset_text_field = "text"
    
# =============================================================================
#     # Sanity checks for the example labels
#     for i in range(20):
#         ex = train_dataset[i]
#         input_ids = ex["input_ids"]
#         labels = ex["labels"]
#         print(f"Example {i} len(input_ids): {len(input_ids)} len(labels): {len(labels)} unique labels: {set(labels)}")
#         # Print where the first non--100 label is
#         for idx, lab in enumerate(labels):
#             if lab != -100:
#                 print("First response token at pos", idx)
#                 break
#         if all([l == -100 for l in labels]):
#             print("WARNING: Entire label is masked!")
#             raise ValueError("\n\nLabels are all masked!!!\n\n")
# =============================================================================
            
    # Patch the model with parameter-efficient finetuning
    model = apply_qlora(model, train_args.max_seq_length)
    
    
    # Use a data collator that pads
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False,
        )
    
    trainer = CustomTrainer(
        model=model,
        args=train_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
        data_collator=data_collator
    )

    trainer_stats = trainer.train()
    print(trainer_stats)

    print(f"{trainer_stats.metrics['train_runtime']} seconds used for training.")
    print(
        f"{round(trainer_stats.metrics['train_runtime']/60, 2)} minutes used for training."
    )


if __name__ == "__main__":
    # Parse both SFTConfig arguments and the extended model/training arguments
    parser = HfArgumentParser((ExperimentArguments, SFTConfig))
    exp_args, train_args = parser.parse_args_into_dataclasses()
    print(exp_args, train_args)
    main(exp_args, train_args)