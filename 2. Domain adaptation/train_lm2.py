# train_lm.py

# -*- coding: utf-8 -*-
"""
Created on Mon Apr 14 16:28:57 2025

@author: Julius de Clercq
"""


#%%             Monkey-patching progress bar


# # Monkey-patching progress bar. This still doesn't work though...
# import os
# import tqdm
# import tqdm.auto

# class custom_tqdm(tqdm.tqdm):
#     def __init__(self, *args, **kwargs):
#         kwargs.setdefault("miniters", int(os.environ["TQDM_MINITERS"]))
#         super().__init__(*args, **kwargs)

# # Patch tqdm globally inside transformers
# # Overriding progress bars to only print every TQDM_MINITERS iteration, set as global in the job script.

# # Patching all tqdm implementations to make sure I also patch the right one (spray and pray strategy)
# import tqdm.auto
# import tqdm.std
# import tqdm.notebook
# import tqdm.rich
# import transformers.trainer_callback
# transformers.trainer_callback.logging.tqdm = transformers.trainer_callback.logging.tqdm_lib = custom_tqdm
# tqdm.tqdm = tqdm.auto.tqdm = tqdm.std.tqdm = tqdm.notebook.tqdm = tqdm.rich.tqdm = tqdm.auto.tqdm = tqdm.tqdm = custom_tqdm





#%%             Imports
import unsloth
import torch
from dataclasses import dataclass, field
from transformers import (
    HfArgumentParser,
    TrainingArguments,
    Trainer,
    default_data_collator
)
# Changing HF verbosity level to 'info' for more details. 
import transformers.utils.logging
transformers.utils.logging.set_verbosity_info()
from datasets import load_dataset
from typing import Optional
import os
import json
import math


#%%             Diagnostics

# Space reserved for diagnostic print statements

#%%             Classes

@dataclass
class ExperimentArguments:
    """Arguments for model and data specification."""
    pretrained_model_name_or_path: str = field(
        metadata={"help": "Model checkpoint or HuggingFace repo id."}
    )
    data_path: str = field(
        metadata={"help": "Path to the training dataset or HuggingFace repo id."}
    )
    eval_data_path: str = field(  # Added for eval dataset path
        metadata={"help": "Path to the evaluation dataset."}
    )
    use_model_cache: bool = field(
        metadata={"help": "Specifying whether to use the cached models or not."}
    )
    model_cache: str = field(
        metadata={"help": "Directory with cached Llama3.1 models on Snellius."}
    )
    max_seq_length: Optional[int] = field(
        metadata={"help": "Maximum sequence length for model, tokenizer, and training."}
    )


class CustomTrainer(Trainer):
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

#%%             Main

def main(exp_args: ExperimentArguments, train_args: TrainingArguments):
    max_seq_length = exp_args.max_seq_length
    if max_seq_length is None:
        print("Warning: --max_seq_length not provided via CLI, defaulting to 8192")
        max_seq_length = 8192
    train_args.max_seq_length = max_seq_length

    # 1. Load pre-tokenized train and eval datasets (streaming for large datasets)
    train_dataset = load_dataset(
        "json",
        data_files={"train": exp_args.data_path},
        split="train",
        streaming=True
    )
    eval_dataset = load_dataset(
        "json",
        data_files={"eval": exp_args.eval_data_path},
        split="eval",
        streaming=False
    )
    print("\n\n\nTrain and Eval Datasets loaded successfully.\n\n\n")

    # Check if resuming from a checkpoint and skip examples if necessary
    processed_examples = 0
    if train_args.resume_from_checkpoint:
        checkpoint_dir = train_args.resume_from_checkpoint
        position_file = os.path.join(checkpoint_dir, "data_stream_position.json")
        if os.path.exists(position_file):
            with open(position_file, "r") as f:
                position_data = json.load(f)
                processed_examples = position_data.get("processed_examples", 0)
            print(f"Resuming from checkpoint. Skipping {processed_examples} examples in data stream.")
            train_dataset = train_dataset.skip(processed_examples)
        else:
            print(f"Warning: No data stream position file found at {position_file}. Starting from beginning of dataset.")

    # 2. Load model and tokenizer without quantization
    cache_kwargs = dict(cache_dir=exp_args.model_cache, local_files_only=True) if exp_args.use_model_cache else {}
    model, tokenizer = unsloth.FastLanguageModel.from_pretrained(
        model_name=exp_args.pretrained_model_name_or_path,
        max_seq_length=max_seq_length,
        dtype=torch.bfloat16,
        load_in_4bit=False,
        attn_implementation="flash_attention",
        use_gradient_checkpointing="unsloth",
        **cache_kwargs
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model.gradient_checkpointing_enable()
    
    
    
# =============================================================================
#     # Checking which parameters are frozen.
#     for name, param in model.named_parameters():
#         if not param.requires_grad:
#             print(f"FROZEN: {name}, shape={param.shape}")
#     
#     exit(0) 
# =============================================================================
    
    # Unfreezing all parameters. 
    for param in model.parameters():
        param.requires_grad = True
    
    
    # 3. Process pre-tokenized data
    def prepare_features(examples):
        return {
            "input_ids": examples["input_ids"],
            "attention_mask": examples["attention_mask"],
            "labels": examples["labels"]
        }

    train_dataset = train_dataset.map(prepare_features, batched=True, remove_columns=list(next(iter(train_dataset)).keys()))
    eval_dataset = eval_dataset.map(prepare_features, batched=True, remove_columns=list(eval_dataset.features.keys()))   

    # 4. Using the default data collator since data is already tokenized.
    data_collator = default_data_collator

    # 5. Initialize the Trainer
    trainer = CustomTrainer(
        model=model,
        args=train_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
        data_collator=data_collator,
    )

    # 6. Start Training
    print("\n\nStarting training...\n")
    print("Attention implementation:", model.config._attn_implementation, "\n\n")
    trainer_stats = trainer.train()

    # 7. Print Training Stats
    print("\nTraining finished.\n")
    print(trainer_stats, "\n\n")
    try:
        train_time = trainer_stats.metrics['train_runtime']
        print(f"{train_time:.2f} seconds used for training.")
        print(f"{train_time/60:.2f} minutes used for training.")
    except KeyError:
        print("Could not extract train_runtime from trainer_stats.")

    # 8. Save model
    if train_args.output_dir:
        print(f"Saving final model checkpoint to {train_args.output_dir}")
        trainer.save_model()
        tokenizer.save_pretrained(train_args.output_dir)

if __name__ == "__main__":
    parser = HfArgumentParser((ExperimentArguments, TrainingArguments))
    exp_args, train_args = parser.parse_args_into_dataclasses()
    main(exp_args, train_args)