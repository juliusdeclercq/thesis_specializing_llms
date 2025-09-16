# -*- coding: utf-8 -*-
"""
Created on Mon Apr 14 16:28:57 2025

@author: Julius de Clercq

Training script using unsloth. Based on template provided by SURF at: 
    https://github.com/sara-nl/LLM-finetune/blob/main/finetune_unsloth.py.
"""

import unsloth # Importing first because it patches the transformers module.
import torch # Importing just in case for dependencies
from dataclasses import dataclass, field


from transformers import (
    HfArgumentParser,
    TrainingArguments,
    Trainer,
    default_data_collator
)
from datasets import load_dataset
from typing import Optional



#%%             Checks





#%%

@dataclass
class ExperimentArguments:
    """Arguments for model and data specification."""
    pretrained_model_name_or_path: str = field(
        metadata={"help": "Model checkpoint or HuggingFace repo id."}
    )
    data_path: str = field(
        metadata={"help": "Path to the dataset or HuggingFace repo id."}
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


def main(exp_args: ExperimentArguments, train_args: TrainingArguments):
    """Main training function."""

    # Keep user's logic for handling max_seq_length via exp_args
    max_seq_length = exp_args.max_seq_length
    if max_seq_length is None:
        print("Warning: --max_seq_length not provided via CLI, defaulting to 8192")
        max_seq_length = 8192
    train_args.max_seq_length = max_seq_length

    # 1. Load pre-tokenized dataset (streaming for large datasets)
    dataset = load_dataset(
        "json",
        data_files={"train": exp_args.data_path},
        split="train",
        streaming=True
    )
    print("\n\n\nDataset loaded successfully.\n\n\n")

    # 2. Load model and tokenizer without quantization
    cache_kwargs = dict(cache_dir=exp_args.model_cache, local_files_only=True) if exp_args.use_model_cache else {}

    # Load tokenizer
    model, tokenizer = unsloth.FastLanguageModel.from_pretrained(
        model_name=exp_args.pretrained_model_name_or_path,
        max_seq_length = max_seq_length,
        dtype = torch.bfloat16,
        load_in_4bit = False,
        attn_implementation="flash_attention",  # This uses native Flash Attention 2.0 from PyTorch
        use_gradient_checkpointing="unsloth",
        **cache_kwargs
    )
    
    # Set padding token if not already set
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Enable gradient checkpointing for memory efficiency
    model.gradient_checkpointing_enable()

    # 3. Process pre-tokenized data
    def prepare_features(examples):
        """Convert pre-tokenized data to the expected format."""
        # The data already contains input_ids, attention_mask, and labels
        # Just ensure they're in the right format (tensors)
        return {
            "input_ids": examples["input_ids"],
            "attention_mask": examples["attention_mask"],
            "labels": examples["labels"]
        }

    # Apply the preparation function
    tokenized_dataset = dataset.map(
        prepare_features,
        batched=True,
        remove_columns=list(next(iter(dataset)).keys())  # Remove original columns after processing
    )

    # 4. Use default data collator since data is already tokenized
    data_collator = default_data_collator

    # 5. Initialize Trainer
    trainer = Trainer(
        model=model,
        args=train_args,
        train_dataset=tokenized_dataset,
        tokenizer=tokenizer,
        data_collator=data_collator,
    )

    # 6. Start Training
    print("\n\nStarting training...\n")
    print("Attention implementation:\n", model.config._attn_implementation, "\n\n")
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
        trainer.save_model()  # Saves full model weights
        tokenizer.save_pretrained(train_args.output_dir)  # Save tokenizer as well


if __name__ == "__main__":
    parser = HfArgumentParser((ExperimentArguments, TrainingArguments))
    exp_args, train_args = parser.parse_args_into_dataclasses()
    main(exp_args, train_args)