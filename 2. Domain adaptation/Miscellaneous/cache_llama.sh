#!/bin/bash

# Load modules
module load 2023
module load Python/3.11.3-GCCcore-12.3.0

# Activate virtual environment
source $HOME/thesis/venv/bin/activate

# Set HuggingFace token
export HF_TOKEN=$(jq -r '.HF_token' "$HOME/thesis/API_keys.json")

# Download model and tokenizer
python - << 'EOF'
from transformers import AutoModelForCausalLM, AutoTokenizer
import os

cache_dir = os.path("/scratch-shared/jdclercq/.cache/huggingface")
model_id = "meta-llama/Meta-Llama-3.1-8B"

print(f"Downloading model and tokenizer for {model_id} to {cache_dir}")
print("This will take a while (~16GB)...\n")

# Download tokenizer
print("Downloading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(
    model_id,
    cache_dir=cache_dir,
    trust_remote_code=True
)
print("✓ Tokenizer downloaded\n")

# Download model
print("Downloading model weights...")
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    cache_dir=cache_dir,
    trust_remote_code=True,
    torch_dtype="auto"
)
print("✓ Model downloaded")
print(f"\nDownload complete! Cache size: ~16GB in {cache_dir}")
print("\n\n\n")
EOF