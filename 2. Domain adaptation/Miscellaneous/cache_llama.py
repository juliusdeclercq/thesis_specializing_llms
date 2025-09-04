from transformers import AutoTokenizer
import os

cache_dir = "/scratch-shared/jdclercq/.cache/huggingface"
model_id = "meta-llama/Meta-Llama-3.1-8B"

print(f"Downloading tokenizer for {model_id} to {cache_dir}")
tokenizer = AutoTokenizer.from_pretrained(
    model_id,
    cache_dir=cache_dir,
    trust_remote_code=True
)
print("Download complete!")