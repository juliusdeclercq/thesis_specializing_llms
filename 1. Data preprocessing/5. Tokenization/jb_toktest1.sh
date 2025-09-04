#!/bin/sh
#SBATCH -J test
#SBATCH -t 1:00:00

#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1

#SBATCH --partition=gpu_h100
#SBATCH --gres=gpu:1         # Max 4 GPUs
#SBATCH --cpus-per-task=16   # 16 per GPU 

#SBATCH --ear=off

#SBATCH --output=slurm_logs/asdf/toktest_%j.out
  
  # Logging commands. Remove or add SBATCH to (de)activate. 
# --mail-type=END
#SBATCH --mail-user=j.l.h.de.clercq@businessdatascience.nl


# echo -e "\nTraining script initialized with ${SLURM_GPUS_ON_NODE} GPUs.\n" 


# ──────────────────────────────
# DIRECTORIES
# ──────────────────────────────

BASE=$HOME/thesis
SCRATCH_DIR="/scratch-shared/$USER"
MODEL_OUTPUT_DIR=$SCRATCH_DIR/models/instruct_model_$SLURM_JOBID
export CACHE_DIR="${SCRATCH_DIR}/huggingface"  
export HF_HOME=$CACHE_DIR
mkdir -p $MODEL_OUTPUT_DIR


# ──────────────────────────────
# PYTHON ENVIRONMENT
# ──────────────────────────────

module purge
module load 2023
module load Python/3.11.3-GCCcore-12.3.0
module load CUDA/12.4.0
source "${BASE}/venv-pixiu/bin/activate"


# ──────────────────────────────
# TESTING TOKENIZER
# ──────────────────────────────

python << EOF

# Investigation script - save as check_mismatch.py
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from safetensors import safe_open

# Paths
base_model_path = "/scratch-shared/jdclercq/models/model_12583757"
adapter_path = "/scratch-shared/jdclercq/models/instruct_13216005/checkpoint-830"

# 1. Check the base model embedding size
print("=== BASE MODEL ===")
model = AutoModelForCausalLM.from_pretrained(base_model_path, device_map="cpu")
print(f"Model embedding size: {model.get_input_embeddings().weight.shape}")
del model  # Free memory

# 2. Check what the adapter expects
print("\n=== ADAPTER ===")
# Look at the adapter safetensors file directly
with safe_open(f"{adapter_path}/adapter_model.safetensors", framework="pt") as f:
    for key in f.keys():
        if "embed_tokens" in key:
            print(f"Found embedding-related key: {key}, shape: {f.get_tensor(key).shape}")

# 3. Check tokenizers
print("\n=== TOKENIZERS ===")
base_tokenizer = AutoTokenizer.from_pretrained(base_model_path)
adapter_tokenizer = AutoTokenizer.from_pretrained(adapter_path)
print(f"Base model tokenizer size: {len(base_tokenizer)}")
print(f"Adapter tokenizer size: {len(adapter_tokenizer)}")

# 4. Check the adapter config
print("\n=== ADAPTER CONFIG ===")
import json
with open(f"{adapter_path}/adapter_config.json", "r") as f:
    config = json.load(f)
    print(f"Base model name in adapter config: {config.get('base_model_name_or_path', 'Not found')}")
    print(f"Target modules: {config.get('target_modules', 'Not found')}")


EOF




