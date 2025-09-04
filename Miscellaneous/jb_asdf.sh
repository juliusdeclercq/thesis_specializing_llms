#!/bin/sh
#SBATCH -J test
#SBATCH -t 1:00:00

#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1

#SBATCH --partition=gpu_h100
#SBATCH --gres=gpu:1         # Max 4 GPUs
#SBATCH --cpus-per-task=16   # 16 per GPU 

#SBATCH --ear=off

#SBATCH --output=slurm_logs/asdf/adapter_fix_%j.out
  
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
# FIXING TOKENIZER
# ──────────────────────────────

python << EOF

######################################################################
######################################################################


# The adapter itself seems to suffer from this mismatch. Let's check what's up. 


import torch
from safetensors.torch import load_file, save_file
import shutil
from datetime import datetime

adapter_path = "/scratch-shared/jdclercq/models/instruct_13216005/checkpoint-830"

# Backup first
backup_suffix = datetime.now().strftime("%Y%m%d_%H%M%S")
backup_path = f"{adapter_path}/adapter_model.safetensors.backup_{backup_suffix}"
shutil.copy2(f"{adapter_path}/adapter_model.safetensors", backup_path)
print(f"Created backup: {backup_path}")

# Load the adapter weights
state_dict = load_file(f"{adapter_path}/adapter_model.safetensors")

print(f"Original adapter has {len(state_dict)} weights")

# Remove embed_tokens and lm_head weights
removed = []
for key in list(state_dict.keys()):
    if "embed_tokens" in key or "lm_head" in key:
        del state_dict[key]
        removed.append(key)

print(f"Removed {len(removed)} weights:")
for key in removed:
    print(f"  - {key}")

print(f"Cleaned adapter has {len(state_dict)} weights (should be 448 LoRA weights)")

# Verify we only have LoRA weights left
non_lora = [k for k in state_dict.keys() if "lora_A" not in k and "lora_B" not in k]
if non_lora:
    print(f"WARNING: Found {len(non_lora)} non-LoRA weights remaining")
else:
    print("✓ Only LoRA weights remain")

# Save the cleaned adapter
save_file(state_dict, f"{adapter_path}/adapter_model.safetensors")
print(f"\nSaved cleaned adapter to: {adapter_path}/adapter_model.safetensors")
print("Your LoRA training is preserved, just removed the problematic full weights")

######################################################################
######################################################################

EOF




