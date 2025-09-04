#!/bin/sh
#SBATCH -J test
#SBATCH -t 1:00:00

#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1

#SBATCH --partition=gpu_h100
#SBATCH --gres=gpu:1         # Max 4 GPUs
#SBATCH --cpus-per-task=16   # 16 per GPU 

#SBATCH --ear=off

#SBATCH --output=slurm_logs/asdf/adapter_check_%j.out
  
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


import json
from safetensors import safe_open

# adapter_path = "/scratch-shared/jdclercq/models/instruct_13216005/checkpoint-830"
adapter_path = "/scratch-shared/jdclercq/models/instruct_13215997/checkpoint-830"

# Check adapter config
print("=== ADAPTER CONFIG ===")
with open(f"{adapter_path}/adapter_config.json", "r") as f:
    config = json.load(f)
    print(f"Target modules: {config.get('target_modules')}")
    print(f"Modules to save: {config.get('modules_to_save')}")
    print(f"Task type: {config.get('task_type')}")

# Check what's in the adapter
print("\n=== WEIGHTS IN ADAPTER ===")
with safe_open(f"{adapter_path}/adapter_model.safetensors", framework="pt") as f:
    lora_weights = []
    full_weights = []
    
    for key in f.keys():
        if "lora_A" in key or "lora_B" in key:
            lora_weights.append(key)
        else:
            full_weights.append((key, f.get_tensor(key).shape))
    
    print(f"\nLoRA weights: {len(lora_weights)} modules")
    print(f"Full weights: {len(full_weights)} modules")
    
    if full_weights:
        print("\nFull weights found:")
        for key, shape in full_weights:
            print(f"  - {key}: {shape}")

######################################################################
######################################################################

EOF




