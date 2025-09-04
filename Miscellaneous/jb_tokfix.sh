#!/bin/sh
#SBATCH -J test
#SBATCH -t 1:00:00

#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1

#SBATCH --partition=gpu_h100
#SBATCH --gres=gpu:1         # Max 4 GPUs
#SBATCH --cpus-per-task=16   # 16 per GPU 

#SBATCH --ear=off

#SBATCH --output=slurm_logs/asdf/fixtok_%j.out
  
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


# The problem is that during instruction tuning, three special tokens were added to the adapter for some reason, which now mismatches with the original model size...
# Considered a few options but the cleanest is to just remove these special tokens. 


from transformers import AutoTokenizer
import os

# Paths
base_model_path = "/scratch-shared/jdclercq/models/model_12583757"
adapter_path = "/scratch-shared/jdclercq/models/instruct_13216005/checkpoint-830"

# Load the correct tokenizer from HuggingFace
print("Loading standard Llama 3.1 8B tokenizer from HuggingFace...")
tokenizer = AutoTokenizer.from_pretrained("meta-llama/Meta-Llama-3.1-8B")

# Set pad token to eos token (standard for Llama)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.pad_token_id = tokenizer.eos_token_id

print(f"Tokenizer vocab size: {len(tokenizer)}")
print(f"Pad token: '{tokenizer.pad_token}' (ID: {tokenizer.pad_token_id})")
print(f"EOS token: '{tokenizer.eos_token}' (ID: {tokenizer.eos_token_id})")
print(f"Special tokens: {tokenizer.special_tokens_map}")

# Save to both locations, overwriting existing tokenizers
print(f"\nOverwriting tokenizer in: {base_model_path}")
tokenizer.save_pretrained(base_model_path, safe_serialization=True)

print(f"Overwriting tokenizer in: {adapter_path}")
tokenizer.save_pretrained(adapter_path, safe_serialization=True)

print("\nDone! Both tokenizers have been replaced with the standard Llama 3.1 8B tokenizer.")

# Quick verification
print("\n=== VERIFICATION ===")
for path in [base_model_path, adapter_path]:
    test_tokenizer = AutoTokenizer.from_pretrained(path)
    print(f"{path}: vocab_size={len(test_tokenizer)}, pad_token='{test_tokenizer.pad_token}'")

######################################################################
######################################################################

EOF




