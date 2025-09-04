#!/bin/sh
#SBATCH -J eval
#SBATCH -t 3:00:00

#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1

#SBATCH --partition=gpu_h100
#SBATCH --gres=gpu:1         # Max 4 GPUs
#SBATCH --cpus-per-task=16   # 16 per GPU 

#SBATCH --ear=off

#SBATCH --output=slurm_logs/eval/eval_%j.out
  
  # Logging commands. Remove or add SBATCH to (de)activate. 
# --mail-type=END
# --mail-user=j.l.h.de.clercq@businessdatascience.nl

START=$(date)
echo -e "\nStarted evaluation job at $START\n"
# Setting directories
BASE=$HOME/thesis
SCRATCH_DIR="/scratch-shared/$USER"

# Set up Python environment.
module purge
module load 2023
module load Python/3.11.3-GCCcore-12.3.0 CUDA/12.4.0
source "${BASE}/venv-pixiu/bin/activate"   # This is the FLARE/PIXIU environment.


# ====================================================== #
# ================    CONFIGURATION    ================= #



# Base Llama 3 8B
#MODEL_NAME="meta-llama/Meta-Llama-3-8B"

# Base Llama 3.1 8B
#MODEL_NAME="meta-llama/Meta-Llama-3.1-8B"
#ADAPTER_CHECKPOINT_PATH="$SCRATCH_DIR/models/instruct_13215997/checkpoint-830"

# DA1: Domain adapted with MSL = 8192
MODEL_NAME="$SCRATCH_DIR/models/model_12583757"
ADAPTER_CHECKPOINT_PATH="$SCRATCH_DIR/models/instruct_13216005/checkpoint-830"

# DA2: Domain adapted with MSL = 2048
#MODEL_NAME="$SCRATCH_DIR/models/model_13219561"
#ADAPTER_CHECKPOINT_PATH="$SCRATCH_DIR/models/instruct_XXXXXXXXXXX/checkpoint-830"


EVAL_OUTPUT_DIR="$BASE/eval-output"
export BATCH_SIZE=256

mkdir -p $EVAL_OUTPUT_DIR
echo -e "\n\nModel name = $MODEL_NAME"
echo -e "LoRA Adapter = $ADAPTER_CHECKPOINT_PATH"
echo -e "Batch Size = $BATCH_SIZE\n\n"


# ====================================================== #
# ============    ENVIRONMENT VARIABLES    ============= #

# Variables for Huggingface and WandB
CONFIG_FILE="${BASE}/acc_config_${SLURM_GPUS_ON_NODE}.yaml" # 1, 2 or 4, denoting number of GPUs
KEYS_FILE="${BASE}/API_keys.json"

export HF_TOKEN=$(jq -r '.HF_token' "${KEYS_FILE}")
export WANDB_API_KEY=$(jq -r '.WandB_API_key' "${KEYS_FILE}")
export HF_HOME="$SCRATCH_DIR/huggingface"  


# ====================================================== #
# ==================    EXECUTION    =================== #
echo -e "\n\nLaunching evaluation script.\n\n"

python PIXIU/src/eval.py \
  --model "hf-causal-llama" \
  --model_args "pretrained=$MODEL_NAME,use_fast=True,peft=$ADAPTER_CHECKPOINT_PATH" \
  --tasks "flare_ner,flare_fpb,flare_fiqasa,flare_headlines" \
  --num_fewshot 5 \
  --batch_size $BATCH_SIZE\
  --output_base_path $EVAL_OUTPUT_DIR \
  --device "cuda:0" \
  --write_out \
  --no_cache

echo -e "\nEvaluation job started at $START"
echo -e "Evaluation job finished at $(date)\n"





