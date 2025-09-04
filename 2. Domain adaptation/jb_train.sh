#!/bin/sh
#SBATCH -J train
#SBATCH -t 120:00:00

#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1

#SBATCH --partition=gpu_h100
#SBATCH --gres=gpu:1         # Max 4 GPUs
#SBATCH --cpus-per-task=16   # 16 per GPU 

#SBATCH --ear=off

#SBATCH --output=slurm_logs/train/train_%j.out
  
  # Logging commands. Remove or add SBATCH to (de)activate. 
#SBATCH --mail-type=END
#SBATCH --mail-user=j.l.h.de.clercq@businessdatascience.nl


echo -e "\nTraining script initialized with ${SLURM_GPUS_ON_NODE} GPUs.\n" 


# Setting directories
BASE=$HOME/thesis
SCRATCH_DIR="/scratch-shared/$USER"
MODEL_OUTPUT_DIR=$SCRATCH_DIR/models/model_$SLURM_JOBID
mkdir -p $MODEL_OUTPUT_DIR

# Set up Python environment.
module purge
module load 2023
module load Python/3.11.3-GCCcore-12.3.0
module load CUDA/12.4.0
source "${BASE}/venv/bin/activate"


# ====================================================== #
# ================    CONFIGURATION    ================= #

MODEL_NAME="meta-llama/Meta-Llama-3.1-8B"

DATA_PATH="${SCRATCH_DIR}/data/2048_DA_data_train.jsonl"
EVAL_DATA_PATH="${SCRATCH_DIR}/data/2048_DA_data_eval.jsonl"

MAX_SEQ_LENGTH=2048
MAX_TRAINING_STEPS=20000


BATCH_SIZE=64
GRAD_ACC_STEPS=1
EVAL_BATCH_SIZE=2

# CHECKPOINT=$SCRATCH_DIR/models/model_12431715/checkpoint-2000
# export TQDM_MINITERS=2


# ====================================================== #
# ============    ENVIRONMENT VARIABLES    ============= #

# Variables for Huggingface and WandB
CONFIG_FILE="${BASE}/acc_config_${SLURM_GPUS_ON_NODE}.yaml" # 1, 2 or 4, denoting number of GPUs
KEYS_FILE="${BASE}/API_keys.json"

export HF_TOKEN=$(jq -r '.HF_token' "${KEYS_FILE}")
export WANDB_API_KEY=$(jq -r '.WandB_API_key' "${KEYS_FILE}")
export CACHE_DIR="${HOME}/.cache/huggingface"  
export HF_HOME=$CACHE_DIR

# Forcing offline mode to force usage of cached model
#export HF_HUB_OFFLINE=1
#export TRANSFORMERS_OFFLINE=1

# Variables for accelerate launch
export MASTER_ADDR=$(scontrol show hostnames $SLURM_JOB_NODELIST | head -n 1)
export MASTER_PORT=33333
export WORLD_SIZE=$SLURM_NTASKS
export RANK=$SLURM_PROCID

# Making sure that CUDA throws errors in case of hardware crashes
export CUDA_LAUNCH_BLOCKING=1
# export TORCH_USE_RTLD_GLOBAL=YES
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True # This minimizes memory fragmentation to avoid OOM errors.

echo -e "\n\nModel = ${MODEL_NAME} \nData = ${DATA_PATH} \nSteps = ${MAX_TRAINING_STEPS}\n\n"


# ====================================================== #
# ==================    EXECUTION    =================== #

# Run script using accelerate launch
echo -e "\n\nLaunching train script.\n\n"
accelerate launch --config_file $CONFIG_FILE $BASE/train_lm2.py \
    --pretrained_model_name_or_path $MODEL_NAME \
    --data_path $DATA_PATH \
    --eval_data_path $EVAL_DATA_PATH \
    --output_dir $MODEL_OUTPUT_DIR \
    --bf16 True \
    --max_seq_length $MAX_SEQ_LENGTH \
    --per_device_train_batch_size $BATCH_SIZE \
    --gradient_accumulation_steps $GRAD_ACC_STEPS \
    --max_steps $MAX_TRAINING_STEPS \
    --gradient_checkpointing True \
    --logging_steps 10 \
    --save_steps 250 \
    --eval_steps 1000 \
    --eval_strategy steps \
    --per_device_eval_batch_size $EVAL_BATCH_SIZE \
    --prediction_loss_only \
    --save_total_limit 30 \
    --optim adamw_torch \
    --report_to wandb \
    --model_cache $CACHE_DIR \
    --learning_rate 2e-4 \
    --warmup_ratio 0.05 \
    --weight_decay 0.01 \
    --adam_beta1 0.9 \
    --adam_beta2 0.95 \
    --max_grad_norm 1.0 \
    --lr_scheduler_type cosine \
    
# --resume_from_checkpoint $CHECKPOINT
