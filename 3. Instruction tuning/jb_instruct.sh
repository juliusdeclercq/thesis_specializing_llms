#!/bin/sh
#SBATCH -J instruct_tune
#SBATCH -t 36:00:00

#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1

#SBATCH --partition=gpu_h100
#SBATCH --gres=gpu:1         # Max 4 GPUs
#SBATCH --cpus-per-task=16   # 16 per GPU 

#SBATCH --ear=off

#SBATCH --output=slurm_logs/instruct/instruct_%j.out
  
# Logging commands. Remove or add SBATCH to (de)activate. 
#SBATCH --mail-type=END
#SBATCH --mail-user=j.l.h.de.clercq@businessdatascience.nl


echo -e "\nInstruction tuning script initialized with ${SLURM_GPUS_ON_NODE} GPUs.\n" 


# ──────────────────────────────
# DIRECTORIES
# ──────────────────────────────

BASE=$HOME/thesis
SCRATCH_DIR="/scratch-shared/$USER"
MODEL_OUTPUT_DIR=$SCRATCH_DIR/models/instruct_$SLURM_JOBID
mkdir -p $MODEL_OUTPUT_DIR


# ──────────────────────────────
# PYTHON ENVIRONMENT
# ──────────────────────────────

module purge
module load 2023
module load Python/3.11.3-GCCcore-12.3.0
module load CUDA/12.4.0
source "${BASE}/venv/bin/activate"


# ──────────────────────────────
# CONFIGURATION
# ──────────────────────────────

# Model and data configuration
MODEL_NAME="$SCRATCH_DIR/models/model_13219561"
# MODEL_NAME="meta-llama/Meta-Llama-3.1-8B"

DATA_DIR="${SCRATCH_DIR}/data"  # Directory containing fin_instruct_train.jsonl and fin_instruct_eval.jsonl
FROM_FOUNDATION_MODEL=true

# Training configuration
NUM_TRAIN_EPOCHS=10
BATCH_SIZE=32
GRAD_ACC_STEPS=2
EVAL_BATCH_SIZE=32
MAX_SEQ_LENGTH=4096

# LoRA configuration
export LORA_RANK=256  # Adjust based on your needs

# Learning rate configuration
LEARNING_RATE=2e-4
WARMUP_RATIO=0.05
WEIGHT_DECAY=0.01


# ──────────────────────────────
# ENVIRONMENT VARIABLES
# ──────────────────────────────

# Variables for Huggingface and WandB
CONFIG_FILE="${BASE}/acc_config_${SLURM_GPUS_ON_NODE}.yaml"
KEYS_FILE="${BASE}/API_keys.json"

export HF_TOKEN=$(jq -r '.HF_token' "${KEYS_FILE}")
export WANDB_API_KEY=$(jq -r '.WandB_API_key' "${KEYS_FILE}")
export CACHE_DIR="${HOME}/.cache/huggingface"  
export HF_HOME=$CACHE_DIR

# Variables for accelerate launch
export MASTER_ADDR=$(scontrol show hostnames $SLURM_JOB_NODELIST | head -n 1)
export MASTER_PORT=33333
export WORLD_SIZE=$SLURM_NTASKS
export RANK=$SLURM_PROCID
export LOCAL_RANK=0

# CUDA configuration
export CUDA_LAUNCH_BLOCKING=1
export TORCH_USE_RTLD_GLOBAL=YES
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True


# ──────────────────────────────
# EXECUTION
# ──────────────────────────────

echo -e "\n\nModel = ${MODEL_NAME}"
echo -e "Data directory = ${DATA_DIR}"
echo -e "Training epochs = ${NUM_TRAIN_EPOCHS}"
echo -e "LoRA rank = ${LORA_RANK}\n\n"

echo -e "Launching instruction tuning script.\n\n"

accelerate launch --config_file $CONFIG_FILE $BASE/instruct_tune.py \
    --model_name_or_path $MODEL_NAME \
    --data_dir $DATA_DIR \
    --cache_dir $CACHE_DIR \
    --train_data_path "$DATA_DIR/fin_instruct_train.jsonl" \
    --eval_data_path "$DATA_DIR/fin_instruct_eval.jsonl" \
    --from_foundation_model $FROM_FOUNDATION_MODEL \
    --output_dir $MODEL_OUTPUT_DIR \
    --bf16 True \
    --max_seq_length $MAX_SEQ_LENGTH \
    --per_device_train_batch_size $BATCH_SIZE \
    --gradient_accumulation_steps $GRAD_ACC_STEPS \
    --num_train_epochs $NUM_TRAIN_EPOCHS \
    --gradient_checkpointing True \
    --logging_steps 10 \
    --save_strategy epoch \
    --eval_strategy epoch \
    --per_device_eval_batch_size $EVAL_BATCH_SIZE \
    --prediction_loss_only \
    --save_total_limit 15 \
    --optim adamw_torch \
    --report_to wandb \
    --learning_rate $LEARNING_RATE \
    --warmup_ratio $WARMUP_RATIO \
    --weight_decay $WEIGHT_DECAY \
    --adam_beta1 0.9 \
    --adam_beta2 0.95 \
    --max_grad_norm 1.0 \
    --lr_scheduler_type cosine \
    --load_best_model_at_end True \
    --metric_for_best_model loss