#!/bin/bash
#SBATCH -J count_steps     
#SBATCH -t 02:00:00        
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=192 
#SBATCH --partition=genoa  


# --- Essential Paths ---
VENV_PATH="$HOME/thesis/venv"
SCRIPT_PATH="$HOME/thesis/step_counter.py"
DATA_PATH="/scratch-shared/$USER/data/shuffled_domain_adaptation_data.jsonl" 
MODEL_ID="meta-llama/Meta-Llama-3.1-8B"
# --- Training Params Needed for Calculation ---
MAX_SEQ_LEN=8192
PER_DEV_BS=8
GRAD_ACC=8
NUM_GPUS=1 # For the training step calculation (even if counting on CPU)
# --- End Essential Paths ---

# Ensure log directory exists
mkdir -p slurm_logs

echo "Job started: $(date)"
echo "Data path: ${DATA_PATH}"

# Load necessary modules (Python is usually essential)
module purge
module load 2023
module load Python/3.11.3-GCCcore-12.3.0
echo "Modules loaded."

# Activate Virtual Environment
source "${VENV_PATH}/bin/activate"
if [ $? -ne 0 ]; then echo "ERROR: Failed to activate venv ${VENV_PATH}."; exit 1; fi
echo "Venv activated."

# Run the script
# Automatically uses requested CPUs via SLURM_CPUS_PER_TASK env var if --num_workers omitted or set in script
python "$HOME/thesis/step_counter2.py" \
    --data_path "${DATA_PATH}" \
    --model_name_or_path "${MODEL_ID}" \
    --max_seq_length ${MAX_SEQ_LEN} \
    --per_device_batch_size ${PER_DEV_BS} \
    --gradient_accumulation_steps ${GRAD_ACC} \
    --num_gpus ${NUM_GPUS} \
    --trust_remote_code \
    --show_progress \
    --num_workers ${SLURM_CPUS_PER_TASK} 

echo "Job finished: $(date)"