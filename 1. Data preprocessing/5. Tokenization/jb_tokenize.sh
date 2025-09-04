#!/bin/bash
#SBATCH -J tokenize
#SBATCH -t 36:00:00

#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=48
#SBATCH --partition=genoa  
#SBATCH --ear=off

#SBATCH --mem=336G
#SBATCH --exclusive

#SBATCH --output=slurm_logs/tokenize/tokenize_%j.out


# --- Configuration ---

export MAX_SEQ_LENGTH=2048
export STRIDE=128
export OUTPUT_FILE="${MAX_SEQ_LENGTH}_DA_data.jsonl"

MODEL_ID="meta-llama/Meta-Llama-3.1-8B"
SCRIPT_PATH="$HOME/thesis/pretokenizer.py"


# ---------------------


# --- Environment Setup ---
BASE_DIR="$HOME/thesis"
SCRATCH_DIR="/scratch-shared/$USER"
VENV_PATH="$HOME/thesis/venv"

INPUT_DATA_PATH="/projects/prjs1109/intermediate/subsetted_data.jsonl"
OUTPUT_DATA_DIR="${SCRATCH_DIR}/data/tokenized"
CACHE_DIR="${SCRATCH_DIR}/.cache/huggingface"


mkdir -p $OUTPUT_DATA_DIR

# --- End Setup ---


echo -e "\n\nJob started: $(date)"
echo "Input data: ${INPUT_DATA_PATH}"
echo "Output dir: ${OUTPUT_DATA_DIR}"
echo "Num workers: ${SLURM_CPUS_PER_TASK}"
echo -e "Model: ${MODEL_ID}\n\n"


# Load modules
module purge
module load 2023
module load Python/3.11.3-GCCcore-12.3.0
source "${VENV_PATH}/bin/activate"


# Set environment variables
KEYS_FILE="${BASE_DIR}/API_keys.json"
export HF_TOKEN=$(jq -r '.HF_token' "${KEYS_FILE}")
export HF_HOME="${CACHE_DIR}"
export MODEL_ID="${MODEL_ID}"
export CACHE_DIR="${CACHE_DIR}"

# export HF_HUB_OFFLINE="1"

# Create output directory
mkdir -p "${OUTPUT_DATA_DIR}"


# Run the preprocessing script
python "${SCRIPT_PATH}" \
    --input_path "${INPUT_DATA_PATH}" \
    --output_dir "${OUTPUT_DATA_DIR}" \
    --model_name_or_path "${MODEL_ID}" \
    --cache_dir "${CACHE_DIR}" \
    --num_workers ${SLURM_CPUS_PER_TASK} \
    --tqdm_step 500


echo "Job finished: $(date)"