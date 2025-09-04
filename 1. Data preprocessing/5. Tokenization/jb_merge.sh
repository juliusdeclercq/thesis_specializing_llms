#!/bin/bash
#SBATCH -J merge_chunks
#SBATCH -t 48:00:00

#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --partition=genoa  
#SBATCH --ear=off

#SBATCH --mem=336G
#SBATCH --exclusive

#SBATCH --output=slurm_logs/merge/merge_%j.out


# This script is written because of OOM and "exceeding disk quota" errors occurring during merging of the worker files, containing the subsets of tokenized examples. 
# Tokenization was excessively memory intensive.   



# --- Configuration ---
SCRIPT_PATH="$HOME/thesis/merge_workers.py"
OUTPUT_DATA_DIR="/scratch-shared/$USER/data/tokenized"
NUM_WORKERS=96
FINAL_OUTPUT="2048_DA_data.jsonl"
# ---------------------


# --- Environment Setup ---
BASE_DIR="$HOME/thesis"
SCRATCH_DIR="/scratch-shared/$USER"
VENV_PATH="$HOME/thesis/venv"
CACHE_DIR="${HOME}/.cache/huggingface"

# --- End Setup ---


echo -e "\n\nJob started: $(date)"
echo "Output dir: ${OUTPUT_DATA_DIR}"
echo "Final output file: ${FINAL_OUTPUT}"
echo "Number of workers to merge: ${NUM_WORKERS}"
echo -e "Script: ${SCRIPT_PATH}\n\n"


# Load modules
module purge
module load 2023
module load Python/3.11.3-GCCcore-12.3.0


# Activate Virtual Environment
source "${VENV_PATH}/bin/activate"


# Set environment variables
KEYS_FILE="${BASE_DIR}/API_keys.json"
export HF_TOKEN=$(jq -r '.HF_token' "${KEYS_FILE}")
export HF_HOME="${CACHE_DIR}"
export HF_HUB_OFFLINE="1"

# Ensure output directory exists
mkdir -p "${OUTPUT_DATA_DIR}"


# Run the merge script
python "${SCRIPT_PATH}" \
    --output_dir "${OUTPUT_DATA_DIR}" \
    --num_workers ${NUM_WORKERS} \
    --final_output "${FINAL_OUTPUT}"


echo "Job finished: $(date)"