#!/bin/bash
#SBATCH -J compress
#SBATCH -t 6:00:00

#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=192
#SBATCH --partition=genoa  
#SBATCH --ear=off

#SBATCH --mem=336G
#SBATCH --exclusive

#SBATCH --output=slurm_logs/compress/compress_%j.out


# --- Environment Setup ---
BASE_DIR="$HOME/thesis"
PROJECT_DIR="/projects/prjs1109"
SCRATCH_DIR="/scratch-shared/jdclercq"
VENV_PATH="$BASE_DIR/venv"
SCRIPT_PATH="$BASE_DIR/compress_jsonl.py"
# ---------------------


# --- Configuration ---

# export INPUT_FILE="$PROJECT_DIR/intermediate/subsetted_data.jsonl"
export INPUT_FILE="$PROJECT_DIR/data/2048_DA_data.jsonl"
export COMPRESSION_LEVEL="-9"
# ---------------------


# Load modules
module purge
module load 2023
module load Python/3.11.3-GCCcore-12.3.0
source "${VENV_PATH}/bin/activate"


# Run the compression script
echo -e "\n\nJob started at: $(date)\n"

echo "Compressing file: $INPUT_FILE"
echo "Compression level: $COMPRESSION_LEVEL"

python "${SCRIPT_PATH}"


echo "Job finished: $(date)"