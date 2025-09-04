#!/bin/sh
#SBATCH -J split_eval
#SBATCH -t 02:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --partition=genoa
#SBATCH --output=slurm_logs/split_eval/split_%j.out
#SBATCH --mail-type=END
#SBATCH --mail-user=j.l.h.de.clercq@businessdatascience.nl

echo -e "\nSplitting eval dataset and appending to train.\n"

# Setting directories
BASE=$HOME/thesis
SCRATCH_DIR="/scratch-shared/$USER"
OUTPUT_DIR="${SCRATCH_DIR}/data/updated_splits"
mkdir -p $OUTPUT_DIR

# Input files
EVAL_FILE="${SCRATCH_DIR}/data/eval_domain_adaptation_data.jsonl"
TRAIN_FILE="${SCRATCH_DIR}/data/train_domain_adaptation_data.jsonl"

# Set up Python environment
module purge
module load 2023
module load Python/3.11.3-GCCcore-12.3.0
source "${BASE}/venv/bin/activate"

# Run the script
echo -e "\nLaunching split script.\n"
python $BASE/split_eval_to_train.py \
    --eval_file $EVAL_FILE \
    --train_file $TRAIN_FILE \
    --output_dir $OUTPUT_DIR \
    --eval_keep_size 10000

echo -e "\nScript completed.\n"