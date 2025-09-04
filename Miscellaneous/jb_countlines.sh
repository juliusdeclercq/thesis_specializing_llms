#!/bin/sh
#SBATCH -J sample
#SBATCH -t 12:00:00

#SBATCH --nodes=1
#SBATCH --ntasks=1    
#SBATCH --cpus-per-task=8
#SBATCH --partition=genoa
#SBATCH --ear=off

#SBATCH --output=slurm_logs/count_lines/count_lines_%j.out


echo -e "\n\nStarted job at $(date +"%H:%M:%S")\n"

# Load Python
module load 2023
module load Python/3.11.3-GCCcore-12.3.0

# Set scratch and input path
SCRATCH_DIR="/scratch-shared/$USER"
INPUT_PATH="$SCRATCH_DIR/data/tokenized/tokenized_domain_adapt_data.jsonl"

# Go to scratch
cd "$SCRATCH_DIR"

# Run the line count script
python "$HOME/thesis/count_lines.py" --file "$INPUT_PATH"

T=$(date +"%H:%M:%S")
echo -e "\nFinished counting at $T\n"