#!/bin/sh
#SBATCH -J sample
#SBATCH -t 12:00:00

#SBATCH --nodes=1
#SBATCH --ntasks=1    
#SBATCH --cpus-per-task=8
#SBATCH --partition=genoa
#SBATCH --ear=off

#SBATCH --output=slurm_logs/sample/sample_%j.out


LINES=500



echo -e "\n\nStarted job at $(date +"%H:%M:%S")\n"


# Loading Python modules.
module load 2023
module load Python/3.11.3-GCCcore-12.3.0
module load SciPy-bundle/2023.07-gfbf-2023a



# Define and create directories. Set working directory to shared-scratch/$USER.
SCRATCH_DIR="/scratch-shared/$USER" 
cd "$SCRATCH_DIR"

#Executing Python script to shuffle the data - CORRECTED CALL
python "$HOME/thesis/make_test_dataset.py" \
  --input_path "$SCRATCH_DIR/data/tokenized/tokenized_domain_adapt_data.jsonl"\
  --scratch_dir "$SCRATCH_DIR" \
  --lines $LINES

T=$(date +"%H:%M:%S")
echo -e "\nFinished sampling at $T\n"