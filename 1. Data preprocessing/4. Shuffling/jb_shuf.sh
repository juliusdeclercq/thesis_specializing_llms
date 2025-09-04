#!/bin/sh
#SBATCH -J shuffle
#SBATCH -t 36:00:00

#SBATCH --nodes=1
#SBATCH --ntasks=1    
#SBATCH --cpus-per-task=192
#SBATCH --partition=genoa
#SBATCH --ear=off

#SBATCH --mem=336G
#SBATCH --exclusive

#SBATCH --output=slurm_logs/shuffle/shuffle_%j.out

#SBATCH --mail-type=END
#SBATCH --mail-user=j.l.h.de.clercq@businessdatascience.nl




echo -e "\n\nStarted job at $(date)\n"

# Define directories
SCRATCH_DIR="/scratch-shared/$USER" 
PROJECT_DIR="/projects/prjs1109"  


# --- Configuration ---
EXAMPLE_LIMIT=  
INPUT_FILE="$SCRATCH_DIR/data/tokenized/2048_DA_data.jsonl"

TRAIN_SIZE=1e7   # 10M should be more than enough for both the 2048 and 8092 msl data
EVAL_SIZE=2e4    # 20k is also plenty. With MSL=8192 this took 1.5 hours to evaluate

# ---------------------


# Load Python modules
module load 2023
module load Python/3.11.3-GCCcore-12.3.0
module load SciPy-bundle/2023.07-gfbf-2023a


echo "Executing Python script..."
python "$HOME/thesis/shuffle_data4.py" \
       "$SCRATCH_DIR" \
       "$PROJECT_DIR" \
       --input_file $INPUT_FILE \
       --train_size $TRAIN_SIZE \
       --eval_size $EVAL_SIZE

echo -e "\nFinished shuffling at $(date)\n"