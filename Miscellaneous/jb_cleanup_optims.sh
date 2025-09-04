#!/bin/bash
#SBATCH -J clean_opts
#SBATCH -t 12:00:00

#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=24
#SBATCH --partition=genoa  
#SBATCH --ear=off


#SBATCH --output=slurm_logs/cleanup/cleanup_%j.out


# --- Configuration ---
export BASE_DIR="/scratch-shared/jdclercq/models/model_12583757"

# --- End Setup ---

module purge
module load 2023
module load Python/3.11.3-GCCcore-12.3.0


echo -e "\n\n\nCleaning up optimizer.pt files from $BASE_DIR using ${SLURM_CPUS_PER_TASK} cores.\n\n\n"


python cleanup_optimizers.py

# Print completion time
echo -e "\n\nJob completed at $(date)\n\n"


echo "Job finished: $(date)"