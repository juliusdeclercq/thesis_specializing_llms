#!/bin/bash
#SBATCH -J delete
#SBATCH -t 12:00:00

#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=192
#SBATCH --partition=genoa  
#SBATCH --ear=off

#SBATCH --mem=336G
#SBATCH --exclusive

#SBATCH --output=slurm_logs/delete/delete_%j.out


# --- Configuration ---
export DIRECTORY_TO_CLEAR="/scratch-shared/jdclercq/apptainer_tmp"

# --- End Setup ---

module purge
module load 2023
module load Python/3.11.3-GCCcore-12.3.0


echo "Job started on $(hostname) at $(date)"
echo "Using ${SLURM_CPUS_PER_TASK} cores"
echo -e "Clearing out ${DIRECTORY_TO_CLEAR}\n"

python clear_dir2.py     # Running script

# Print completion time
echo -e "\n\nJob completed at $(date)\n\n"


echo "Job finished: $(date)"