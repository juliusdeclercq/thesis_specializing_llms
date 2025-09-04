#!/bin/sh
#SBATCH -J mrg_sbst
#SBATCH -t 8:00:00

#SBATCH --nodes=1
#SBATCH --ntasks=1 
#SBATCH --partition=genoa
#SBATCH --ear=off

#SBATCH --mail-type=END
#SBATCH --mail-user=j.l.h.de.clercq@businessdatascience.nl

#SBATCH --output=slurm_logs/subset/mergesubset_%j.out


echo -e "\n\nStarted job at $(date)\n"


# ──────────────────────────────
# CONFIGURATION
# ──────────────────────────────
year=$1     # Specify the year as the first (and only) argument passed with the sbatch command for execution.

export SCRATCH_DIR=/scratch-shared/$USER
export PROJECT_DIR=/projects/prjs1109 
export SCRATCH_OUTPUT=$SCRATCH_DIR/output
export OUTPUT_FILE="subsetted_data.jsonl"

# ──────────────────────────────
# Module environment
# ──────────────────────────────
module load 2023
module load Python/3.11.3-GCCcore-12.3.0
module load SciPy-bundle/2023.07-gfbf-2023a
source "${HOME}/thesis/venv/bin/activate"


# ──────────────────────────────
# Execution
# ──────────────────────────────
echo -e "\n\nMerging subsets...\n\n"
python "$HOME/thesis/subset_merge.py"
exit_code=$?

echo -e "\n\nFinished merging at $(date)\n"

# ──────────────────────────────
# Cleanup
# ──────────────────────────────
if [ $exit_code -eq 0 ]; then
  echo -e "\nCleaning $year scratch data\n"
  for file in "$SCRATCH_OUTPUT"/*; do
    [ -f "$file" ] && rm "$file"
  done
else
  echo -e "\nSkipping cleanup due to error (exit code $exit_code)\n"
fi


echo -e "\n\nFinished job at $(date)\n"