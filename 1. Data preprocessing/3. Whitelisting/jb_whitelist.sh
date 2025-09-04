#!/bin/sh
#SBATCH -J whitelist
#SBATCH -t 8:00:00

#SBATCH --nodes=1
#SBATCH --ntasks=1 
#SBATCH --cpus-per-task=192
#SBATCH --partition=genoa
#SBATCH --ear=off

#SBATCH --mem=336G
#SBATCH --exclusive

#SBATCH --mail-type=END
#SBATCH --mail-user=j.l.h.de.clercq@businessdatascience.nl

#SBATCH --output=slurm_logs/subset/subset_%j.out


echo -e "\n\nStarted job at $(date)\n"


# ──────────────────────────────
# CONFIGURATION
# ──────────────────────────────
year=$1     # Specify the year as the first (and only) argument passed with the sbatch command for execution.

export SCRATCH_DIR=/scratch-shared/$USER
export PROJECT_DIR=/projects/prjs1109 
export NUMEXPR_MAX_THREADS=$SLURM_CPUS_PER_TASK


# ──────────────────────────────
# Module environment
# ──────────────────────────────
module load 2023
module load Python/3.11.3-GCCcore-12.3.0
module load SciPy-bundle/2023.07-gfbf-2023a
source "${HOME}/thesis/venv/bin/activate"


# ──────────────────────────────
# Data preparation
# ──────────────────────────────
mkdir -p "$SCRATCH_DIR/output" "$PROJECT_DIR"
cd "$SCRATCH_DIR"

cp "$HOME/thesis/whitelist.pkl" "$SCRATCH_DIR"

INPUT_YEAR_DIR="$SCRATCH_DIR/intermediate/$year"
if [ ! -d "$INPUT_YEAR_DIR" ]; then
  mkdir -p "$INPUT_YEAR_DIR"
  echo -e "\nCopying $year data from project to scratch space...\n"
  cp -r "/projects/prjs1109/intermediate/$year/filings" "$INPUT_YEAR_DIR"
else
  echo "Input directory $INPUT_YEAR_DIR already exists. Skipping copy."
fi

echo -e "\n\nData migrated to shared-scratch at $(date)\n"

# ──────────────────────────────
# Execution
# ──────────────────────────────
echo -e "\n\nSubsetting year: $year\n\n"
python "$HOME/thesis/subsetting4.py" "$year"
exit_code=$?

echo -e "\n\nFinished subsetting at $(date)\n"

# ──────────────────────────────
# Cleanup
# ──────────────────────────────
if [ $exit_code -eq 0 ]; then
  echo -e "\nCleaning $year scratch data\n"
  rm -r "$INPUT_YEAR_DIR"
else
  echo -e "\nSkipping cleanup due to error (exit code $exit_code)\n"
fi

echo -e "\n\nFinished job at $(date)\n"