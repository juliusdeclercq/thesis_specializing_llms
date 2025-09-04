#!/bin/bash
#SBATCH -J cmprss_tar
#SBATCH -t 9:00:00

#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=192
#SBATCH --partition=genoa  
#SBATCH --ear=off

#SBATCH --mem=336G
#SBATCH --exclusive

#SBATCH --output=slurm_logs/compress/compresstar_%j.out

# ---------------------


# --- Configuration ---

YEAR=${1}     # Pass the year as the first positional argument. 

# ---------------------


# Stop if no year is specified.
if [ -z "$YEAR" ]; then
  echo "Error: No year specified."
  echo "Usage: $0 <YEAR>"
  exit 1
fi

# Run the compression script
echo -e "\n\nJob started at: $(date)\n"
echo "Compressing files for year $YEAR"

find /projects/prjs1109/intermediate/$YEAR -type f -path '*/filings/*.tar' | \
pv -l -s $(find /projects/prjs1109/intermediate/$YEAR -type f -path '*/filings/*.tar' | wc -l) | \
xargs -P 128 -I {} zstd -9 --rm {}


echo "Job finished: $(date)"