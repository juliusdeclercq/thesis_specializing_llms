#!/bin/sh
#SBATCH -J LRGEclnr
#SBATCH -t 24:00:00

#SBATCH --nodes=1
#SBATCH --ntasks=1    
#SBATCH --cpus-per-task=6
#SBATCH --mem=30G
#SBATCH --partition=genoa
#SBATCH --ear=off


#SBATCH --mail-type=END
#SBATCH --mail-user=j.l.h.de.clercq@businessdatascience.nl

# This script was written because of OOM errors occurring in 2024. To resolve this, the largest filings of 2024 were processed separately with this script, as these formed the bottlenecks leading to OOM. 
module load 2023
module load Python/3.11.3-GCCcore-12.3.0
module load SciPy-bundle/2023.07-gfbf-2023a
# pip install --user openpyxl bs4 pdfplumber

T=$(date +"%H:%M:%S") 
echo -e "\n\nStarted job at $T\n"

# Set sequence of years to process 
# IMPORTANT: Set --ntasks and --nodes equal to the length of the sequence.
# Note that the sequence contains both boundaries (e.g. (seq 2012 2014) = 2012, 2013, 2014). 
years="2024_RETRY"

# Define and create directories. Set working directory to shared-scratch/$USER.
SCRATCH_DIR=/scratch-shared/$USER
SCRATCH_OUTPUT_DIR=/scratch-shared/$USER/intermediate
PROJECT_DIR=/projects/prjs1109/intermediate 
mkdir -p $SCRATCH_DIR
mkdir -p $PROJECT_DIR
cd $SCRATCH_DIR

# Manage input and output directories.
for year in $years; do
  INPUT_YEAR_DIR="$SCRATCH_DIR/data/$year"
  OUTPUT_YEAR_DIR="$SCRATCH_DIR/output/$year"
  #Copy input files to scratch, if not already there.
  if [ ! -d "$INPUT_YEAR_DIR" ]; then
    cp -r /projects/prjs1109/data/raw/$year $SCRATCH_DIR/data &
  else
    echo "Input directory $YEAR_DIR already on /scratch-shared. Skipping copy."
  fi
  # Remove existing output for the year if already on scratch
  if [ ! -d "$OUTPUT_YEAR_DIR" ]; then
    echo ""
  else
    echo "Removing existing output for year $year"
    rm -r $OUTPUT_YEAR_DIR
  fi
done
wait

T=$(date +"%H:%M:%S")
echo ""
echo -e "\n\nData migrated to shared-scratch at $T\n"


#Executing Python script in loop for each year with positional arguments: year , output_directory
for year in $years; do
  echo -e "\n\nRunning year: $year\n\n"
  mkdir -p $SCRATCH_OUTPUT_DIR/$year   # make temporary output directory on shared scratch space
  srun -n 1 --nodes=1 --exclusive python $HOME/thesis/EDGAR_cleaner5.py  $year $SCRATCH_DIR $SCRATCH_OUTPUT_DIR &
done
wait

T=$(date +"%H:%M:%S")
echo -e "\n\nFinished processing at $T\n"

# Copy output directory from scratch to project space, and delete existing output there if present.
for year in $years; do
  if [ ! -d "$PROJECT_DIR/$year" ]; then
    cp -r $SCRATCH_OUTPUT_DIR/$year $PROJECT_DIR &
  else
    echo "Removing old output from project space for year $year"
    rm -r $PROJECT_DIR/$year
    cp -r $SCRATCH_OUTPUT_DIR/$year $PROJECT_DIR &
  fi
done
wait

# Clean up intermediate directories on /scratch-shared. Leaving input on scratch-shared in case the run fails.
for year in $years; do
  echo "Cleaning up /scratch-shared output for year $year."
  rm -r $SCRATCH_OUTPUT_DIR/$year &
done
wait

T=$(date +"%H:%M:%S")
echo -e "\n\nMigrated data to project space at $T\n"


