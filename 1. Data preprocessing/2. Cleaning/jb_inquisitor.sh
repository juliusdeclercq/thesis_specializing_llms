#!/bin/sh
#SBATCH -J inqstr
#SBATCH -t 6:00:00

#SBATCH --ntasks=1            
#SBATCH --nodes=1       
#SBATCH --cpus-per-task=44
#SBATCH --mem=336G
#SBATCH --ear=off
#SBATCH --exclusive
#SBATCH --partition=genoa

# Directory pattern with year range
# BASE_DIR="/projects/prjs1109/data/raw"

module load 2023
module load Python/3.11.3-GCCcore-12.3.0
module load SciPy-bundle/2023.07-gfbf-2023a

python $HOME/thesis/tests/inquisitor2.py
