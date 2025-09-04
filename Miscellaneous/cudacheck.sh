#!/bin/sh
#SBATCH -J jb
#SBATCH -t 12:00:00

#SBATCH --nodes=1
#SBATCH --ntasks=1    
#SBATCH --cpus-per-task=1
#SBATCH --partition=genoa
#SBATCH --ear=off


source venv/bin/activate


python $HOME/thesis/cuda_version_checker.py