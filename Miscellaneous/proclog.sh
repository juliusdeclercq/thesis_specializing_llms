#!/bin/sh
#SBATCH -J proc_log
#SBATCH -t 2:00:00

#SBATCH --nodes=1
#SBATCH --ntasks=1    
#SBATCH --cpus-per-task=6
#SBATCH --partition=genoa
#SBATCH --ear=off

module load 2023
module load Python/3.11.3-GCCcore-12.3.0
module load SciPy-bundle/2023.07-gfbf-2023a


python $HOME/thesis/log_aggregator.py