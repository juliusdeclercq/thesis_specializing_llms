#!/bin/sh
#SBATCH -J agg_info
#SBATCH -t 12:00:00

#SBATCH --nodes=1
#SBATCH --ntasks=1    
#SBATCH --cpus-per-task=13
#SBATCH --mem=130G
#SBATCH --partition=genoa
#SBATCH --ear=off

#SBATCH --mail-type=END
#SBATCH --mail-user=j.l.h.de.clercq@businessdatascience.nl


module load 2023
module load Python/3.11.3-GCCcore-12.3.0
module load SciPy-bundle/2023.07-gfbf-2023a


python $HOME/thesis/filing_info_aggregator.py