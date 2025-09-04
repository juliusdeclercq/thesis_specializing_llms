#!/bin/sh
#SBATCH -J test
#SBATCH -t 1:00:00

#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1

#SBATCH --partition=gpu_h100
#SBATCH --gres=gpu:1         # Max 4 GPUs
#SBATCH --cpus-per-task=16   # 16 per GPU 

#SBATCH --ear=off

#SBATCH --output=slurm_logs/asdf/test_%j.out
  
  # Logging commands. Remove or add SBATCH to (de)activate. 
# --mail-type=END
#SBATCH --mail-user=j.l.h.de.clercq@businessdatascience.nl


echo -e "\nTraining script initialized with ${SLURM_GPUS_ON_NODE} GPUs.\n" 


# ──────────────────────────────
# DIRECTORIES
# ──────────────────────────────

BASE=$HOME/thesis
SCRATCH_DIR="/scratch-shared/$USER"
MODEL_OUTPUT_DIR=$SCRATCH_DIR/models/instruct_model_$SLURM_JOBID
export CACHE_DIR="${SCRATCH_DIR}/.cache/huggingface"  
mkdir -p $MODEL_OUTPUT_DIR


# ──────────────────────────────
# PYTHON ENVIRONMENT
# ──────────────────────────────

module purge
module load 2023
module load Python/3.11.3-GCCcore-12.3.0
module load CUDA/12.4.0
source "${BASE}/venv/bin/activate"


# ──────────────────────────────
# TESTING PACKAGE VERSIONS
# ──────────────────────────────

nvidia-smi

echo "Checking CUDA availability:"
python -c "import torch; print('CUDA available:', torch.cuda.is_available()); print('CUDA devices:', torch.cuda.device_count())"





