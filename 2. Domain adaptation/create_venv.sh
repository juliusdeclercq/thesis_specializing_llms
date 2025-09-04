#!/bin/sh

#SBATCH -J c_venv
#SBATCH -t 2:30:00

#SBATCH --partition=gpu_h100
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1

#SBATCH --output=slurm_logs/venv_creation/venv_%j.out

VENV_NO=${1}



module load 2023
module load Python/3.11.3-GCCcore-12.3.0 CUDA/12.4.0
python -m venv venv$VENV_NO
source venv$VENV_NO/bin/activate


pip install -U pip
pip install -U setuptools
pip install ninja
pip install psutil 
pip install packaging
pip install torch==2.6.0 --index-url https://download.pytorch.org/whl/cu124
pip install accelerate 
pip install wandb
pip install hf_xet
pip install tqdm




# Fixing version compatibility issues

# Installing Unsloth
#pip install \
#    --no-build-isolation \
#    --no-cache-dir \
#    "unsloth[cu124-ampere-torch260] @ git+https://github.com/unslothai/unsloth.git"
pip install --force-reinstall --no-cache-dir unsloth unsloth_zoo # --upgrade


# Installing older version of flash attention due to some dependency conflict with the newest version.
echo -e "\n\nRecompiling flash-attn for CUDA 12.4...\n"
pip uninstall flash-attn -y
export TORCH_CUDA_ARCH_LIST="8.0;9.0"     # 8.0 is for A100 GPUs, 9.0 for H100 GPUs
MAX_JOBS=4 pip install flash-attn==2.7.4.post1 --no-build-isolation --no-cache-dir

# Installing older markupsafe version
pip install "markupsafe<2.1"

# Now doing torch beceause of some CUDA issue...
#pip uninstall torch
#pip install torch --index-url https://download.pytorch.org/whl/cu124


echo -e "\n\nSuccessfully created the virtual environment on a gpu_h100 node!\n\n"

# Verify package installations
python << EOF
import torch
import flash_attn
import unsloth
print(f'PyTorch version: {torch.__version__}')
print(f'PyTorch CUDA version: {torch.version.cuda}')
print(f'Flash Attention version: {flash_attn.__version__}')
print('\nAll packages imported successfully!')
print('CUDA available:', torch.cuda.is_available())
print('CUDA devices:', torch.cuda.device_count())
EOF