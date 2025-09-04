#!/bin/sh

#SBATCH -J c_venv
#SBATCH -t 1:30:00

#SBATCH --partition=gpu_h100
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1

module load 2023
module load Python/3.11.3-GCCcore-12.3.0 CUDA/12.4.0

python -m venv venv-pixiu
source venv-pixiu/bin/activate


pip install --upgrade pip  # Always a good idea
cd PIXIU
pip install -r requirements.txt
cd src/financial-evaluation
pip install -e .[multilingual]

echo -e "\n\nSuccessfully created the PIXIU virtual environment on a gpu_h100 node!\n\n"

echo -e "\n\nVerifing packages...\n"

which python
which pip
pip list

echo -e "\n\nVerification finished\n"