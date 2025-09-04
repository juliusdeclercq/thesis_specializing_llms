#!/bin/sh
#SBATCH -J ruler
#SBATCH -t 36:00:00

#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1

#SBATCH --partition=gpu_h100
#SBATCH --gres=gpu:1         # Max 4 GPUs
#SBATCH --cpus-per-task=16   # 16 per GPU 

#SBATCH --ear=off

#SBATCH --output=slurm_logs/ruler/ruler_%j.out
  
  # Logging commands. Remove or add SBATCH to (de)activate. 
# --mail-type=END
# --mail-user=j.l.h.de.clercq@businessdatascience.nl

# Setting directories
BASE=$HOME/thesis
SCRATCH_DIR="/scratch-shared/$USER"

# No need to purge or load modules or set up a Python environment as everything here is run from the RULER container.

# ====================================================== #
# ================    CONFIGURATION    ================= #

MODEL_NAME="DA1"         # DA1, llama3-8b, llama3-70b

EVAL_OUTPUT_DIR="$BASE/eval-output"
export BATCH_SIZE=512

mkdir -p $EVAL_OUTPUT_DIR
echo -e "\n\nModel name = $MODEL_NAME"
echo -e "Batch Size = $BATCH_SIZE\n\n"


# ====================================================== #
# ============    ENVIRONMENT VARIABLES    ============= #

# Variables for Huggingface and WandB
CONFIG_FILE="${BASE}/acc_config_${SLURM_GPUS_ON_NODE}.yaml" # 1, 2 or 4, denoting number of GPUs
KEYS_FILE="${BASE}/API_keys.json"

export HF_TOKEN=$(jq -r '.HF_token' "${KEYS_FILE}")
export WANDB_API_KEY=$(jq -r '.WandB_API_key' "${KEYS_FILE}")
export HF_HOME="$SCRATCH_DIR/huggingface"  
export NLTK_DATA=/scratch-shared/jdclercq/nltk_data  # or $HOME/nltk_data if you have enough space

mkdir -p $NLTK_DATA

IMAGE="$HOME/thesis/ruler.sif"
RULER_REPO="$BASE/RULER"

# ====================================================== #
# ==================    EXECUTION    =================== #


echo -e "\n\nGetting data...\n\n"
singularity exec --nv --bind $RULER_REPO:/RULER $IMAGE \
bash -c '
  cd /RULER/scripts/data/synthetic/json
  [ -s PaulGrahamEssays.json ] || python download_paulgraham_essay.py
  [ -s squad.json ] && [ -s hotpotqa.json ] || bash download_qa_dataset.sh
'

# Downloading NLTK data from the RULER container. 
# This makes sure it uses the same Python environment and I won't get any conflicts.
singularity exec --nv \
  --env NLTK_DATA=$NLTK_DATA \
  --bind $NLTK_DATA:$NLTK_DATA \
  $IMAGE \
   python <<EOF 
import nltk
import os
punkt_datasets = ['punkt', 'punkt_tab']
for punkt_dataset in punkt_datasets:
  nltk.download(punkt_dataset)
  pth = os.path.join(os.environ['NLTK_DATA'], 'tokenizers', punkt_dataset)
  if not os.path.exists(pth):
    raise RuntimeError(f"NLTK punkt data {punkt_dataset} not found where expected: {pth}")
  print(f'\n{punkt_dataset}:\n', os.listdir(pth), "\n")
EOF


echo -e "\n\nLaunching RULER evaluation script.\n\n"
singularity exec --nv \
  --env NLTK_DATA=$NLTK_DATA \
  --bind $RULER_REPO:/RULER \
  --bind $NLTK_DATA:$NLTK_DATA \
  --bind $SCRATCH_DIR:$SCRATCH_DIR \
  --cwd /RULER/scripts \
  $IMAGE \
  bash run.sh $MODEL_NAME synthetic
