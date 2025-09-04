#!/bin/sh
#SBATCH -J ruler
#SBATCH -t 10:00:00

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

# Set up Python environment.
module purge
module load 2023
module load Python/3.11.3-GCCcore-12.3.0 CUDA/12.4.0
source "${BASE}/venv-pixiu/bin/activate"   # This is the FLARE/PIXIU environment.


# ====================================================== #
# ================    CONFIGURATION    ================= #

MODEL_NAME="DA1"         # DA1, llama3-8b, llama3-70b

EVAL_OUTPUT_DIR="$BASE/eval-output"
export BATCH_SIZE=256

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


IMAGE="$HOME/thesis/ruler.sif"
RULER_REPO="$BASE/RULER"

# ====================================================== #
# ==================    EXECUTION    =================== #


echo -e "\n\nChecking package versions...\n\n"
singularity exec --nv --bind $RULER_REPO:/RULER $IMAGE \
python <<EOF
import transformers
print("transformers:", transformers.__version__)
import tokenizers
print("tokenizers:", tokenizers.__version__)
from transformers import AutoTokenizer
tok = AutoTokenizer.from_pretrained("/scratch-shared/jdclercq/models/model_12583757")
sentence = "For I am the sea and nobody owns me! \nA world forgetting, by a world forgot."
token_ids = tok.encode(sentence)
tokens = tok.tokenize(sentence)
print("Sentence:", sentence)
print("Token IDs:", token_ids)
print("Subword tokens:", tokens)
EOF








