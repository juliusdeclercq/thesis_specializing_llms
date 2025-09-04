#!/bin/sh
#SBATCH -J shuffle
#SBATCH -t 12:00:00

#SBATCH --nodes=1
#SBATCH --ntasks=1    
#SBATCH --cpus-per-task=8
#SBATCH --partition=genoa
#SBATCH --ear=off




LINES="1000"



echo -e "\nStarted job at $(date +%H:%M:%S)"

module load 2023
module load Python/3.11.3-GCCcore-12.3.0
module load SciPy-bundle/2023.07-gfbf-2023a
source $HOME/thesis/venv/bin/activate


export HF_TOKEN=$(jq -r '.HF_token' "${HOME}/thesis/API_keys.json")

echo -e "\nStarted counting at $(date +%H:%M:%S)"

SCRATCH_DIR="/scratch-shared/$USER" 
cd "$SCRATCH_DIR"

python $HOME/thesis/step_counter.py \
    --data_path "/scratch-shared/jdclercq/data/test_${LINES}.jsonl" \
    --model_name_or_path "meta-llama/Meta-Llama-3.1-8B" \
    --max_seq_length 8192 \
    --num_gpus 1 \
    --per_device_batch_size 8 \
    --gradient_accumulation_steps 8

echo -e "\nFinished counting at $(date +%H:%M:%S)"