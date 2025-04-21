#!/bin/bash
#SBATCH --job-name=nlp_simsce               # Job name
#SBATCH --output=logs/my_job.out          # Standard output log
#SBATCH --error=logs/my_job.err           # Error log
#SBATCH --partition=killable              # Specify partition
#SBATCH --account=gpu-research           
#SBATCH --time=1440                       # Max runtime (HH:MM:SS)
#SBATCH --nodes=1                         # Number of nodes
#SBATCH --ntasks=1                        # Number of tasks
#SBATCH --cpus-per-task=8                 # CPU cores per task
#SBATCH --mem=32G                         # Memory per node
#SBATCH --gpus=1                          # Number of GPUs (if needed)

# Activate Conda environment
source /home/joberant/NLP_2425a/doronaloni/anaconda3/etc/profile.d/conda.sh
conda activate nlp_env

# Set HF_HOME and HUGGINGFACE_HUB_CACHE to a writable directory
export HF_HOME=/home/joberant/NLP_2425a/doronaloni/huggingface_cache
export HUGGINGFACE_HUB_CACHE=/home/joberant/NLP_2425a/doronaloni/huggingface_cache
mkdir -p $HF_HOME

# Navigate to your project directory
export PYTHONPATH=$PYTHONPATH:/home/joberant/NLP_2425a/doronaloni/NLP-llm2vec-DictaLM2.0
cd /home/joberant/NLP_2425a/doronaloni/NLP-llm2vec-DictaLM2.0

# Run your Python script
python experiments/run_simcse.py /vol/joberant_nobck/data/NLP_368307701_2425a/doronaloni/NLP-llm2vec-DictaLM2.0/train_configs/simcse/DictaLM.json
