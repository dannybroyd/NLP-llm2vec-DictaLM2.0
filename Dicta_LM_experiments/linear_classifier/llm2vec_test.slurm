#!/bin/bash
#SBATCH --job-name=llm2vec_test           # Job name
#SBATCH --output=logs/llm2vec_802.out          # Standard output log
#SBATCH --error=logs/llm2vec_802.err           # Error log
#SBATCH --partition=killable              # Specify partition
#SBATCH --account=gpu-research           
#SBATCH --constraint=l40s              
#SBATCH --time=1440                       # Max runtime (HH:MM:SS)
#SBATCH --nodes=1                         # Number of nodes
#SBATCH --ntasks=1                        # Number of tasks
#SBATCH --cpus-per-task=8                 # CPU cores per task
#SBATCH --mem=16G                         # Memory per node
#SBATCH --gpus=1                          # Number of GPUs (if needed)

# Activate Conda environment
source /home/joberant/NLP_2425a/doronaloni/anaconda3/etc/profile.d/conda.sh
conda activate nlp_env

# Set HF_HOME and HUGGINGFACE_HUB_CACHE to a writable directory
export HF_HOME=/home/joberant/NLP_2425a/doronaloni/NLP-llm2vec-DictaLM2.0/Dicta_LM_experiments/linear_classifier/huggingface_cache
export HUGGINGFACE_HUB_CACHE=/home/joberant/NLP_2425a/doronaloni/NLP-llm2vec-DictaLM2.0/Dicta_LM_experiments/linear_classifier/huggingface_cache
mkdir -p $HF_HOME

# Navigate to your project directory
cd /home/joberant/NLP_2425a/doronaloni/NLP-llm2vec-DictaLM2.0/Dicta_LM_experiments/linear_classifier/

# Run your Python script
python llm2vec_linear_classifier.py
