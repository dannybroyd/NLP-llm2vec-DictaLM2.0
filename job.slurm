#!/bin/bash
#SBATCH --job-name=my_job                # Job name
#SBATCH --output=logs/my_job.out          # Standard output log
#SBATCH --error=logs/my_job.err           # Error log
#SBATCH --partition=studentkillable       # Specify partition
#SBATCH --time=02:00:00                   # Max runtime (HH:MM:SS)
#SBATCH --nodes=1                         # Number of nodes
#SBATCH --ntasks=1                        # Number of tasks
#SBATCH --cpus-per-task=8                 # CPU cores per task
#SBATCH --mem=16G                         # Memory per node
#SBATCH --gpus=1                          # Number of GPUs (if needed)

# Activate Conda environment
source /home/joberant/NLP_2425a/doronaloni/anaconda3/etc/profile.d/conda.sh
conda activate nlp_env

# Set HF_HOME and HUGGINGFACE_HUB_CACHE to a writable directory
export HF_HOME=/home/joberant/NLP_2425a/doronaloni/huggingface_cache
export HUGGINGFACE_HUB_CACHE=/home/joberant/NLP_2425a/doronaloni/huggingface_cache
mkdir -p $HF_HOME

# Set CUDA allocation config to try to mitigate fragmentation issues
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# Navigate to your project directory
cd /home/joberant/NLP_2425a/doronaloni/NLP-llm2vec-DictaLM2.0

# Run your Python script
python experiments/run_mntp.py train_configs/mntp/DictaLM.json
