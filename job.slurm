#!/bin/bash --login
#SBATCH --job-name=kanji_finetune
#SBATCH --partition=gpu
#SBATCH --gres=gpu:v100s:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=9:29:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err


# Activate your Python environment
source ~/miniforge3/etc/profile.d/conda.sh
conda activate kanji_diff
export PATH=~/miniforge3/envs/kanji_diff/bin:$PATH

# Print environment info (optional for debugging)

# Run the fine-tuning script
PYTHONPATH="" python fine_tune.py
