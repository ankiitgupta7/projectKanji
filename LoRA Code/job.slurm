#!/bin/bash --login
#SBATCH --job-name=kanji_finetune
#SBATCH --output=finetune_%j.out
#SBATCH --error=finetune_%j.err
#SBATCH --partition=gpu
#SBATCH --gres=gpu:v100s:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=12:00:00

# Load conda environment (adjust the path if necessary)
source ~/miniforge3/etc/profile.d/conda.sh
conda activate kanji-gpu
export PATH=~/miniforge3/envs/kanji-gpu/bin:$PATH

echo "Running final_tune.py on MSU HPCC with a Tesla V100s GPU..."

# Run the training script
python final_tune.py
