#!/bin/bash

# EXAMPLE USAGE:
# sbatch job.sh configs/default.yaml

#SBATCH --job-name=lab2-autoencoder
#SBATCH --account=mth250011p
#SBATCH --partition=GPU-shared
#SBATCH --gpus=1
#SBATCH --cpus-per-task=5
#SBATCH --time=10:00:00
#SBATCH --output=logs/job-%j.out
#SBATCH --error=logs/job-%j.err

module load anaconda3
conda activate env_214_test

python run_autoencoder.py "$1"
