#!/bin/bash
#
# Lab 2 on Bridges-2: pretrain autoencoder or run get_embedding.
# - Pretrain: 161 unlabeled images, all patches; output checkpoints, lightning_logs/, slurm-<jobid>.out
# - Embedding: needs more RAM (--mem=64G); run get_embedding.py to write image15/17/18_ae.csv
#
# Usage on Bridges-2 login node:
#   sbatch job.sh
#   sbatch job.sh configs/default.yaml
#   sbatch job.sh embedding
#   sbatch job.sh embedding configs/default.yaml checkpoints_finetune/finetune-epoch=000.ckpt
#

########################
# Slurm job definition #
########################

# charge to the class account (don't change)
#SBATCH --account=mth250011p

# job name (you can customize)
#SBATCH --job-name=lab2-ae-pretrain

# number of CPUs to use on the GPU-shared partition (8 cores)
#SBATCH --cpus-per-task=8

# wall-clock time limit (2 hours for pretrain; 1 h enough for embedding)
#SBATCH --time=02:00:00

# memory (needed for get_embedding.py loading all 164 images; harmless for pretrain)
#SBATCH --mem=64G

# console output and errors (%j = job ID, so each run gets its own file, no overwrite)
#SBATCH -o slurm-%j.out
#SBATCH -e slurm-%j.err

# 1 H100-80 GPU on GPU-shared partition
#SBATCH --partition=GPU-shared
#SBATCH --gpus=h100-80:1

########################
# Job execution script #
########################

# Load Conda and activate STAT 214 environment
module load anaconda3
conda activate stat214

# Run from lab2-repo/code (adjust path if your repo lives elsewhere)
cd ~/lab2-repo/code

# -u: unbuffered output so tail -f slurm-<jobid>.out shows progress in real time
if [ "${1:-}" = "embedding" ]; then
  CONFIG=${2:-configs/default.yaml}
  CKPT=${3:-checkpoints_finetune/finetune-epoch=005.ckpt}
  echo "Running get_embedding: config=${CONFIG} checkpoint=${CKPT}"
  python -u get_embedding.py "${CONFIG}" "${CKPT}"
else
  CONFIG_PATH=${1:-configs/default.yaml}
  echo "Using config: ${CONFIG_PATH}"
  python -u run_autoencoder.py "${CONFIG_PATH}"
fi