#!/bin/bash

#SBATCH --job-name=avsegformer-sav-train
#SBATCH --account=project_2005102
#SBATCH --output=./sbatch_logs/%J.log
#SBATCH --error=./sbatch_logs/%J.log
#SBATCH --verbose
#SBATCH --nodes=1
#SBATCH --partition=gpusmall
#SBATCH --ntasks-per-node=2
#SBATCH --gres=gpu:a100:2,nvme:100
#SBATCH --cpus-per-task=32
#SBATCH --mem-per-gpu=122500M
#SBATCH --time=36:00:00

export PATH="/scratch/project_2005102/sophie/segformer_conda/bin:$PATH"

module load tykky
module load gcc/11
module load cuda/11.5

export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:64

set -e

cd /scratch/project_2005102/sophie/repos/AVSegFormer

# Pass the config file as an argument to train.sh
srun bash train.sh "$1"