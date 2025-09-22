#!/bin/bash

#SBATCH --job-name=avsegformer-sav-train
#SBATCH --account=project_2005102
#SBATCH --output=./sbatch_logs/%J.log
#SBATCH --error=./sbatch_logs/%J.log
#SBATCH --verbose
#SBATCH --nodes=1
#SBATCH --partition=gpumedium
#SBATCH --ntasks-per-node=2
#SBATCH --gres=gpu:a100:8,nvme:100
#SBATCH --cpus-per-task=32
#SBATCH --mem-per-gpu=122500M
#SBATCH --time=36:00:00

export PATH="/scratch/project_2005102/sophie/segformer_conda/bin:$PATH"

module load tykky

# to avoid OoO errors
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:24

# debugging
export NCCL_DEBUG=INFO
export NCCL_DEBUG_SUBSYS=ALL
export PYTHONFAULTHANDLER=1


set -e

# see the train.sh for details and configs
cd /scratch/project_2005102/sophie/repos/AVSegFormer
srun bash train.sh \
    "s4" \
    /scratch/project_2005102/sophie/repos/AVSegFormer/config/sav/pvt2/0909_epochs1_sav-pretrained_s4.py \
    "train.py"
