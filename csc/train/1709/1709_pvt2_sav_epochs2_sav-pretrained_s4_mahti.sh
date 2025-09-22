#!/bin/bash

#SBATCH --job-name=avsegformer-sav-train
#SBATCH --account=project_2005102
#SBATCH --output=./sbatch_logs/%J.log
#SBATCH --error=./sbatch_logs/%J.log
#SBATCH --verbose
#SBATCH --nodes=1
#SBATCH --partition=gpumedium
#SBATCH --gres=gpu:a100:4
#SBATCH --cpus-per-task=32
#SBATCH --mem-per-gpu=122500M
#SBATCH --time=36:00:00

export PATH="/scratch/project_2005102/sophie/segformer_conda/bin:$PATH"

module load tykky

set -e

# see the train.sh for details and configs
cd /scratch/project_2005102/sophie/repos/AVSegFormer
srun bash train.sh \
    "s4" \
    /scratch/project_2005102/sophie/repos/AVSegFormer/config/sav/pvt2/1709/1709_epochs2_sav-pretrained_s4.py \
    "train.py"
