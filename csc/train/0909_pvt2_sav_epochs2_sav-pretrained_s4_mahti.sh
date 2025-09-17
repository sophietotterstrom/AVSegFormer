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
#SBATCH --mem=245G
#SBATCH --time=36:00:00

export PATH="/scratch/project_2005102/sophie/segformer_conda/bin:$PATH"

module load tykky
module load gcc/11
module load cuda/11.5


set -e

# see the train.sh for details and configs
cd /scratch/project_2005102/sophie/repos/AVSegFormer
srun bash train.sh \
    "s4" \
    /scratch/project_2005102/sophie/repos/AVSegFormer/config/sav/pvt2/0909_epochs2_sav-pretrained_s4.py \
    "train.py"
