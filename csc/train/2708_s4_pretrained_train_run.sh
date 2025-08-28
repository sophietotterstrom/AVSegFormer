#!/bin/bash

#SBATCH --job-name=avsegformer-sav-train
#SBATCH --account=project_2005102
#SBATCH --output=./sbatch_logs/%J.log
#SBATCH --error=./sbatch_logs/%J.log
#SBATCH --verbose
#SBATCH --nodes=1
#SBATCH --partition=gpu
#SBATCH --ntasks-per-node=2
#SBATCH --gres=gpu:v100:2,nvme:100
#SBATCH --cpus-per-task=10
#SBATCH --mem-per-gpu=122500M
#SBATCH --time=36:00:00

export PATH="/scratch/project_2005102/sophie/segformer_conda/bin:$PATH"

module load tykky
module load gcc/11
module load cuda/11

set -e

# see the train.sh for details and configs
cd /scratch/project_2005102/sophie/repos/AVSegFormer
srun bash train.sh \
    "ms3" \
    /scratch/project_2005102/sophie/repos/AVSegFormer/config/sav/s4/AVSegFormer_pretrained_S4_ms3_sav_avsbench_test.py \
    "sav_train.py"
