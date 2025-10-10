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
#SBATCH --time=04:00:00

export PATH="/scratch/project_2005102/sophie/segformer_conda/bin:$PATH"

module load tykky
module load gcc/11
module load cuda/11

set -e

pip list

# see the train.sh for details and configs
cd /scratch/project_2005102/sophie/repos/AVSegFormer

PYTHONPATH="$(dirname $0)/..":$PYTHONPATH \
srun python scripts/ms3/sav_train.py \
    /scratch/project_2005102/sophie/repos/AVSegFormer/config/sav/pvt2/1809/1809_duration5s_epochs1.py \
    --duration 5
