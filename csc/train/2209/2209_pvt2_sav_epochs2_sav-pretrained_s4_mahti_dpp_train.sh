#!/bin/bash

#SBATCH --job-name=avsegformer-sav-train
#SBATCH --account=project_2005102
#SBATCH --output=./sbatch_logs/%J.log
#SBATCH --error=./sbatch_logs/%J.log
#SBATCH --verbose
#SBATCH --nodes=1
#SBATCH --partition=gpumedium
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:a100:4,nvme:100
#SBATCH --cpus-per-task=32
#SBATCH --mem-per-gpu=122500M
#SBATCH --time=36:00:00

export PATH="/scratch/project_2005102/sophie/segformer_conda/bin:$PATH"

module load tykky

set -e

# see the train.sh for details and configs
cd /scratch/project_2005102/sophie/repos/AVSegFormer

SESSION="s4"
TRAIN_FILE="dpp_train.py"
TRAIN_FILE_PATH="scripts/$SESSION/$TRAIN_FILE"
CONFIG="/scratch/project_2005102/sophie/repos/AVSegFormer/config/sav/pvt2/1709/1709_epochs2_sav-pretrained_s4.py"

#PYTHONPATH="$(dirname $0)/..":$PYTHONPATH \
export PYTHONPATH="${PYTHONPATH}:/scratch/project_2005102/sophie/repos/AVSegFormer"

export RDZV_HOST=$(hostname)
export RDZV_PORT=29400


srun torchrun \
    --nnodes=$SLURM_JOB_NUM_NODES \
    --nproc_per_node=4 \
    --rdzv_id=$SLURM_JOB_ID \
    --rdzv_backend=c10d \
    --rdzv_endpoint="$RDZV_HOST:$RDZV_PORT" \
    scripts/$SESSION/$TRAIN_FILE $CONFIG

