#!/bin/bash

#SBATCH --job-name=avsegformer-sav-train-debug
#SBATCH --account=project_2000936
#SBATCH --output=./sbatch_logs/%J.log
#SBATCH --error=./sbatch_logs/%J.log
#SBATCH --verbose
#SBATCH --nodes=1
#SBATCH --partition=gputest
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:a100:1,nvme:100
#SBATCH --cpus-per-task=10
#SBATCH --mem-per-gpu=122500M
#SBATCH --time=00:15:00

export PATH="/scratch/project_2005102/sophie/segformer_conda/bin:$PATH"

module load tykky

set -e

# see the train.sh for details and configs
cd /scratch/project_2005102/sophie/repos/AVSegFormer

SESSION="avss"
TRAIN_FILE="dpp_train.py"
TRAIN_FILE_PATH="scripts/$SESSION/$TRAIN_FILE"
CONFIG="/scratch/project_2005102/sophie/repos/AVSegFormer/config/sav/pvt2/avss/AVSegFormer_pvt2_avss.py"

export PYTHONPATH="${PYTHONPATH}:/scratch/project_2005102/sophie/repos/AVSegFormer"

export RDZV_HOST=$(hostname)
export RDZV_PORT=29400


srun torchrun \
    --nnodes=$SLURM_JOB_NUM_NODES \
    --nproc_per_node=1 \
    --rdzv_id=$SLURM_JOB_ID \
    --rdzv_backend=c10d \
    --rdzv_endpoint="$RDZV_HOST:$RDZV_PORT" \
    scripts/$SESSION/$TRAIN_FILE $CONFIG

