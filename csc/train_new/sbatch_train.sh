#!/bin/bash

#SBATCH --account=project_2005102
#SBATCH --output=./sbatch_logs/%J.log
#SBATCH --error=./sbatch_logs/%J.log
#SBATCH --verbose
#SBATCH --nodes=1
#SBATCH --partition=gpumedium
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --mem-per-gpu=122500M

CONFIG_FILE=$1

export PATH="/scratch/project_2005102/sophie/segformer_conda/bin:$PATH"
module load tykky
set -e

cd /scratch/project_2005102/sophie/repos/AVSegFormer

# Parse training parameters
SESSION=$(grep "^session:" $CONFIG_FILE | sed 's/session: *"\?\([^"]*\)"\?/\1/')
TRAIN_FILE=$(grep "^train_file:" $CONFIG_FILE | sed 's/train_file: *"\?\([^"]*\)"\?/\1/')
CONFIG_PATH=$(grep "^config_path:" $CONFIG_FILE | sed 's/config_path: *"\?\([^"]*\)"\?/\1/')
NPROC_PER_NODE=$(grep "^nproc_per_node:" $CONFIG_FILE | sed 's/nproc_per_node: *\([0-9]*\)/\1/')

if [ -z "$NPROC_PER_NODE" ]; then NPROC_PER_NODE=4; fi

export PYTHONPATH="${PYTHONPATH}:/scratch/project_2005102/sophie/repos/AVSegFormer"
export RDZV_HOST=$(hostname)
export RDZV_PORT=29400

srun torchrun \
    --nnodes=$SLURM_JOB_NUM_NODES \
    --nproc_per_node=$NPROC_PER_NODE \
    --rdzv_id=$SLURM_JOB_ID \
    --rdzv_backend=c10d \
    --rdzv_endpoint="$RDZV_HOST:$RDZV_PORT" \
    scripts/$SESSION/$TRAIN_FILE $CONFIG_PATH