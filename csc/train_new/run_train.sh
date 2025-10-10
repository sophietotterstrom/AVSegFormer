#!/bin/bash

# Usage: run_train.sh config.yml

CONFIG_FILE=$1

if [ -z "$CONFIG_FILE" ]; then
    echo "Usage: ./train_runner.sh <config.yml>"
    exit 1
fi

# Parse YAML to get job parameters
NUM_GPUS=$(grep "^num_gpus:" $CONFIG_FILE | sed 's/num_gpus: *\([0-9]*\)/\1/')
DURATION=$(grep "^duration:" $CONFIG_FILE | sed 's/duration: *"\?\([^"]*\)"\?/\1/')
JOB_NAME=$(grep "^job_name:" $CONFIG_FILE | sed 's/job_name: *"\?\([^"]*\)"\?/\1/')

# Defaults
if [ -z "$NUM_GPUS" ]; then NUM_GPUS=4; fi
if [ -z "$DURATION" ]; then DURATION="12:00:00"; fi
if [ -z "$JOB_NAME" ]; then JOB_NAME="avsegformer-train"; fi

# Submit job with dynamic parameters
sbatch --job-name="$JOB_NAME" \
       --gres=gpu:a100:$NUM_GPUS,nvme:100 \
       --time="$DURATION" \
       train_job.sh "$CONFIG_FILE"