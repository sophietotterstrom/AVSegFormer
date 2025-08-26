#!/bin/bash

#SBATCH --job-name=sav-datapipeline-debug
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
#SBATCH --time=00:59:00

export PATH="/scratch/project_2005102/sophie/sophies_conda/bin:$PATH"
set -e

# Usage: sbatch puhti_sav-datapipeline_debug.sh <input_path> <output_path>
INPUT_PATH="$1"
OUTPUT_PATH="$2"

if [ -z "$INPUT_PATH" ] || [ -z "$OUTPUT_PATH" ]; then
	echo "Usage: sbatch puhti_sav-datapipeline_run.sh <input_path> <output_path>"
	exit 1
fi

# Run the AVS data pipeline with provided arguments
srun bash avs_pipeline/scripts/run_pipeline.sh \
    -i "$INPUT_PATH" \
    -o "$OUTPUT_PATH" \
    --duration_s 10 \
    --dataset savdataset \
    --model small_16k \
    --num_workers 2 \
    --negative_prompt "background sounds" \
    --skip_combine \
    --verbose