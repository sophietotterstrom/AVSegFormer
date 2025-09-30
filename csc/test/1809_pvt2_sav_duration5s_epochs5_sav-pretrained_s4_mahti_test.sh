#!/bin/bash

#SBATCH --job-name=avsegformer-sav-test
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
#SBATCH --time=1:00:00



export PATH="/scratch/project_2005102/sophie/segformer_conda/bin:$PATH"

module load tykky
module load gcc/11
module load cuda/11

cd /scratch/project_2005102/sophie/repos/AVSegFormer
bash test.sh \
    "s4" \
    "/scratch/project_2005102/sophie/repos/AVSegFormer/config/sav/pvt2/2909/2909_epochs5_sav-pretrained_s4.py" \
    "/scratch/project_2005102/sophie/repos/AVSegFormer/work_dir/1809_duration5s_epochs5_sav-pretrained_s4/MS3_best.pth"
