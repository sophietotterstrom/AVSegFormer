#!/bin/bash

#SBATCH --job-name=avsegformer-sav-test
#SBATCH --account=project_2005102
#SBATCH --output=./sbatch_logs/%J.log
#SBATCH --error=./sbatch_logs/%J.log
#SBATCH --verbose
#SBATCH --nodes=1
#SBATCH --partition=gpu
#SBATCH --ntasks-per-node=2
#SBATCH --gres=gpu:v100:2,nvme:100
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-gpu=122500M
#SBATCH --time=01:00:00



export PATH="/scratch/project_2005102/sophie/segformer_conda/bin:$PATH"

module load tykky
module load gcc/11
module load cuda/11

cd /scratch/project_2005102/sophie/repos/AVSegFormer

bash test.sh \
    "ms3" \
    "config/sav/pvt2/AVSegFormer_pvt2_ms3_sav_avsbench_test.py" \
    "work_dir/AVSegFormer_pvt2_ms3_sav/2608_40_per_dir_MS3_best.pth" 
    #--save_pred_mask
