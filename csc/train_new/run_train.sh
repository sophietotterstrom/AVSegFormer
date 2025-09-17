#!/bin/bash

CONFIG_FILE=$1

cd /scratch/project_2005102/sophie/repos/AVSegFormer
srun bash train.sh \
    "s4" \
    /scratch/project_2005102/sophie/repos/AVSegFormer/config/sav/pvt2/0909_epochs1_sav-pretrained_s4.py \
    "train.py"
