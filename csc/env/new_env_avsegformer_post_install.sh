# 1. module load tykky
# 2. conda-containerize update /scratch/project_2005102/sophie/segformer_conda --post-install /scratch/project_2005102/sophie/repos/AVSegFormer/csc/env/env_avsegformer_post_install.sh

pip install torch==1.11.0+cu113 torchvision==0.12.0+cu113 torchaudio==0.11.0 --extra-index-url https://download.pytorch.org/whl/cu113

pip install mmcv-full -f https://download.openmmlab.com/mmcv/dist/cu111/torch1.10.0/index.html
pip install pandas
pip install timm
pip install resampy
pip install soundfile

pip install -U setuptools
pip install ninja
pip install pycocotools

pip install "opencv-python>=4.10,<4.11"

# tensorboard
pip install tensorboard


# for first time running this post-install script, comment these out
cd /scratch/project_2005102/sophie/repos/AVSegFormer/ops
sh make.sh