pip install torch==1.10.0+cu111 torchvision==0.11.0+cu111 torchaudio==0.10.0 -f https://download.pytorch.org/whl/torch_stable.html
pip install mmcv-full -f https://download.openmmlab.com/mmcv/dist/cu111/torch1.10.0/index.html
pip install pandas
pip install timm
pip install resampy
pip install soundfile

pip install -U setuptools
pip install ninja
pip install pycocotools

# for first time running this post-install script, comment these out
cd /scratch/project_2005102/sophie/repos/AVSegFormer/ops
sh make.sh