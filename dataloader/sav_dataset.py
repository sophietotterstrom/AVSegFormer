import os
import torch
import torch.nn as nn
from torch.utils.data import Dataset

import numpy as np
import pandas as pd
import pickle
from pathlib import Path

import cv2
from PIL import Image
from torchvision import transforms

import json
import pycocotools.mask as mask_util
from typing import List


# old
def load_image_in_PIL_to_Tensor(path, mode='RGB', transform=None):
    img_PIL = Image.open(path).convert(mode)
    if transform:
        img_tensor = transform(img_PIL)
        return img_tensor
    return img_PIL

# old
def load_audio_lm(audio_lm_path):
    with open(audio_lm_path, 'rb') as fr:
        audio_log_mel = pickle.load(fr)
    audio_log_mel = audio_log_mel.detach()  # [DURATION, 1, 96, 64]
    return audio_log_mel

    
# from sav_utils
def decode_video(video_path: str) -> List[np.ndarray]:
    """
    Decode the video and return the RGB frames
    """
    video = cv2.VideoCapture(video_path)
    video_frames = []
    while video.isOpened():
        ret, frame = video.read()
        if ret:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            video_frames.append(frame)
        else:
            break
    return video_frames


class SAVDataset(Dataset):
    """Dataset for multiple sound source segmentation"""

    def __init__(self, split='train', cfg=None):
        super(SAVDataset, self).__init__()

        self.split = split
        self.mask_num = 5
        self.cfg = cfg

        df_all = pd.read_csv(cfg.anno_csv, sep=',')
        self.df_split = df_all[df_all['split'] == split]
        print(f"{len(self.df_split)}/{len(df_all)} videos are used for {self.split}")
        
        self.img_transform = transforms.Compose([
            transforms.Resize(cfg.img_size), # [512, 512]
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
        ])
        
        self.mask_transform = transforms.Compose([
            transforms.Resize(cfg.img_size), # [512, 512]
            transforms.ToTensor(),
        ])

    def __getitem__(self, index):
        
        df_one_video = self.df_split.iloc[index]
        video_name = df_one_video[0]
        video_subdir = df_one_video[-1]
        
        ######### parse paths #########      
        mask_path = os.path.join(
            self.cfg.base_dir, video_subdir, f"{video_name}_manual.json"
        )
        # if manual annotation doesn't exist, check for auto
        if not os.path.exists(mask_path):
            mask_path = os.path.join(
                self.cfg.base_dir, video_subdir, f"{video_name}_auto.json"
            )

        audio_lm_path = os.path.join(
            self.cfg.base_dir, video_subdir, 
            self.cfg.dir_audio_log_mel, video_name + '.pkl'
        )

        ######### load data #########
        audio_log_mel = load_audio_lm(audio_lm_path)

        # decode mask from RLE format to frames
        mask_rle = json.load(open(mask_path))
        mask_bin = mask_util.decode(mask_rle)

        # load video to frames
        frames = decode_video(mp4_path)

        print(mask_bin.shape)
        print(frames.shape)


        
        imgs_tensor = torch.stack(imgs, dim=0)
        masks_tensor = torch.stack(masks, dim=0)

        return imgs_tensor, audio_log_mel, masks_tensor, video_name

    def __len__(self):
        return len(self.df_split)
