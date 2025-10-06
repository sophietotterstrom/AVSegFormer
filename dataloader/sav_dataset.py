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

FPS = 24


def load_audio_lm(audio_lm_path):
    with open(audio_lm_path, 'rb') as fr:
        audio_log_mel = pickle.load(fr)
    audio_log_mel = audio_log_mel.detach().float()  # [DURATION, 1, 96, 64]
    return audio_log_mel


# function from from sav_utils
# originally META's code from SAM-2 repo
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

        # TODO remove when implemented in the pipeline itself
        df_split_temp = df_all[df_all['split'] == split]
        self.df_split = df_split_temp[df_split_temp['caption'] != "no object detected"]
        
        # ImageNet normalization parameters
        self.mean = np.array([0.485, 0.456, 0.406])
        self.std = np.array([0.229, 0.224, 0.225])
    
    def _process_frame(self, frame):
        frame_resized = cv2.resize(
            frame, 
            (self.cfg.img_size[1], self.cfg.img_size[0])
        )

        frame_float = frame_resized.astype(np.float32) / 255.0
        frame_norm = (frame_float - self.mean) / self.std

        # Convert to tensor and change format from HWC to CHW
        img_tensor = torch.from_numpy(frame_norm).permute(2, 0, 1).float()
        return img_tensor

    def _process_mask(self, mask_bin):        
        mask_resized = cv2.resize(
            mask_bin, 
            (self.cfg.img_size[1], self.cfg.img_size[0]), 
            interpolation=cv2.INTER_NEAREST
        )
        # ensure binary mask
        mask_bin = (mask_resized > 0).astype(np.uint8)
        mask_tensor = torch.from_numpy(mask_bin).float().unsqueeze(0)
        return mask_tensor

    def __getitem__(self, index):
        
        df_one_video = self.df_split.iloc[index]
        video_name = df_one_video[0]
        video_subdir = df_one_video[-1]

        print(f"{video_subdir}/{video_name}")
        
        ######### parse paths #########
        video_path = os.path.join(
            self.cfg.base_dir, video_subdir, f"{video_name}.mp4"
        )

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

        # video into imgs and mask to match
        all_frames = decode_video(video_path)
        annotation = json.load(open(mask_path))

        # SAV FPS=24, downsample to FPS=1 as per model architecture
        frames = all_frames[::FPS]
        imgs = []
        masks = []

        for i, frame in enumerate(frames):
            img_tensor = self._process_frame(frame)
            imgs.append(img_tensor)

            # find frame specific mask and decode
            orig_frame_idx = i * FPS
            mask_rle = annotation["masklet"][orig_frame_idx]
            mask_bin = mask_util.decode(mask_rle)
            
            mask_tensor = self._process_mask(mask_bin)
            masks.append(mask_tensor)
        
        print(f"Mask len {len(masks)}, imgs {len(imgs)}")
        
        imgs_tensor = torch.stack(imgs, dim=0)
        masks_tensor = torch.stack(masks, dim=0)

        return imgs_tensor, audio_log_mel, masks_tensor, video_name

    def __len__(self):
        return len(self.df_split)
