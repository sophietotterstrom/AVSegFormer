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

METADATA_FILE_NAME = "florence_base_captions.csv" # "metadata.csv"

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


def rle_decode(rle):
    """Decode RLE encoded mask using pycocotools."""
    if isinstance(rle, dict) and 'counts' in rle:
        # If it's already in the correct format, use it directly
        return mask_util.decode(rle)
    else:
        # Handle string format RLE if needed
        return None  # Implement as needed



class SAVDataset(Dataset):
    """Dataset for multiple sound source segmentation"""

    def __init__(self, split='train', cfg=None):
        """
        Config (path: None): {
            'anno_csv': 'data/Single-source/s4_meta_data.csv', 
            'dir_img': 'data/Single-source/s4_data/visual_frames', 
            'dir_audio_log_mel': 'data/Single-source/s4_data/audio_log_mel', 
            'dir_mask': 'data/Single-source/s4_data/gt_masks', 
            'img_size': (224, 224), 
            'batch_size': 1
        }
        """
        
        super(SAVDataset, self).__init__()
        self.split = split
        self.mask_num = 5
        self.cfg = cfg
        df_all = pd.read_csv(cfg.anno_csv, sep=',')
        
        self.df_split = df_all[df_all['split'] == split]
        print(f"{len(self.df_split)}/{df_all} videos are used for {self.split}")
        
        
        ## custom recursive handling
        # Create a list to store all video data
        self.all_videos = []
        
        # Get all subdirectories under base_dir
        base_path = Path(cfg.base_dir)
        subdirs = [d for d in base_path.iterdir() if d.is_dir()]
        print(subdirs)
        
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
        
        img_base_path = os.path.join(self.cfg.dir_img, video_name)
        
        mask_base_path = os.path.join(
            self.cfg.dir_mask, self.split, video_name
        )
        
        audio_lm_path = os.path.join(
            self.cfg.dir_audio_log_mel, self.split, video_name + '.pkl'
        )
        audio_log_mel = load_audio_lm(audio_lm_path)
        # audio_lm_tensor = torch.from_numpy(audio_log_mel)
        imgs, masks = [], []
        for img_id in range(1, 6):
            img = load_image_in_PIL_to_Tensor(
                os.path.join(
                    img_base_path, "%s.mp4_%d.png" % (video_name, img_id)
                ), transform=self.img_transform
            )
            imgs.append(img)
        for mask_id in range(1, self.mask_num + 1):
            mask = load_image_in_PIL_to_Tensor(
                os.path.join(
                    mask_base_path, "%s_%d.png" % (video_name, mask_id)
                ), transform=self.mask_transform, 
                mode='P'
            )
            masks.append(mask)
            
        imgs_tensor = torch.stack(imgs, dim=0)
        masks_tensor = torch.stack(masks, dim=0)

        return imgs_tensor, audio_log_mel, masks_tensor, video_name

    def __len__(self):
        return len(self.df_split)
