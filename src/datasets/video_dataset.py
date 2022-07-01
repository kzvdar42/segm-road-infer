import os
from typing import Dict

import addict
import cv2
import numpy as np
import torch

from src.datasets.base_dataset import BaseDataset


class VideoDataset(BaseDataset):
    
    @classmethod
    def get_video_metadata(cls, video_path):
        cap = cv2.VideoCapture(video_path)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        num_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        cap.release()
        del cap
        return width, height, num_frames, fps

    def __init__(self, video_path: str, cfg: addict.Dict, n_skip_frames: int = 0, return_raw_imgs: bool = False):
        super().__init__(cfg, image_load_format='bgr', return_raw_imgs=return_raw_imgs)
        self.video_path = video_path
        self.base_path = os.path.split(video_path)[0]
        self.orig_width, self.orig_height, self.len, self.fps = self.get_video_metadata(video_path)
        self.width, self.height = self.input_shape
        self.n_skip_frames = n_skip_frames
        self.cap = None

    def __len__(self):
        return int(np.ceil(self.len / max(self.n_skip_frames, 1)))
    
    def get_frame(self, index: int) -> np.ndarray:
        # If tries to get not the next frame, set cap pos to right position
        if self.cap is None:
            self.cap = cv2.VideoCapture(self.video_path)
        cap_pos = int(self.cap.get(cv2.CAP_PROP_POS_FRAMES))
        if cap_pos != index:
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, index)
        return self.cap.read()[1]

    def __getitem__(self, index: int) -> Dict:
        index = index * max(1, self.n_skip_frames)
        img = self.get_frame(index)#.astype(np.float32)
        img_orig = img.copy() if self.return_raw_imgs else None

        # Add img_path for consistency
        img_path = os.path.join(self.base_path, f'{index+1:0>5}.jpg')
        # set nearest image_mask
        img_mask = self.load_nearest_mask(img_path)

        img = self.preprocess_numpy(img, img_mask)
        if img_mask is not None:
            img_mask = torch.from_numpy(img_mask == 255)
        return img, img_mask, {
            "height": self.orig_height, "width": self.orig_width, 'image_path': img_path, 'img_orig': img_orig,
        }
