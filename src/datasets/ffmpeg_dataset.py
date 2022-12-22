import os

import addict
import numpy as np
import torch

from src.datasets.base_dataset import BaseDataset
from src.utils.ffmpeg import get_video_metadata, ffmpeg_start_in_process
from src.utils.decode import decode_to_numpy


class FfmpegVideoDataset(BaseDataset):

    def __init__(self, video_path: str, cfg: addict.Dict, n_skip_frames: int = 0, return_raw_imgs: bool = False):
        super().__init__(cfg, image_load_format='rgb', return_raw_imgs=return_raw_imgs)
        self.video_path = video_path
        self.base_path = os.path.split(video_path)[0]
        self.orig_width, self.orig_height, self.len, self.fps, self.codec_name = get_video_metadata(video_path)
        self.width, self.height = self.input_shape
        assert n_skip_frames == 0, "FfmpegVideoDataset doesn\'t support n_skip_frames"
        self.index = 0
        self.in_popen = ffmpeg_start_in_process(cfg.ffmpeg, video_path, self.input_shape, self.codec_name)

    def __len__(self) -> int:
        return self.len

    def get_next_frame(self, idx: int):
        # read buffer
        assert self.index == idx, "Tried to access frames out of order!"
        in_bytes = self.in_popen.stdout.read(self.width * self.height * 3 // 2)
        self.index += 1
        if not in_bytes:
            raise StopIteration
        # decode buffer
        # img = decode_to_torch(in_bytes, self.height, self.width, device)
        return decode_to_numpy(in_bytes, self.height, self.width).astype(np.float32)

    def __getitem__(self, idx) -> dict:
        # add img_path for consistency
        img_path = os.path.join(self.base_path, f'{self.index+1:0>5}.jpg')

        # get nearest image_mask
        img_mask = self.load_nearest_mask(img_path)

        # get frame and transform (no need to resize, as it's already done in ffmpeg)
        img = self.get_next_frame(idx)
        img_orig = img.copy() if self.return_raw_imgs else None
        img = self.preprocess_numpy(img, img_mask, resize=False)
        if img_mask is not None:
            img_mask = torch.from_numpy(img_mask == 255)
        # img = torch.as_tensor(img)
        # img = self.preprocess_torch(img, img_mask)
        return img, img_mask, {
            "height": self.orig_height, "width": self.orig_width, "image_path": img_path, 'img_orig': img_orig,
        }
