import os
from typing import Dict

import addict
import cv2
import numpy as np
import torch
import ffmpeg

from src.datasets.base_dataset import BaseDataset
from src.utils.ffmpeg import (
    ffmpeg_start_in_process
)


device = 'cpu'

rgb_from_yuv_mat = torch.tensor([
    [1.164,  0,      1.596],
    [1.164, -0.392, -0.813],
    [1.164,  2.017,  0    ],
], device=device).T
rgb_from_yuv_off = torch.tensor([[[16, 128, 128]]], device=device)

def yuv2rgb(image):
    image -= rgb_from_yuv_off
    image @= rgb_from_yuv_mat
    return torch.clamp(image, 0, 255)

def decode_to_torch(in_bytes, height, width, device, out_dtype=torch.float32):
    k = width*height
    y = torch.empty(k,    dtype=torch.uint8).set_(torch.ByteStorage.from_buffer(in_bytes[0:k],        byte_order = 'native')).reshape((height, width)).type(out_dtype).to(device)
    u = torch.empty(k//4, dtype=torch.uint8).set_(torch.ByteStorage.from_buffer(in_bytes[k:k+(k//4)], byte_order = 'native')).reshape((height//2, width//2)).type(out_dtype).to(device)
    v = torch.empty(k//4, dtype=torch.uint8).set_(torch.ByteStorage.from_buffer(in_bytes[k+(k//4):],  byte_order = 'native')).reshape((height//2, width//2)).type(out_dtype).to(device)
    u = u.repeat_interleave(2, dim=-1).repeat_interleave(2, dim=-2)
    v = v.repeat_interleave(2, dim=-1).repeat_interleave(2, dim=-2)

    return yuv2rgb(torch.stack((y,u,v), -1))

def decode_to_numpy(in_bytes, height, width):
    return cv2.cvtColor(
        np.frombuffer(in_bytes, dtype=np.uint8).reshape((height + height//2, width)),
        cv2.COLOR_YUV420p2RGB
    )

class FfmpegVideoDataset(BaseDataset):

    @classmethod
    def get_video_metadata(cls, video_path):
        probe = ffmpeg.probe(video_path)
        video_stream = next((stream for stream in probe['streams'] if stream['codec_type'] == 'video'), None)
        width = int(video_stream['width'])
        height = int(video_stream['height'])
        num_frames = int(video_stream['nb_frames'])
        fps = eval(video_stream['avg_frame_rate'])
        codec_name = video_stream['codec_name']
        return width, height, num_frames, fps, codec_name

    def __init__(self, video_path: str, cfg: addict.Dict, n_skip_frames: int = 0, return_raw_imgs: bool = False):
        super().__init__(cfg, image_load_format='rgb', return_raw_imgs=return_raw_imgs)
        self.video_path = video_path
        self.base_path = os.path.split(video_path)[0]
        self.orig_width, self.orig_height, self.len, self.fps, self.codec_name = self.get_video_metadata(video_path)
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

    def __getitem__(self, idx) -> Dict:
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
