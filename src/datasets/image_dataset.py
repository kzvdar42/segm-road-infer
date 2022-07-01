from typing import List

import addict
import cv2
import numpy as np
import torch
try:
    from turbojpeg import TurboJPEG
    jpeg = TurboJPEG()
    use_turbojpeg = True
except:
    use_turbojpeg = False

from src.datasets.base_dataset import BaseDataset

class ImageDataset(BaseDataset):
    
    def __init__(self, image_paths: List[str], cfg: addict.Dict, return_raw_imgs: bool = False):
        super().__init__(cfg, image_load_format='bgr', return_raw_imgs=return_raw_imgs)
        self.image_paths = image_paths

    def __len__(self):
        return len(self.image_paths)

    def imread(self, img_path: str) -> np.ndarray:
        if use_turbojpeg:
            # TurboJPEG reads in BGR format
            with open(img_path, 'rb') as in_file:
                return jpeg.decode(in_file.read())
        return cv2.imread(img_path)

    def __getitem__(self, index: int):
        img_path = self.image_paths[index]

        # get nearest image_mask
        img_mask = self.load_nearest_mask(img_path)

        # Read and transform the image
        img = self.imread(img_path)#.astype(np.float32)
        img_orig = img.copy() if self.return_raw_imgs else None
        height, width = img.shape[:2]
        img = self.preprocess_numpy(img, img_mask)
        if img_mask is not None:
            img_mask = torch.from_numpy(img_mask == 255)

        return img, img_mask, {
            "height": height, "width": width, 'image_path': img_path, 'img_orig': img_orig,
        }
