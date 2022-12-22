from abc import ABC
import os
from functools import lru_cache

import addict
from PIL import Image
import cv2
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset

from src.utils.index import index_ego_masks

class BaseDataset(Dataset, ABC):

    def __init__(self, cfg: addict.Dict, image_load_format: str, return_raw_imgs: bool = False):
        # data load config
        self.image_load_format = image_load_format
        self.return_raw_imgs = return_raw_imgs
        # ego masks config
        self._img_mask = None
        self.img_name_to_ego_mask_paths = None
        self.base_path = cfg.base_path
        if cfg.apply_ego_mask_from:
            self.img_name_to_ego_mask_paths = index_ego_masks(cfg.apply_ego_mask_from)
            self.img_name_keys = np.array(list(self.img_name_to_ego_mask_paths.keys()))
            self.img_name_values = np.array(list(self.img_name_to_ego_mask_paths.values()))
        # model cfgs
        self.image_format = cfg.model_cfg.image_format
        self.input_shape = cfg.model_cfg.input_shape
        self.model_type = cfg.model_cfg.model_type
        assert type(cfg.model_cfg.img_mean) is type(cfg.model_cfg.img_std), \
            "both mean and std should be provided"
        self.img_mean, self.img_std = cfg.model_cfg.img_mean, cfg.model_cfg.img_std
        if cfg.model_cfg.img_mean is not None:
            self.img_mean = np.array(cfg.model_cfg.img_mean, dtype=np.float64).reshape(1, -1)
            self.img_stdinv = 1 / np.array(cfg.model_cfg.img_std, dtype=np.float64).reshape(1, -1)
            # self.img_mean = torch.FloatTensor(cfg.model_cfg.img_mean).view(1,1,1,-1)
            # self.img_std = torch.FloatTensor(cfg.model_cfg.img_std).view(1,1,1,-1)
        assert self.model_type in ['onnx', 'torch']
        assert self.image_format in ['rgb', 'bgr']

    @property
    def img_mask(self):
        return self._img_mask

    @img_mask.setter
    def img_mask(self, img_mask):
        self._img_mask = cv2.resize(img_mask, self.input_shape, interpolation=cv2.INTER_NEAREST)

    @lru_cache(maxsize=32)
    def cached_load_mask(self, mask_path: str):
        img_mask = np.array(Image.open(mask_path)) # cv2.imread(mask_path, -1)
        return cv2.resize(img_mask, self.input_shape, interpolation=cv2.INTER_NEAREST)

    def load_nearest_mask(self, img_path):
        if self.img_name_to_ego_mask_paths is None:
            return None
        ego_mask_rel_path = os.path.splitext(os.path.relpath(img_path, self.base_path))[0]
        # if indexing based on numbers
        if ego_mask_rel_path.isdigit():
            ego_mask_num = int(ego_mask_rel_path)
            mask_idx = np.searchsorted(self.img_name_keys, ego_mask_num)
            try:
                img_mask = self.cached_load_mask(self.img_name_values[mask_idx])
            except IndexError:
                img_mask = self.cached_load_mask(self.img_name_values[-1])
            self._img_mask = img_mask
            return img_mask
        # if indexing based on strings
        if ego_mask_rel_path in self.img_name_to_ego_mask_paths:
            self._img_mask = self.cached_load_mask(self.img_name_to_ego_mask_paths[ego_mask_rel_path])
        return self.img_mask

    def preprocess_numpy(self, img: np.ndarray, img_mask: np.ndarray = None,
                   resize : bool = True) -> torch.Tensor:
        # Resize
        if resize and self.input_shape is not None and (img.shape[0] != self.input_shape[1] or img.shape[1] != self.input_shape[0]):
            img = cv2.resize(img, self.input_shape, interpolation=cv2.INTER_LINEAR)
        # Convert to RGB/BGR if needed
        if self.image_format != self.image_load_format:
            cv2.cvtColor(img, cv2.COLOR_BGR2RGB, img)
        # Apply mask if provided
        if img_mask is not None:
            cv2.bitwise_and(img, img, img, mask=img_mask)
        # Normalize
        if self.img_mean is not None:
            cv2.subtract(img, self.img_mean, img)
            cv2.multiply(img, self.img_stdinv, img)
        return torch.from_numpy(img.transpose((2, 0, 1)))

    def preprocess_torch(self, img: torch.Tensor, img_mask: torch.Tensor = None,
                         resize : bool = True) -> torch.Tensor:
        # Resize
        if resize and self.input_shape is not None:
            img = F.interpolate(img, self.input_shape, mode='linear')
        # Convert to RGB/BGR if needed
        if self.image_format != self.image_load_format:
            img = img[:, :, ::-1]
        # Apply mask if provided
        if img_mask is not None:
            img[img_mask] = 0
        # Normalize
        if self.img_mean is not None:
            img = (img - self.img_mean) / self.img_std
        return img.permute((2, 0, 1))

    def collate_fn(self, batch):
        images, img_masks, metadata = zip(*batch)
        img_masks = None if img_masks[0] is None else torch.stack(img_masks, 0)
        return torch.stack(images, 0), img_masks, metadata

    def collate_fn_numpy(self, batch):
        images, img_masks, metadata = zip(*batch)
        images = np.stack(images, 0)
        img_masks = None if img_masks[0] is None else np.stack(img_masks, 0)

        if self.image_format != self.image_load_format:
            images = images[..., ::-1]
        # Apply mask if provided
        if img_masks is not None:
            images[img_masks] = 0
            img_masks = torch.from_numpy(img_masks)
        # Normalize
        if self.img_mean is not None:
            images = (images - self.img_mean) / self.img_std

        images = images.transpose((0, 3, 1, 2))
        images = torch.from_numpy(images)
        return images, img_masks, metadata

    def collate_fn_torch(self, batch):
        images, img_masks, metadata = zip(*batch)
        images = torch.stack(images, 0)
        img_masks = None if img_masks[0] is None else torch.stack(img_masks, 0)

        if self.image_format != self.image_load_format:
            images = images[:, :, ::-1]
        # Apply mask if provided
        if img_masks is not None:
            images[img_masks] = 0
        # Normalize
        if self.img_mean is not None:
            images = (images - self.img_mean) / self.img_std

        images = images.permute((0, 3, 1, 2))
        return images, img_masks, metadata

    @classmethod
    def postprocess(cls, preds: torch.Tensor, metadata: dict, img_masks: torch.Tensor,
                    ego_mask_cls_id: int, resize_img: bool = False) -> torch.Tensor:
        """Reshape predictions back to original image shape."""
        assert len(preds) == len(metadata)
        # preds = preds.type(torch.uint8)
        if img_masks is not None:
            preds.masked_fill_(img_masks, ego_mask_cls_id)
            # preds[img_masks] = ego_mask_cls_id
        if resize_img and (preds.shape[1] != metadata[0]['height'] or preds.shape[2] != metadata[0]['width']):
            # FIXME: Maybe there is a better way?
            res = [cv2.resize(pred, (metadata[0]['width'], metadata[0]['height']), interpolation=cv2.INTER_NEAREST) for pred in preds.cpu().numpy()]
            preds = torch.from_numpy(np.stack(res, 0))
            # import torchvision.transforms.functional as F
            # preds = F.resize(preds, (metadata[0]['width'], metadata[0]['height']), interpolation=F.InterpolationMode.NEAREST)
        return preds
