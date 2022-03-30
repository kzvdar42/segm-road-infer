import os
import yaml
from typing import Callable, Generator

import addict
import numpy as np

maskformer_home_path = os.environ.get('MASKFORMER_HOME')

def load_model_config(cfg_path: str) -> addict.Dict:
    with open(cfg_path, 'r') as in_stream:
        cfg = yaml.safe_load(in_stream)
        if 'config_file' in cfg.keys():
            if not cfg['config_file'].startswith('/'):
                cfg['config_file'] = os.path.join(maskformer_home_path, cfg['config_file'])
        return addict.Dict(cfg)

def create_folder_for_file(file_path: str) -> None:
    folder_path = os.path.split(file_path)[0]
    if folder_path:
        os.makedirs(folder_path, exist_ok=True)


def is_image(path: str) -> bool:
    path = path.lower()
    for ext in ['jpg', 'jpeg', 'png']:
        if path.endswith(ext):
            return True
    return False


def get_subfolders_with_files(path: str, is_file_func: Callable, yield_by_one: bool = False) -> Generator:
    for dp, _, fn in os.walk(path):
        file_paths = [os.path.join(dp, f) for f in fn if is_file_func(f)]
        if file_paths:
            if yield_by_one:
                for file_path in file_paths:
                    yield file_path
            else:
                yield file_paths


def get_classes(dataset_type: str):
    classes = []
    if isinstance(dataset_type, list):
        classes = dataset_type
    elif dataset_type == 'cityscapes':
        with open('cityscapes-classes.txt') as in_file:
            classes = [c.strip() for c in in_file.read().split('\n') if c.strip()]
 `   elif dataset_type == 'mapillary':
        with open('mapillary-classes.txt') as in_file:
            classes = [c.strip() for c in in_file.read().split('\n') if c.strip()]
    else:
        raise ValueError(f'Unknown model dataset type! Got {dataset_type}')
    
    cls_name_to_id = {c:c_id for c_id, c in enumerate(classes)}
    cls_id_to_name = {c_id:c for c, c_id in cls_name_to_id.items()}
    return classes, cls_name_to_id, cls_id_to_name


def cityscapes_palette(grayscale: bool = False):
    if grayscale:
        return [[0,  0, 0], [1,  1, 1], [2,  2,  2], [3, 3, 3], [4, 4, 4], 
                [5, 5, 5], [6, 6, 6], [7, 7, 7], [8, 8, 8], [9, 9, 9], [10, 10, 10],
                [11, 11, 11], [12, 12, 12], [13, 13, 13], [14, 14, 14], 
                [15, 15, 15], [16, 16, 16], [17, 17, 17], [18, 18, 18], [19, 19, 19]]
    return [[128, 64, 128], [244, 35, 232], [70, 70, 70], [102, 102, 156],
            [190, 153, 153], [153, 153, 153], [250, 170, 30], [220, 220, 0],
            [107, 142, 35], [152, 251, 152], [70, 130, 180], [220, 20, 60],
            [255, 0, 0], [0, 0, 142], [0, 0, 70], [0, 60, 100], [0, 80, 100],
            [0, 0, 230], [119, 11, 32], [255, 255, 255]]


def ade_palette(grayscale: bool = False):
    if grayscale:
        raise Exception('Grayscale ade palette not inplemented, please add.')
    """ADE20K palette for external use."""
    return [[120, 120, 120], [180, 120, 120], [6, 230, 230], [80, 50, 50],
            [4, 200, 3], [120, 120, 80], [140, 140, 140], [204, 5, 255],
            [230, 230, 230], [4, 250, 7], [224, 5, 255], [235, 255, 7],
            [150, 5, 61], [120, 120, 70], [8, 255, 51], [255, 6, 82],
            [143, 255, 140], [204, 255, 4], [255, 51, 7], [204, 70, 3],
            [0, 102, 200], [61, 230, 250], [255, 6, 51], [11, 102, 255],
            [255, 7, 71], [255, 9, 224], [9, 7, 230], [220, 220, 220],
            [255, 9, 92], [112, 9, 255], [8, 255, 214], [7, 255, 224],
            [255, 184, 6], [10, 255, 71], [255, 41, 10], [7, 255, 255],
            [224, 255, 8], [102, 8, 255], [255, 61, 6], [255, 194, 7],
            [255, 122, 8], [0, 255, 20], [255, 8, 41], [255, 5, 153],
            [6, 51, 255], [235, 12, 255], [160, 150, 20], [0, 163, 255],
            [140, 140, 140], [250, 10, 15], [20, 255, 0], [31, 255, 0],
            [255, 31, 0], [255, 224, 0], [153, 255, 0], [0, 0, 255],
            [255, 71, 0], [0, 235, 255], [0, 173, 255], [31, 0, 255],
            [11, 200, 200], [255, 82, 0], [0, 255, 245], [0, 61, 255],
            [0, 255, 112], [0, 255, 133], [255, 0, 0], [255, 163, 0],
            [255, 102, 0], [194, 255, 0], [0, 143, 255], [51, 255, 0],
            [0, 82, 255], [0, 255, 41], [0, 255, 173], [10, 0, 255],
            [173, 255, 0], [0, 255, 153], [255, 92, 0], [255, 0, 255],
            [255, 0, 245], [255, 0, 102], [255, 173, 0], [255, 0, 20],
            [255, 184, 184], [0, 31, 255], [0, 255, 61], [0, 71, 255],
            [255, 0, 204], [0, 255, 194], [0, 255, 82], [0, 10, 255],
            [0, 112, 255], [51, 0, 255], [0, 194, 255], [0, 122, 255],
            [0, 255, 163], [255, 153, 0], [0, 255, 10], [255, 112, 0],
            [143, 255, 0], [82, 0, 255], [163, 255, 0], [255, 235, 0],
            [8, 184, 170], [133, 0, 255], [0, 255, 92], [184, 0, 255],
            [255, 0, 31], [0, 184, 255], [0, 214, 255], [255, 0, 112],
            [92, 255, 0], [0, 224, 255], [112, 224, 255], [70, 184, 160],
            [163, 0, 255], [153, 0, 255], [71, 255, 0], [255, 0, 163],
            [255, 204, 0], [255, 0, 143], [0, 255, 235], [133, 255, 0],
            [255, 0, 235], [245, 0, 255], [255, 0, 122], [255, 245, 0],
            [10, 190, 212], [214, 255, 0], [0, 204, 255], [20, 0, 255],
            [255, 255, 0], [0, 153, 255], [0, 41, 255], [0, 255, 204],
            [41, 0, 255], [41, 255, 0], [173, 0, 255], [0, 245, 255],
            [71, 0, 255], [122, 0, 255], [0, 255, 184], [0, 92, 255],
            [184, 255, 0], [0, 133, 255], [255, 214, 0], [25, 194, 194],
            [102, 255, 0], [92, 0, 255]]


def colorize_pred(result: np.ndarray, grayscale: bool) -> np.ndarray:
    palette = ade_palette(grayscale) if np.max(result) > 20 else cityscapes_palette(grayscale)
    res_img = np.zeros((*result.shape, 3), dtype=np.uint8)
    for cls_idx in np.unique(result):
        res_img[result == cls_idx] = palette[cls_idx]
    return res_img


def apply_mask(img: np.ndarray, new_patch: np.ndarray, alpha: bool = 1) -> np.ndarray:
    """Put patch on top of the image."""
    patch_mask = (new_patch != 0) * alpha
    res_img = img * (1-patch_mask) + new_patch * patch_mask
    res_img = np.clip(res_img, 0, 255, dtype=np.uint8, casting='unsafe')
    return res_img
