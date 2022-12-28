import os

import cv2
import numpy as np

from .path import create_folder_for_file


def colorize_pred(result: np.ndarray, palette: np.ndarray) -> np.ndarray:
    if len(result.shape) == 2:
        result = cv2.cvtColor(result, cv2.COLOR_GRAY2BGR)
    cv2.LUT(result, palette[np.newaxis, ...], result)
    return result


def apply_mask(img: np.ndarray, new_patch: np.ndarray, alpha: float = 1) -> np.ndarray:
    """Put patch on top of the image."""
    patch_mask = (new_patch != 0) * alpha
    unclipped_res = img * (1-patch_mask) + new_patch * patch_mask
    np.clip(unclipped_res, 0, 255, dtype=np.uint8, casting='unsafe', out=img)
    return img


def show_preds(predictions: np.ndarray, metadata: dict, cls_palette: np.ndarray, show: bool = True,
               save_to: str = None, window_size: tuple[int, int] = None) -> bool:
    """Show or save colorized predictions."""
    window_size = window_size or (1280, 720)
    for pred, meta in zip(predictions, metadata):
        img_name = os.path.split(meta['image_path'])[1]
        img_orig = meta['img_orig']
        if cls_palette is None:
            res_img = cv2.cvtColor(pred, cv2.COLOR_GRAY2BGR)
        else:
            res_img = colorize_pred(pred, cls_palette)
        res_img = cv2.resize(res_img, img_orig.shape[:2][::-1])
        res_img = apply_mask(img_orig, res_img, alpha=0.5)
        if save_to is not None:
            out_path = os.path.join(save_to, img_name)
            create_folder_for_file(out_path)
            cv2.imwrite(out_path, res_img)
        if show:
            cv2.imshow('pred_vis', cv2.resize(res_img, window_size))
            # esc to quit
            if cv2.waitKey(0) == 27:
                return True
    return False
