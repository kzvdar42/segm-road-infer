import cv2
import numpy as np


def colorize_pred(result: np.ndarray, palette: np.ndarray) -> np.ndarray:
    if len(result.shape) == 2:
        result = cv2.cvtColor(result, cv2.COLOR_GRAY2BGR)
    cv2.LUT(result, palette[np.newaxis, ...], result)
    return result


def apply_mask(img: np.ndarray, new_patch: np.ndarray, alpha: float = 1) -> np.ndarray:
    """Put patch on top of the image."""
    patch_mask = (new_patch != 0) * alpha
    res_img = img * (1-patch_mask) + new_patch * patch_mask
    np.clip(res_img, 0, 255, dtype=np.uint8, casting='unsafe', out=img)
    return img
