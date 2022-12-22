import time
import threading, queue

import addict
import cv2
from PIL import Image
import numpy as np

from src.utils.path import create_folder_for_file, get_out_path

from .base import AbstractWriter


def save_preds_as_masks(predictions: np.ndarray, metadata: dict, in_base_path: str,
                        out_path: str, ext: str, cls_palette: np.ndarray = None) -> None:
    """Save predicted mask."""
    for pred, meta in zip(predictions, metadata):
        pred_out_path = get_out_path(meta['image_path'], out_path, in_base_path, ext)
        create_folder_for_file(pred_out_path)
        if pred.dtype == np.uint16 and len(pred.shape) == 3:
            try:
                cv2.cvtColor(pred, cv2.COLOR_RGB2BGR, pred)
                is_saved = cv2.imwrite(pred_out_path, pred)
            except cv2.error:
                is_saved = False
            finally:
                if not is_saved:
                    print(f'Failed to save!', pred_out_path, type(pred), pred.shape)
            continue
        pred_image = Image.fromarray(pred)
        if cls_palette is not None:
            pred_image.putpalette(cls_palette)
        try:
            pred_image.save(pred_out_path)
        except OSError:
            print(f'Failed to save!', pred_out_path, type(pred), pred.shape)

def create_img_writer_thread(in_base_path: str, out_path: str,
                             ext: str, cls_palette: np.ndarray = None):
    """Create image writer thread.
    
    Returns created thread and it's input queue.
    """
    q = queue.Queue()

    def worker(in_base_path: str, out_path: str, ext: str, cls_palette: np.ndarray = None):
        while True:
            try:
                data = q.get_nowait()
                if data is None:
                    break
                predictions, metadata = data
                save_preds_as_masks(predictions, metadata, in_base_path, out_path, ext, cls_palette)
                q.task_done()
            except queue.Empty:
                time.sleep(0.5)
                continue

    # Turn-on the worker thread.
    thread = threading.Thread(target=worker, args=[in_base_path, out_path, ext, cls_palette], daemon=False)
    thread.start()
    return thread, q

class PillowWriter(AbstractWriter):
        
    def _start_process(self, cfg: addict.Dict):
        self.img_num_bits: str = cfg.img_num_bits
        assert self.img_num_bits in ('8bit', '16bit'), \
            "Pillow can only write in 8/16 bit mode."
        self.use_tracking_id = cfg.get('use_tracking_id', True)
        self.in_base_path = cfg.in_base_path
        self.out_path = cfg.out_path
        self.out_width, self.out_height = cfg.out_width, cfg.out_height
        self.ext = cfg.ext
        self.cls_palette = cfg.get('cls_palette')
        self.write_thread, self.out_queue = create_img_writer_thread(
            self.in_base_path, self.out_path, self.ext, self.cls_palette
        )

    def __call__(self, predictions: np.ndarray, metadata: dict) -> None:
        self.out_queue.put_nowait((predictions, metadata))

    def close(self) -> None:
        # Exit from or kill out writer thread
        print('Waiting for pillow writing thread to exit...')
        self.out_queue.put(None)
        self.write_thread.join()
