import os
import subprocess
import time
from typing import Tuple, List
import threading, queue
import multiprocessing
import yaml

import addict
import cv2
from PIL import Image
import numpy as np

from src.datasets import FfmpegVideoDataset, VideoDataset
from .image import colorize_pred, apply_mask
from .index import create_folder_for_file, get_out_path
from .ffmpeg import ffmpeg_start_out_process


mask2former_home_path = os.environ.get('MASK2FORMER_HOME')

def load_model_config(cfg_path: str) -> addict.Dict:
    """Load config file for the predictor."""
    with open(cfg_path, 'r') as in_stream:
        cfg = yaml.safe_load(in_stream)
        if 'config_file' in cfg.keys():
            if not cfg['config_file'].startswith('/'):
                assert mask2former_home_path is not None, 'Need to set `mask2former_home_path` env var!'
                cfg['config_file'] = os.path.join(mask2former_home_path, cfg['config_file'])
        return addict.Dict(cfg)


class PseudoTqdm:
    """Class to use as in-place replacement for tqdm then you don't need a pbar."""

    def __init__(self):
        self.start_t = time.time() 
        self.n_runs = 0

    @property
    def rate(self):
        return self.n_runs / (time.time() - self.start_t)

    def update(self, n_runs : int):
        self.n_runs += n_runs
    
    def close(self):
        pass

def show_preds(predictions: np.ndarray, metadata: dict, cls_palette: np.ndarray, show: bool = True,
               save_to: str = None, window_size: Tuple[int, int] = None) -> bool:
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


def save_preds_as_masks(predictions: np.ndarray, metadata: dict, in_base_path: str, out_path: str, cls_palette: np.ndarray, ext: str) -> None:
    for pred, meta in zip(predictions, metadata):
        pred_out_path = get_out_path(meta['image_path'], out_path, in_base_path, ext)
        create_folder_for_file(pred_out_path)
        pred_image = Image.fromarray(pred)
        if cls_palette is not None:
            pred_image.putpalette(cls_palette)
        try:
            pred_image.save(pred_out_path)
        except OSError:
            print(f'Didn\'t save!', pred_out_path, type(pred), pred.shape)


def save_preds_to_ffmpeg(predictions: np.ndarray, video_writer: subprocess.Popen) -> None:
    video_writer.stdin.write(predictions.tobytes())

def save_preds_to_ffmpeg_with_palette(
            predictions: np.ndarray, metadata: list[dict], cls_palette: np.ndarray,
            video_writer: subprocess.Popen
        ) -> None:
    for pred, meta in zip(predictions, metadata):
        img_orig = meta['img_orig']
        img_orig = cv2.resize(img_orig, pred.shape[:2][::-1])
        if cls_palette is None:
            res_img = cv2.cvtColor(pred, cv2.COLOR_GRAY2BGR)
        else:
            res_img = colorize_pred(pred, cls_palette)
        res_img = apply_mask(img_orig, res_img, alpha=0.5)
        video_writer.stdin.write(res_img.astype(np.uint8).tobytes())


def default_ffmpeg_args():
    return dict(
        out_vcodec = "libx264",
        out_pix_fmt = "yuv420p",
        output_args = dict(crf=0),
        in_global_args = "-hide_banner -loglevel error",
        out_global_args = "-hide_banner -loglevel error",
        max_timeout = 15,
    )

def create_img_writer_thread(in_base_path: str, out_path: str, cls_palette: np.ndarray, ext: str):
    q = queue.Queue()

    def worker(in_base_path: str, out_path: str, cls_palette: np.ndarray, ext: str):
        while True:
            try:
                predictions, metadata = q.get_nowait()
                save_preds_as_masks(predictions, metadata, in_base_path, out_path, cls_palette, ext)
                q.task_done()
            except queue.Empty:
                time.sleep(0.5)
                continue

    # Turn-on the worker thread.
    thread = threading.Thread(target=worker, args=[in_base_path, out_path, cls_palette, ext], daemon=False)
    thread.start()
    return thread, q

def create_video_writer_thread(out_queue):
    q = multiprocessing.Queue()

    def worker():
        while True:
            try:
                predictions = q.get_nowait()
                out_queue.write(predictions)
            except queue.Empty:
                time.sleep(0.5)
                continue
    
    # Turn-on the worker process.
    process = multiprocessing.Process(target=worker, daemon=True)
    process.start()
    return process, q

def create_out_writer(args, dataset):
    # Create videowriter if saving as a video
    if args.out_format == 'mp4':
        out_width, out_height, fps = 1920, 1080, 30
        if isinstance(dataset, (FfmpegVideoDataset, VideoDataset)):
            out_width, out_height, fps =  dataset.orig_width, dataset.orig_height, dataset.fps
        print(f'Saving results to videofile with {fps} fps and {(out_width, out_height)} frame size.')
        ffmpeg_out = ffmpeg_start_out_process(
            args.ffmpeg, args.out_path, *args.model_cfg.input_shape, out_width, out_height, fps
        )
        # os.set_blocking(ffmpeg_out.stdin.fileno(), False)
        return ffmpeg_out
    # TODO: Saving images using separate thread is not tested 
    # cls_palette = None if args.only_ego_vehicle else args.cls_palette
    # return create_img_writer_thread(args.base_path, args.out_path, cls_palette, args.out_format)
