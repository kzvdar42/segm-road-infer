import subprocess
from typing import Optional

import addict
import numpy as np

from src.utils.ffmpeg import ffmpeg_start_out_process
from src.utils.path import create_folder_for_file

from .base import AbstractWriter


class FfmpegWriter(AbstractWriter):

    @property
    def write_process(self) -> subprocess.Popen:
        if self._write_process is None:
            self._write_process = ffmpeg_start_out_process(
                self.cfg.ffmpeg, self.cfg.out_path, self.in_width, self.in_height,
                self.out_width, self.out_height, self.cfg.out_fps
            )
        return self._write_process

    @property
    def exit_code(self) -> Optional[int]:
        return self.write_process.returncode

    def _start_process(self, cfg: addict.Dict):
        self.img_num_bits: str = cfg.get('img_num_bits', '10bit')
        assert self.img_num_bits in ('8bit', '10bit'), \
            "FFMPEG can only write in 8/10 bit mode."
        self.use_tracking_id = cfg.get('use_tracking_id', True)
        self.max_timeout = None
        if cfg.ffmpeg.max_timeout >= 0:
            self.max_timeout = cfg.ffmpeg.max_timeout
        self._write_process = None
        self.in_width, self.in_height = cfg.get('in_width'), cfg.get('in_height')
        self.out_width, self.out_height = cfg.get('out_width'), cfg.get('out_height')
        # create folder for output file
        create_folder_for_file(cfg.out_path)
        # we will supply 16bit rgb images
        cfg.ffmpeg['in_pix_fmt'] = 'rgb48be'

    def __call__(self, predictions: np.ndarray, metadata: dict) -> None:
        # set output/input shape if not provided by config.
        if self.in_width is None:
            self.in_height, self.in_width = predictions.shape[1], predictions.shape[2]
            if self.out_width is None:
                self.out_height, self.out_width = metadata[0]['height'], metadata[0]['width']

        self.write_process.stdin.write(predictions.tobytes())

    def close(self) -> None:
        # Exit from or kill out writer process
        print('Waiting for ffmpeg to exit...')
        try:
            self.write_process.communicate(timeout=self.max_timeout)
        except subprocess.TimeoutExpired:
            print(f'Waited for {self.max_timeout} seconds. Killing ffmpeg!')
            self.write_process.kill()
        finally:
            print(f'ffmpeg exited with code {self.write_process.returncode}')
        pass

# TODO: Add saving preds with palette (further code is from prev version)
# def save_preds_to_ffmpeg_with_palette(
#             predictions: np.ndarray, metadata: list[dict], cls_palette: np.ndarray,
#             video_writer: subprocess.Popen
#         ) -> None:
#     for pred, meta in zip(predictions, metadata):
#         img_orig = meta['img_orig']
#         img_orig = cv2.resize(img_orig, pred.shape[:2][::-1])
#         if cls_palette is None:
#             res_img = cv2.cvtColor(pred, cv2.COLOR_GRAY2BGR)
#         else:
#             res_img = colorize_pred(pred, cls_palette)
#         res_img = apply_mask(img_orig, res_img, alpha=0.5)
#         video_writer.stdin.write(res_img.astype(np.uint8).tobytes())
