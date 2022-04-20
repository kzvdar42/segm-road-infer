from abc import ABC
import os
import sys
from functools import lru_cache
from typing import List, Dict

import addict
import cv2
import numpy as np
import torch
from tqdm.auto import tqdm
import ffmpeg
import onnxruntime as ort
from torch.utils.data import Dataset
try:
    from turbojpeg import TurboJPEG
    jpeg = TurboJPEG()
    use_turbojpeg = True
except:
    use_turbojpeg = False

from utils import (
    get_subfolders_with_files, is_image, get_out_path
)


def index_ego_masks(root_path: str) -> Dict[str, str]:
    img_name_to_ego_mask_paths = {}
    for ego_mask_path in get_subfolders_with_files(root_path, is_image, True):
        rel_ego_mask_name = os.path.splitext(os.path.relpath(ego_mask_path, root_path))[0]
        img_name_to_ego_mask_paths[rel_ego_mask_name] = ego_mask_path
    print(f'Indexed ego masks, found {len(img_name_to_ego_mask_paths)}')
    return img_name_to_ego_mask_paths

class BaseDataset(Dataset, ABC):

    def __init__(self, cfg: addict.Dict):
        # ego masks config
        self._img_mask = None
        self.img_name_to_ego_mask_paths = None
        self.base_path = cfg.base_path
        if cfg.apply_ego_mask_from:
            self.img_name_to_ego_mask_paths = index_ego_masks(cfg.apply_ego_mask_from)
        # model cfgs
        self.image_format = cfg.model_cfg.image_format
        self.input_shape = cfg.model_cfg.input_shape
        self.model_type = cfg.model_cfg.model_type
        self.img_mean = cfg.model_cfg.img_mean
        self.img_std = cfg.model_cfg.img_std
        assert self.model_type in ['onnx', 'torch']
        assert self.image_format in ['rgb', 'bgr']
    
    @property
    def img_mask(self):
        return self._img_mask

    @img_mask.setter
    def img_mask(self, img_mask):
        self._img_mask = cv2.resize(img_mask, self.input_shape, interpolation=cv2.INTER_NEAREST) == 255

    @lru_cache(maxsize=16)
    def _cached_load_mask(self, mask_path):
        return cv2.imread(mask_path, -1)

    def load_nearest_mask(self, img_path):
        if self.img_name_to_ego_mask_paths is None:
            return None
        ego_mask_rel_path = os.path.splitext(os.path.relpath(img_path, self.base_path))[0]
        if ego_mask_rel_path in self.img_name_to_ego_mask_paths:
            self.img_mask = self._cached_load_mask(self.img_name_to_ego_mask_paths[ego_mask_rel_path])
        return self.img_mask

    def preprocess(self, img: np.ndarray) -> np.ndarray:
        # Convert to RGB if needed
        if self.image_format == "rgb":
            img = img[:, :, ::-1]
        # Resize
        if self.input_shape is not None:
            img = cv2.resize(img, self.input_shape, interpolation=cv2.INTER_LINEAR)
        # Apply mask if provided
        if self.img_mask is not None:
            img[self.img_mask] = 0
        # Normalize
        if self.img_mean is not None or self.img_std is not None:
            self.img_mean = self.img_mean or [0, 0, 0]
            self.img_std = self.img_std or [1, 1, 1]
            img = (img - self.img_mean) / self.img_std
        img = img.transpose((2, 0, 1)).astype(np.float32)
        if self.model_type == 'torch':
            img = torch.as_tensor(img)
        return img

    def collate_fn(self, batch):
        images, metadata = [], []
        for sample in batch:
            images.append(sample.pop('image'))
            metadata.append(sample)
        if self.model_type == 'onnx':
            images = np.stack(images, 0)
        elif self.model_type == 'torch':
            images = torch.stack(images, 0)
        else:
            raise ValueError(f'Unknown model type ({self.model_type})')
        return images, metadata

    @classmethod
    def postprocess(cls, preds: List[np.ndarray], metadata: Dict, ego_mask_cls_id: int, resize_img: bool = False) -> List[np.ndarray]:
        """Reshape predictions back to original image shape and convert to uint8."""
        processed = []
        assert len(preds) == len(metadata)
        for pred, meta in zip(preds, metadata):
            # Prepare
            if pred.shape[0] == 1:
               pred = pred[0]
            pred = pred.astype(np.uint8)
            # Apply ego mask
            if meta['mask'] is not None:
                pred[meta['mask']] = ego_mask_cls_id
            # Resize
            if resize_img and (pred.shape[0] != meta['height'] or pred.shape[1] != meta['width']):
                pred = cv2.resize(
                    pred, (meta['width'], meta['height']),
                    interpolation=cv2.INTER_NEAREST,
                )
            processed.append(pred)
        return processed


def index_images_from_folder(input_folder: str, n_skip_frames: int = None) -> List[str]:
    # Index images
    print('Indexing images...')
    image_paths = get_subfolders_with_files(input_folder, is_image, True)
    image_paths = list(image_paths)
    print(f'Found {len(image_paths)} images!')
    if n_skip_frames > 0:
        image_paths_ = image_paths
        image_paths = image_paths[::n_skip_frames]
        # Keep last image
        if image_paths_[-1] not in image_paths:
            image_paths.append(image_paths_[-1])
        print(f'Skipping {n_skip_frames} images each time, {len(image_paths)} left')
    return image_paths


def index_images(args) -> List[str]:
    """Index image files from folder/image/txt_file."""
    image_paths = []
    if args.in_path.endswith('.txt'):
        # First line is the base_path for input images!
        with open(args.in_path) as in_stream:
            args.base_path = in_stream.readline().strip()
            image_paths = [l.strip() for l in in_stream.readlines()]
        print(f'Got {len(image_paths)} from input file.')
    else:
        args.base_path = os.path.abspath(args.in_path)
        image_paths = index_images_from_folder(args.in_path, args.n_skip_frames)
    if args.skip_processed:
        pbar = tqdm(image_paths, desc='Check if processed', leave=False)
        image_paths = []
        for img_path in pbar:
            if os.path.isfile(get_out_path(img_path, args.out_path, args.base_path)):
                image_paths.append(img_path)
        print(f'Skipped already processed files, {len(image_paths)} left')
    return image_paths


class ImageDataset(BaseDataset):
    
    def __init__(self, image_paths: List[str], cfg: addict.Dict):
        super().__init__(cfg)
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
        img = self.imread(img_path)
        height, width = img.shape[:2]
        img = self.preprocess(img)

        return {
            "image": img, "height": height, "width": width, 'image_path': img_path, "mask": img_mask,
        }


class VideoDataset(BaseDataset):
    
    @classmethod
    def get_video_metadata(cls, video_path):
        cap = cv2.VideoCapture(video_path)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        num_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        cap.release()
        del cap
        return width, height, num_frames, fps

    def __init__(self, video_path: str, cfg: addict.Dict, n_skip_frames: int = 0):
        super().__init__(cfg)
        self.video_path = video_path
        self.base_path = os.path.split(video_path)[0]
        self.orig_width, self.orig_height, self.len, self.fps = self.get_video_metadata(video_path)
        self.width, self.height = self.input_shape
        self.n_skip_frames = n_skip_frames
        self.cap = None

    def __len__(self):
        return int(np.ceil(self.len / max(self.n_skip_frames, 1)))

    def __getitem__(self, index: int) -> Dict:
        # If tries to get not the next frame, set cap pos to right position
        index = index * max(1, self.n_skip_frames)
        if self.cap is None:
            self.cap = cv2.VideoCapture(self.video_path)
        cap_pos = int(self.cap.get(cv2.CAP_PROP_POS_FRAMES))
        if cap_pos != index:
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, index)
        img = self.cap.read()[1]
        # Add img_path for consistency
        img_path = os.path.join(self.base_path, f'{index+1:0>5}.jpg')
        
        # set nearest image_mask
        img_mask = self.load_nearest_mask(img_path)

        img = self.preprocess(img)
        return {
            "image": img, "height": self.orig_height, "width": self.orig_width, 'image_path': img_path, "mask": img_mask,
        }


def ffmpeg_start_in_process(ffmpeg_args, in_filename, scale):
    return (
        ffmpeg
        .input(in_filename) #  hwaccel_output_format='cuda' hwaccel='cuda', vcodec='hevc_cuvid'
        .video
        .filter('scale', scale[0], scale[1])
        .filter('setsar', '1')
        .output('pipe:', format='rawvideo', pix_fmt='yuv420p')  # vcodec='h264_nvenc'
        .global_args(*ffmpeg_args.in_global_args.split(' ') if ffmpeg_args.in_global_args else [])
        .run_async(pipe_stdout=True)
    )


def ffmpeg_start_out_process(ffmpeg_args, out_filename, in_width, in_height, out_width, out_height, fps=30):
    return (
        ffmpeg
        .input('pipe:', format='rawvideo', pix_fmt='gray', s='{}x{}'.format(in_width, in_height), framerate=fps) # , hwaccel='cuvid', hwaccel_output_format='cuda'
        .filter('scale', out_width, out_height, flags='neighbor')
        .output(
            out_filename,
            vcodec=ffmpeg_args.out_vcodec,
            pix_fmt=ffmpeg_args.out_pix_fmt,
            **ffmpeg_args.output_args,
        )
        .global_args(*ffmpeg_args.out_global_args.split(' ') if ffmpeg_args.out_global_args else [])
        .overwrite_output()
        .run_async(pipe_stdin=True)
    )

class FfmpegVideoDataset(BaseDataset):

    @classmethod
    def get_video_metadata(cls, video_path):
        probe = ffmpeg.probe(video_path)
        video_stream = next((stream for stream in probe['streams'] if stream['codec_type'] == 'video'), None)
        width = int(video_stream['width'])
        height = int(video_stream['height'])
        num_frames = int(video_stream['nb_frames'])
        fps = int(eval(video_stream['r_frame_rate']))
        return width, height, num_frames, fps
    
    def __init__(self, video_path: str, cfg: addict.Dict, n_skip_frames: int = 0):
        super().__init__(cfg)
        self.video_path = video_path
        self.base_path = os.path.split(video_path)[0]
        self.orig_width, self.orig_height, self.len, self.fps = self.get_video_metadata(video_path)
        self.width, self.height = self.input_shape
        assert n_skip_frames == 0, 'FfmpegVideoDataset doesn\'t support n_skip_frames'
        self.index = 0
        self.in_pipe = ffmpeg_start_in_process(cfg.ffmpeg, video_path, self.input_shape)
    
    def __len__(self) -> int:
        return self.len

    def preprocess(self, img: np.ndarray) -> np.ndarray:
        # Convert to BGR if needed
        if self.image_format == "bgr":
            img = img[:, :, ::-1]
        # Apply mask if provided
        if self.img_mask is not None:
            img[self.img_mask] = 0
        # Normalize
        if self.img_mean is not None or self.img_std is not None:
            self.img_mean = self.img_mean or [0, 0, 0]
            self.img_std = self.img_std or [1, 1, 1]
            img = (img - self.img_mean) / self.img_std
        img = img.transpose((2, 0, 1)).astype(np.float32)
        if self.model_type == 'torch':
            img = torch.as_tensor(img)
        return img

    def __getitem__(self, idx) -> Dict:
        assert self.index == idx
        in_bytes = self.in_pipe.stdout.read(self.width * self.height * 3 // 2)
        self.index += 1
        if not in_bytes:
            raise StopIteration
        # decode buffer
        k = self.width*self.height
        img = np.stack((
            np.frombuffer(in_bytes[0:k],dtype=np.uint8).reshape((self.height, self.width)),
            cv2.resize(np.frombuffer(in_bytes[k:k+(k//4)],dtype=np.uint8).reshape((self.height//2, self.width//2)), (self.width,self.height)),
            cv2.resize(np.frombuffer(in_bytes[k+(k//4):],dtype=np.uint8).reshape((self.height//2, self.width//2)), (self.width,self.height)),
        ), axis=-1)
        img = cv2.cvtColor(img.copy(), cv2.COLOR_YUV2RGB)
        # add img_path for consistency
        img_path = os.path.join(self.base_path, f'{self.index:0>5}.jpg')
        # get nearest image_mask
        img_mask = self.load_nearest_mask(img_path)
        img = self.preprocess(img)
        return {
            "image": img, "height": self.orig_height, "width": self.orig_width, "image_path": img_path, "mask": img_mask,
        }


class Predictor(ABC):
    """Abstract class for prediction model."""

    def __init__(self, cfg):
        self.cfg = cfg
        pass

    def __call__(self, batch):
        self.model(batch)
        pass


class ONNXPredictor(Predictor):
    """Class for ONNX based predictor."""

    def __init__(self, cfg):
        super().__init__(cfg)
        # Set graph optimization level
        so = ort.SessionOptions()
        # so.execution_mode = ort.ExecutionMode.ORT_PARALLEL#ORT_SEQUENTIAL
        so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        # Load model
        self.ort_session = ort.InferenceSession(
            cfg.weights_path, sess_options=so, providers=cfg.providers,
        )
        ort_inputs = self.ort_session.get_inputs()
        assert len(ort_inputs) == 1, "Support only models with one input"
        self.input_shape = ort_inputs[0].shape
        self.input_name = ort_inputs[0].name
    
    def __call__(self, batch):
        return self.ort_session.run(None, {self.input_name: batch})[0][0]


class DetectronPredictor(Predictor):
    """Class for Detectron2 based predictor."""

    @classmethod
    def setup_cfg(cls, our_model_cfg):
        # load config from file and command-line arguments
        from detectron2.config import get_cfg
        from detectron2.projects.deeplab import add_deeplab_config
        cfg = get_cfg()
        add_deeplab_config(cfg)
        maskformer_home_path = os.environ.get('MASKFORMER_HOME')
        mask2former_home_path = os.environ.get('MASK2FORMER_HOME')
        try:
            sys.path.insert(1, mask2former_home_path)
            from mask2former import add_maskformer2_config
            add_maskformer2_config(cfg)
        except:
            print("Coudn't import mask2former, please set MASK2FORMER_HOME env var")
            pass
        try:
            sys.path.insert(1, maskformer_home_path)
            from mask_former import add_mask_former_config
            add_mask_former_config(cfg)
        except:
            pass
        cfg.merge_from_file(our_model_cfg.config_file)
        cfg.merge_from_list(our_model_cfg.opts)
        cfg.freeze()
        return cfg

    def __init__(self, our_model_cfg):
        super().__init__(our_model_cfg)
        self.det2_cfg = self.setup_cfg(our_model_cfg)
        self.use_fp16 = our_model_cfg.use_fp16
        # Load model
        from detectron2.modeling import build_model
        from detectron2.checkpoint import DetectionCheckpointer
        self.model = build_model(self.det2_cfg)
        self.model.eval()
        from mask2former_custom_infer import forward as _forward
        import types
        self.model._forward = self.model.forward
        self.model.forward = types.MethodType(_forward, self.model)
        checkpointer = DetectionCheckpointer(self.model)
        checkpointer.load(our_model_cfg.weights_path)
        assert our_model_cfg.image_format == self.det2_cfg.INPUT.FORMAT.lower(), \
            f'Input format must be the same! Got {our_model_cfg.image_format} and {self.det2_cfg.INPUT.FORMAT.lower()}'
    
    def __call__(self, batch):
        with torch.no_grad():
            with torch.cuda.amp.autocast(self.use_fp16):
                predictions = self.model(batch)
        return self.postprocess(predictions)
    
    def postprocess(self, predictions: List[torch.Tensor]) -> List[np.ndarray]:
        processed = []
        for pred in predictions:
            processed.append(pred['sem_seg'].argmax(dim=0).detach().cpu().numpy())
        return processed
