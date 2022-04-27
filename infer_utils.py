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
import torch.nn.functional as F
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
        if rel_ego_mask_name.isdigit():
            rel_ego_mask_name = int(rel_ego_mask_name)
        img_name_to_ego_mask_paths[rel_ego_mask_name] = ego_mask_path
    print(f'Indexed ego masks, found {len(img_name_to_ego_mask_paths)}')
    return img_name_to_ego_mask_paths

class BaseDataset(Dataset, ABC):

    def __init__(self, cfg: addict.Dict, image_load_format: str):
        # data load config
        self.image_load_format = image_load_format
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
        img_mask = cv2.imread(mask_path, -1)
        return cv2.resize(img_mask, self.input_shape, interpolation=cv2.INTER_NEAREST)

    def load_nearest_mask(self, img_path):
        if self.img_name_to_ego_mask_paths is None:
            return None
        ego_mask_rel_path = os.path.splitext(os.path.relpath(img_path, self.base_path))[0]
        # if indexing based on numbers
        if ego_mask_rel_path.isdigit():
            ego_mask_num = int(ego_mask_rel_path)
            mask_idx = np.searchsorted(self.img_name_keys, ego_mask_num)
            img_mask = self.cached_load_mask(self.img_name_values[mask_idx])
            self._img_mask = img_mask
            return img_mask
        # if indexing based on strings
        if ego_mask_rel_path in self.img_name_to_ego_mask_paths:
            self._img_mask = self.cached_load_mask(self.img_name_to_ego_mask_paths[ego_mask_rel_path])
        return self.img_mask

    def preprocess_numpy(self, img: np.ndarray, img_mask: np.ndarray = None,
                   resize : bool = True) -> torch.Tensor:
        # Resize
        if resize and self.input_shape is not None:
            img = cv2.resize(img, self.input_shape, interpolation=cv2.INTER_LINEAR)
        # Convert to RGB/BGR if needed
        if self.image_format != self.image_load_format:
            cv2.cvtColor(img, cv2.COLOR_BGR2RGB, img)
            # img = img[..., ::-1]
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
    def postprocess(cls, preds: torch.Tensor, metadata: Dict, img_masks: torch.Tensor,
                    ego_mask_cls_id: int, resize_img: bool = False) -> torch.Tensor:
        """Reshape predictions back to original image shape."""
        assert len(preds) == len(metadata)
        # preds = preds.type(torch.uint8)
        if img_masks is not None:
            preds.masked_fill_(img_masks, ego_mask_cls_id)
            # preds[img_masks] = ego_mask_cls_id
        # if resize_img and (preds.shape[1] != metadata[0]['height'] or preds.shape[2] != metadata[0]['width']):
            # FIXME: Maybe there is a better way?
            # res = [cv2.resize(pred, (metadata[0]['width'], metadata[0]['height']), interpolation=cv2.INTER_NEAREST) for pred in preds.cpu().numpy()]
            # preds = np.stack(res, 0)
            # import torchvision.transforms.functional as F
            # preds = F.resize(preds, (metadata[0]['width'], metadata[0]['height']), interpolation=F.InterpolationMode.NEAREST)
        return preds


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
        super().__init__(cfg, image_load_format='bgr')
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
        height, width = img.shape[:2]
        img = self.preprocess_numpy(img, img_mask)
        if img_mask is not None:
            img_mask = torch.from_numpy(img_mask == 255)

        return img, img_mask, {
            "height": height, "width": width, 'image_path': img_path,
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
        super().__init__(cfg, image_load_format='bgr')
        self.video_path = video_path
        self.base_path = os.path.split(video_path)[0]
        self.orig_width, self.orig_height, self.len, self.fps = self.get_video_metadata(video_path)
        self.width, self.height = self.input_shape
        self.n_skip_frames = n_skip_frames
        self.cap = None

    def __len__(self):
        return int(np.ceil(self.len / max(self.n_skip_frames, 1)))
    
    def get_frame(self, index: int) -> np.ndarray:
        # If tries to get not the next frame, set cap pos to right position
        if self.cap is None:
            self.cap = cv2.VideoCapture(self.video_path)
        cap_pos = int(self.cap.get(cv2.CAP_PROP_POS_FRAMES))
        if cap_pos != index:
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, index)
        return self.cap.read()[1]

    def __getitem__(self, index: int) -> Dict:
        index = index * max(1, self.n_skip_frames)
        img = self.get_frame(index)#.astype(np.float32)

        # Add img_path for consistency
        img_path = os.path.join(self.base_path, f'{index+1:0>5}.jpg')
        # set nearest image_mask
        img_mask = self.load_nearest_mask(img_path)

        img = self.preprocess_numpy(img, img_mask)
        if img_mask is not None:
            img_mask = torch.from_numpy(img_mask == 255)
        return img, img_mask, {
            "height": self.orig_height, "width": self.orig_width, 'image_path': img_path,
        }


def ffmpeg_start_in_process(ffmpeg_args, in_filename, scale):
    return (
        ffmpeg
        .input(in_filename) #  hwaccel_output_format='cuda', hwaccel='cuda', vcodec='hevc_cuvid'
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
        .input('pipe:', format='rawvideo', pix_fmt='gray', s='{}x{}'.format(in_width, in_height), framerate=fps) # , hwaccel='cuvid', hwaccel_output_format='cuda', hwaccel='auto'
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

def ffmpeg_start_out_imgs_process(ffmpeg_args, out_path, out_format, in_width, in_height): # , out_width, out_height
    return (
        ffmpeg
        .input('pipe:', format='rawvideo', pix_fmt='gray', s='{}x{}'.format(in_width, in_height))
        # .filter('scale', out_width, out_height, flags='neighbor')
        .output(
            os.path.join(out_path, f'%05d.{out_format}'),
            pix_fmt='gray',
        )
        .global_args(*ffmpeg_args.out_global_args.split(' ') if ffmpeg_args.out_global_args else [])
        .overwrite_output()
        .run_async(pipe_stdin=True)
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
    k = width*height
    img = np.stack((
        np.frombuffer(in_bytes[0:k],dtype=np.uint8).reshape((height, width)),
        # TODO: is there something faster than this?
        # np.frombuffer(in_bytes[k:k+(k//4)],dtype=np.uint8).reshape((height//2, width//2)).repeat(2, axis=-1).repeat(2, axis=-2),
        # np.frombuffer(in_bytes[k+(k//4):],dtype=np.uint8).reshape((height//2, width//2)).repeat(2, axis=-1).repeat(2, axis=-2),
        cv2.resize(np.frombuffer(in_bytes[k:k+(k//4)],dtype=np.uint8).reshape((height//2, width//2)), None, fx=2, fy=2, interpolation=cv2.INTER_NEAREST),
        cv2.resize(np.frombuffer(in_bytes[k+(k//4):],dtype=np.uint8).reshape((height//2, width//2)), None, fx=2, fy=2, interpolation=cv2.INTER_NEAREST),
    ), axis=-1)
    return cv2.cvtColor(img, cv2.COLOR_YUV2RGB)

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
        super().__init__(cfg, image_load_format='rgb')
        self.video_path = video_path
        self.base_path = os.path.split(video_path)[0]
        self.orig_width, self.orig_height, self.len, self.fps = self.get_video_metadata(video_path)
        self.width, self.height = self.input_shape
        assert n_skip_frames == 0, "FfmpegVideoDataset doesn\'t support n_skip_frames"
        self.index = 0
        self.in_pipe = ffmpeg_start_in_process(cfg.ffmpeg, video_path, self.input_shape)
    
    def __len__(self) -> int:
        return self.len
    
    def get_next_frame(self, idx: int):
        # read buffer
        assert self.index == idx, "Tried to access frames out of order!"
        in_bytes = self.in_pipe.stdout.read(self.width * self.height * 3 // 2)
        self.index += 1
        if not in_bytes:
            raise StopIteration
        # decode buffer
        # img = decode_to_torch(in_bytes, self.height, self.width, device)
        img = decode_to_numpy(in_bytes, self.height, self.width).astype(np.float32)
        return img

    def __getitem__(self, idx) -> Dict:
        # add img_path for consistency
        img_path = os.path.join(self.base_path, f'{self.index+1:0>5}.jpg')

        # get nearest image_mask
        img_mask = self.load_nearest_mask(img_path)

        # get frame and transform (no need to resize, as it's already done in ffmpeg)
        img = self.get_next_frame(idx)
        img = self.preprocess_numpy(img, img_mask, resize=False)
        if img_mask is not None:
            img_mask = torch.from_numpy(img_mask == 255)
        # img = torch.as_tensor(img)
        # img = self.preprocess_torch(img, img_mask)
        return img, img_mask, {
            "height": self.orig_height, "width": self.orig_width, "image_path": img_path,
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
        ort_outputs = self.ort_session.get_outputs()
        assert len(ort_inputs) == 1 and len(ort_outputs) == 1, "Support only models with one input/output"
        self.input_shape = ort_inputs[0].shape
        self.input_name = ort_inputs[0].name
        self.output_shape = ort_outputs[0].shape
        self.output_name = ort_outputs[0].name
        self.io_binding = self.ort_session.io_binding()
        self.seg_output = None

    def __call__(self, in_batch):
        # Create output tensor
        if self.seg_output is None or self.seg_output.shape[1] != in_batch.shape[0]:
            self.seg_output = torch.empty((1, in_batch.shape[0], in_batch.shape[-2], in_batch.shape[-1]), dtype=torch.int64, device='cuda')
            self.io_binding.bind_output(name=self.output_name, device_type='cuda', device_id=0, element_type=np.float32, shape=tuple(self.seg_output.shape), buffer_ptr=self.seg_output.data_ptr())

        self.io_binding.bind_input(name=self.input_name, device_type='cuda', device_id=0, element_type=np.float32, shape=tuple(in_batch.shape), buffer_ptr=in_batch.data_ptr())
        torch.cuda.synchronize() # sync for non_blocking data transfer
        self.ort_session.run_with_iobinding(self.io_binding)
        return self.seg_output[0]


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
                return self.model(batch)
