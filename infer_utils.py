from abc import ABC
import os
import sys
from typing import List, Dict

import addict
import cv2
import numpy as np
import torch
import onnxruntime as ort
from torch.utils.data import Dataset
try:
    from turbojpeg import TurboJPEG
    jpeg = TurboJPEG()
    use_turbojpeg = True
except:
    use_turbojpeg = False

class BaseDataset(Dataset, ABC):

    def __init__(self, cfg: addict.Dict, img_mask=None):
        self.img_mask = img_mask
        self.image_format = cfg.image_format
        self.input_shape = cfg.input_shape
        self.model_type = cfg.model_type
        self.img_mean = cfg.img_mean
        self.img_std = cfg.img_std
        assert self.model_type in ['onnx', 'torch']
        assert self.image_format in ['rgb', 'bgr']
    
    def preprocess(self, img: np.ndarray) -> np.ndarray:
        # Convert to RGB if needed
        if self.image_format == "rgb":
            img = img[:, :, ::-1]
        # Apply mask if provided
        if self.img_mask is not None:
            img[self.img_mask == 255] = 0
        # Resize
        if self.input_shape is not None:
            img = cv2.resize(img, self.input_shape, interpolation=cv2.INTER_LINEAR)
        # Normalize
        if self.img_mean is not None or self.img_std is not None:
            self.img_mean = self.img_mean or [0, 0, 0]
            self.img_std = self.img_std or [1, 1, 1]
            img = (img - self.img_mean) / self.img_std
        img = img.transpose((2, 0, 1)).astype(np.float32)
        if self.model_type == 'torch':
            img = torch.as_tensor(img)
        else:
            img = img# / 255
        return img

    def collate_fn(self, batch):
        images = []
        metadata = []
        for sample in batch:
            if self.model_type == 'onnx':
                images.append(sample.pop('image'))
            elif self.model_type == 'torch':
                images.append(sample.copy())
                del sample['image']
            else:
                raise ValueError(f'Unknown model type ({self.model_type})')
            metadata.append(sample)
        if self.model_type == 'onnx':
            images = np.stack(images, 0)
        return images, metadata
    
    @classmethod
    def postprocess(cls, preds: List[np.ndarray], metadata: Dict) -> List[np.ndarray]:
        """Reshape predictions back to original image shape and convert to uint8."""
        processed = []
        assert len(preds) == len(metadata)
        for pred, meta in zip(preds, metadata):
            if pred.shape[0] == 1:
               pred = pred[0] 
            if pred.shape[0] != meta['height'] or pred.shape[1] != meta['width']:
                pred = cv2.resize(pred.astype(np.uint8), (meta['width'], meta['height']))
            else:
                pred = pred.astype(np.uint8)
            processed.append(pred)
        return processed


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
        # Get data
        img_path = self.image_paths[index]
        
        # Read and transform the image
        img = self.imread(img_path)
        height, width = img.shape[:2]
        img = self.preprocess(img)

        return {
            "image": img, "height": height, "width": width, 'image_path': img_path
        }


class VideoDataset(BaseDataset):

    def __init__(self, video_path: str, cfg: addict.Dict, n_skip_frames: int = 0):
        super().__init__(cfg)
        self.video_path = video_path
        self.base_path = os.path.split(video_path)[0]
        self.cap = cv2.VideoCapture(video_path)
        self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.len = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.n_skip_frames = n_skip_frames
    
    def __len__(self):
        return int(np.ceil(self.len / max(self.n_skip_frames, 1)))
    
    def __getitem__(self, index: int):
        # Get data
        # If tries to get not the next frame, set cap pos to right position
        # index_ = index
        index = index * max(1, self.n_skip_frames)
        # print(f'Loading frame {index_} -> {index}')
        cap_pos = int(self.cap.get(cv2.CAP_PROP_POS_FRAMES))
        if cap_pos != index:
            # print(f'Changing pos from {cap_pos} to {index}!')
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, index)
            # print(f'Changed pos from {cap_pos} to {index}!')
        img = self.cap.read()[1]
        # Add img_path for consistency
        img_path = os.path.join(self.base_path, f'{index+1:0>6}.jpg')
        
        height, width = img.shape[:2]
        img = self.preprocess(img)
        return {
            "image": img, "height": height, "width": width, 'image_path': img_path
        }


class MyDataloader:
    """Barebone dataloader"""

    def __init__(self, dataset, batch_size):
        self.dataset = dataset
        self.batch_size = batch_size
        self.pos = 0
    
    def __len__(self):
        return int(np.ceil(len(self.dataset) / self.batch_size))
    
    def __iter__(self):
        assert self.pos == 0, "Can start only once!"
        return self
    
    def __next__(self):
        batch = []
        for _ in range(self.batch_size):
            if self.pos >= len(self.dataset):
                if batch:
                    return self.dataset.collate_fn(batch)
                else:
                    raise StopIteration
            data = self.dataset[self.pos]
            batch.append(data)
            self.pos += 1
        return self.dataset.collate_fn(batch)

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
        sess_options = ort.SessionOptions()
        sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_EXTENDED
        # Load model
        self.ort_session = ort.InferenceSession(
            cfg.weights_path, providers=cfg.providers,
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
