from abc import ABC
import os
import sys
from typing import List, Dict

import addict
import cv2
import numpy as np
import torch
import onnxruntime as ort
from torch.utils.data import Dataset, DataLoader
try:
    from turbojpeg import TurboJPEG
    jpeg = TurboJPEG()
    use_turbojpeg = True
except:
    use_turbojpeg = False

class ImageDataset(Dataset):
    
    def __init__(self, image_paths: List[str], cfg: addict.Dict):
        self.image_paths = image_paths
        self.image_format = cfg.image_format
        self.input_shape = cfg.input_shape
        self.model_type = cfg.model_type
        self.img_mean = cfg.img_mean
        self.img_std = cfg.img_std
        self.img_mask = None
        assert self.model_type in ['onnx', 'torch']
        assert self.image_format in ['rgb', 'bgr']
        

    def __len__(self):
        return len(self.image_paths)
    
    def imread(self, img_path: str) -> np.ndarray:
        if use_turbojpeg:
            # TurboJPEG reads in BGR format
            with open(img_path, 'rb') as in_file:
                img = jpeg.decode(in_file.read())
        else:
            img = cv2.imread(img_path)
        if self.image_format == "rgb":
            return img[:, :, ::-1]
        return img
    
    def preprocess(self, img: np.ndarray) -> np.ndarray:
        # Apply mask if provided
        if self.img_mask is not None:
            img[self.img_mask == 255] = 0
        # Resize
        if self.input_shape is not None:
            img = cv2.resize(img, self.input_shape, interpolation=cv2.INTER_LINEAR)
        # Normalize
        if self.img_mean is not None:
            img = (img - self.img_mean) / self.img_std
        img = img.transpose((2, 0, 1)).astype(np.float32)
        if self.model_type == 'torch':
            img = torch.as_tensor(img)
        else:
            img = img# / 255
        return img

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
    
    def postprocess(self, preds: List[np.ndarray], metadata: Dict) -> List[np.ndarray]:
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
