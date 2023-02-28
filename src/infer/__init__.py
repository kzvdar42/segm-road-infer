from .base_predictor import BasePredictor
from .detectron_predictor import DetectronPredictor
from .onnx_predictor import ONNXPredictor
from .openvino_predictor import OpenVinoPredictor

__all__ = ['BasePredictor', 'DetectronPredictor', 'ONNXPredictor', 'load_predictor']

def load_predictor(model_cfg: dict) -> BasePredictor:
    """Load predictor based on `model_cfg`."""
    model_type = model_cfg.pop('model_type')
    if model_type == 'onnx':
        predictor = ONNXPredictor(model_cfg)
    elif model_type == 'torch':
        predictor = DetectronPredictor(model_cfg)
    elif model_type == 'openvino':
        predictor = OpenVinoPredictor(model_cfg)
    else:
        raise ValueError(f'Unknown model type ({model_type})')
    return predictor
