from abc import ABC

class BasePredictor(ABC):
    """Abstract class for prediction model."""

    def __init__(self, cfg):
        self.cfg = cfg
        pass

    def __call__(self, batch):
        self.model(batch)
        pass
