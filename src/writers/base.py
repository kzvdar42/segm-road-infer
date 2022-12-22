from abc import ABC, abstractmethod

import addict
import numpy as np


class AbstractWriter(ABC):

    def __init__(self, cfg: addict.Dict):
        self.cfg = cfg
        self._start_process(cfg)

    @abstractmethod
    def _start_process(self, cfg: addict.Dict):
        pass

    @abstractmethod
    def __call__(self, detections: np.ndarray, metadata: dict) -> None:
        pass

    def close(self) -> None:
        pass
