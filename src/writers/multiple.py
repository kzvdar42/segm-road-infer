import addict
import numpy as np

from .base import AbstractWriter


class MultipleWriters(AbstractWriter):

    def _start_process(self, cfg: addict.Dict):
        self.writers: list[AbstractWriter] = cfg.writers

    def __call__(self, detections: np.ndarray, metadata: dict) -> None:
        for writer in self.writers:
            writer(detections, metadata)

    def close(self) -> None:
        for writer in self.writers:
            writer.close()
