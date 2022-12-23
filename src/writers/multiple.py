from typing import Optional

import addict
import numpy as np

from .base import AbstractWriter


class MultipleWriters(AbstractWriter):

    def _start_process(self, cfg: addict.Dict):
        self.writers: list[AbstractWriter] = cfg.writers

    @property
    def exit_code(self) -> Optional[int]:
        exit_code = None
        for writer in self.writers:
            if writer.exit_code is not None:
                if exit_code is None:
                    exit_code = writer.exit_code
                elif writer.exit_code > exit_code:
                    exit_code = writer.exit_code
        return exit_code

    def __call__(self, detections: np.ndarray, metadata: dict) -> None:
        for writer in self.writers:
            writer(detections, metadata)

    def close(self) -> None:
        for writer in self.writers:
            writer.close()
