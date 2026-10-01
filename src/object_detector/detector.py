"""One public interface for both inference backends."""

from time import perf_counter

import numpy as np

from .backends import OnnxBackend, TorchBackend
from .config import DetectorConfig
from .types import Prediction


class Detector:
    def __init__(self, config: DetectorConfig | None = None):
        self.config = config or DetectorConfig()
        backend = OnnxBackend if self.config.backend == "onnx" else TorchBackend
        self.backend = backend(self.config)
        self.names = self.backend.names

    def predict(self, image: np.ndarray) -> Prediction:
        if (
            not isinstance(image, np.ndarray)
            or image.dtype != np.uint8
            or image.ndim != 3
            or image.shape[2] != 3
            or image.size == 0
        ):
            raise ValueError("Expected a nonempty HWC BGR uint8 image.")
        start = perf_counter()
        detections = self.backend.predict(image)
        elapsed = (perf_counter() - start) * 1000
        h, w = image.shape[:2]
        return Prediction(detections, w, h, elapsed, self.config.backend)
