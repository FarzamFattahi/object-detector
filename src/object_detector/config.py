"""Validated settings shared by the CLI, dashboard and both backends."""

import math
from dataclasses import dataclass


@dataclass(frozen=True)
class DetectorConfig:
    model: str = "yolo11n.pt"
    backend: str = "torch"
    device: str = "cpu"
    image_size: int = 640
    confidence: float = 0.25
    iou: float = 0.45
    classes: tuple[int, ...] | None = None
    max_detections: int = 300
    threads: int = 4

    def __post_init__(self):
        if self.backend not in {"torch", "onnx"}:
            raise ValueError("Backend must be torch or onnx.")
        if not self.model.strip():
            raise ValueError("Model path cannot be empty.")
        if self.image_size < 32 or self.image_size % 32:
            raise ValueError("Image size must be a positive multiple of 32.")
        for name in ("confidence", "iou"):
            value = getattr(self, name)
            if not math.isfinite(value) or not 0 <= value <= 1:
                raise ValueError(f"{name} must be between 0 and 1.")
        if self.max_detections < 1 or self.threads < 1:
            raise ValueError("max_detections and threads must be positive.")
        if self.classes is not None and (not self.classes or any(c < 0 for c in self.classes)):
            raise ValueError("Class IDs must be nonnegative; omit the filter to select all.")
        if self.backend == "onnx" and self.device != "cpu":
            raise ValueError("This ONNX backend uses CPUExecutionProvider; use --device cpu.")
