"""Framework-independent prediction records in original image coordinates."""

from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class Detection:
    class_id: int
    label: str
    confidence: float
    xyxy: tuple[float, float, float, float]


@dataclass(frozen=True)
class Prediction:
    detections: tuple[Detection, ...]
    width: int
    height: int
    latency_ms: float
    backend: str

    def to_dict(self) -> dict:
        return asdict(self)
