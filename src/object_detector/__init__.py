"""Computer vision inference and deployment tools."""

from .config import DetectorConfig
from .detector import Detector
from .types import Detection, Prediction

__all__ = ["Detection", "Detector", "DetectorConfig", "Prediction"]
__version__ = "1.0.0"
