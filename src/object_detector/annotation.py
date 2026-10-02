"""Consistent, legible annotations without modifying the original image."""

import cv2
import numpy as np

from .types import Prediction

PALETTE = [(235, 99, 37), (80, 130, 10), (30, 92, 220), (160, 60, 125), (10, 120, 180)]


def annotate(
    image: np.ndarray, prediction: Prediction, *, show_confidence: bool = True
) -> np.ndarray:
    canvas = image.copy()
    h, w = canvas.shape[:2]
    font_scale = max(0.45, min(h, w) / 1000)
    thickness = max(2, round(min(h, w) / 400))
    for detection in prediction.detections:
        x1, y1, x2, y2 = [int(round(v)) for v in detection.xyxy]
        x1, x2 = np.clip([x1, x2], 0, w - 1).tolist()
        y1, y2 = np.clip([y1, y2], 0, h - 1).tolist()
        color = PALETTE[detection.class_id % len(PALETTE)]
        cv2.rectangle(canvas, (x1, y1), (x2, y2), color, thickness)
        text = detection.label
        if show_confidence:
            text += f" {detection.confidence:.2f}"
        (tw, th), baseline = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, 1)
        tx = min(x1, max(0, w - tw - 8))
        ty = y1 - th - baseline - 8 if y1 > th + baseline + 8 else y1
        cv2.rectangle(
            canvas, (tx, ty), (min(w - 1, tx + tw + 8), ty + th + baseline + 8), color, -1
        )
        cv2.putText(
            canvas,
            text,
            (tx + 4, ty + th + 4),
            cv2.FONT_HERSHEY_SIMPLEX,
            font_scale,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )
    return canvas
