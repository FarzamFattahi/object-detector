"""Warmed batch-one latency and one-to-one detection parity measurements."""

import importlib.metadata
import platform
from dataclasses import asdict
from datetime import datetime, timezone
from time import perf_counter

import numpy as np

from .detector import Detector
from .geometry import box_iou
from .types import Prediction


def compare_predictions(reference: Prediction, candidate: Prediction, min_iou=0.99) -> dict:
    unmatched = set(range(len(candidate.detections)))
    pairs = []
    for detection in reference.detections:
        options = [i for i in unmatched if candidate.detections[i].class_id == detection.class_id]
        if not options:
            continue
        overlaps = box_iou(
            np.array(detection.xyxy), np.array([candidate.detections[i].xyxy for i in options])
        )
        position = int(np.argmax(overlaps))
        if overlaps[position] >= min_iou:
            index = options[position]
            unmatched.remove(index)
            other = candidate.detections[index]
            pairs.append(
                (
                    float(overlaps[position]),
                    abs(detection.confidence - other.confidence),
                    max(abs(a - b) for a, b in zip(detection.xyxy, other.xyxy, strict=True)),
                )
            )
    return {
        "reference_count": len(reference.detections),
        "candidate_count": len(candidate.detections),
        "matched_count": len(pairs),
        "all_matched": len(pairs) == len(reference.detections) == len(candidate.detections),
        "min_iou": min((p[0] for p in pairs), default=None),
        "max_confidence_difference": max((p[1] for p in pairs), default=None),
        "max_coordinate_difference_px": max((p[2] for p in pairs), default=None),
    }


def benchmark(detector: Detector, images: list[np.ndarray], repeats=30, warmup=5) -> dict:
    if not images or repeats < 1 or warmup < 1:
        raise ValueError("Provide images and positive repeats/warmup counts.")
    for i in range(warmup):
        detector.predict(images[i % len(images)])
    timings = []
    for i in range(repeats):
        start = perf_counter()
        detector.predict(images[i % len(images)])
        timings.append((perf_counter() - start) * 1000)
    return {
        "config": asdict(detector.config),
        "warmup": warmup,
        "repeats": repeats,
        "scope": "BGR array to detections: preprocessing + inference + NMS; excludes I/O/drawing",
        "mean_ms": float(np.mean(timings)),
        "median_ms": float(np.median(timings)),
        "p95_ms": float(np.percentile(timings, 95)),
        "prediction_fps": 1000 / np.mean(timings),
        "samples_ms": timings,
    }


def environment() -> dict:
    return {
        "measured_at_utc": datetime.now(timezone.utc).isoformat(),
        "os": platform.platform(),
        "cpu": platform.processor(),
        "python": platform.python_version(),
        "packages": {
            name: importlib.metadata.version(name)
            for name in ("torch", "ultralytics", "onnxruntime", "opencv-python", "numpy")
        },
    }
