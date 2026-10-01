"""Image I/O and constant-memory frame-by-frame video processing."""

import json
import math
from collections.abc import Callable
from pathlib import Path
from time import perf_counter

import cv2
import numpy as np

from .annotation import annotate
from .detector import Detector

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff"}


def read_image(path: str | Path) -> np.ndarray:
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Image not found: {path}")
    image = cv2.imdecode(np.fromfile(path, dtype=np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"Cannot decode image: {path}")
    return image


def write_image(path: str | Path, image: np.ndarray):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    ok, encoded = cv2.imencode(path.suffix, image)
    if not ok:
        raise OSError(f"Cannot encode image: {path}")
    encoded.tofile(path)


def process_image(detector: Detector, source: Path, output: Path) -> dict:
    if source.resolve() == output.resolve():
        raise ValueError("Output must differ from the source image.")
    image = read_image(source)
    prediction = detector.predict(image)
    write_image(output, annotate(image, prediction))
    record = {"source": source.name, **prediction.to_dict()}
    output.with_suffix(".json").write_text(json.dumps(record, indent=2), encoding="utf-8")
    return record


def process_video(
    detector: Detector,
    source: str | int,
    output: Path | None = None,
    *,
    max_frames: int | None = None,
    show: bool = False,
    on_progress: Callable[[int, int], None] | None = None,
) -> dict:
    """Write silent mp4v/AVI plus JSONL; retain source FPS and release handles on errors.

    Each JSONL row contains a frame index, source timestamp and original-coordinate boxes.
    A webcam is accessed only when the caller explicitly supplies an integer source.
    """
    if max_frames is not None and max_frames < 1:
        raise ValueError("max_frames must be positive.")
    if output and isinstance(source, str) and Path(source).resolve() == output.resolve():
        raise ValueError("Output must differ from the source video.")
    if output and output.suffix.lower() not in {".mp4", ".avi"}:
        raise ValueError("Video output must end in .mp4 or .avi.")
    capture = cv2.VideoCapture(source)
    writer = None
    records = None
    frames = 0
    total_latency = 0.0
    total_detections = 0
    start = perf_counter()
    try:
        if not capture.isOpened():
            raise ValueError(f"Cannot open video/camera: {source}")
        fps = capture.get(cv2.CAP_PROP_FPS)
        if not math.isfinite(fps) or fps <= 0:
            fps = 30.0
        total = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        if output:
            output.parent.mkdir(parents=True, exist_ok=True)
            records = output.with_suffix(".jsonl").open("w", encoding="utf-8")
        while max_frames is None or frames < max_frames:
            ok, frame = capture.read()
            if not ok:
                break
            prediction = detector.predict(frame)
            annotated = annotate(frame, prediction)
            if output:
                if writer is None:
                    h, w = frame.shape[:2]
                    codec = "mp4v" if output.suffix.lower() == ".mp4" else "MJPG"
                    writer = cv2.VideoWriter(
                        str(output), cv2.VideoWriter_fourcc(*codec), fps, (w, h)
                    )
                    if not writer.isOpened():
                        raise OSError("Video encoder unavailable; try an .avi output.")
                writer.write(annotated)
                record = {
                    "frame_index": frames,
                    "timestamp_seconds": frames / fps,
                    **prediction.to_dict(),
                }
                records.write(json.dumps(record) + "\n")
            frames += 1
            total_latency += prediction.latency_ms
            total_detections += len(prediction.detections)
            if on_progress:
                on_progress(frames, min(total, max_frames) if max_frames and total > 0 else total)
            if show:
                cv2.imshow("Object Detector - Q to stop", annotated)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
        if frames == 0:
            raise ValueError("Video opened but no frames could be decoded.")
    finally:
        capture.release()
        if writer is not None:
            writer.release()
        if records is not None:
            records.close()
        if show:
            cv2.destroyAllWindows()
    elapsed = perf_counter() - start
    return {
        "frames": frames,
        "source_fps": fps,
        "elapsed_seconds": elapsed,
        "processing_fps": frames / elapsed,
        "mean_prediction_ms": total_latency / frames,
        "detections_total": total_detections,
        "audio_preserved": False,
    }
