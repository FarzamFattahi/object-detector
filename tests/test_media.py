import json

import cv2
import numpy as np
import pytest

from object_detector.annotation import annotate
from object_detector.media import process_image, process_video, read_image, write_image
from object_detector.types import Detection, Prediction


class StubDetector:
    def predict(self, image):
        h, w = image.shape[:2]
        return Prediction((Detection(0, "person", 0.9, (5, 5, w - 5, h - 5)),), w, h, 1, "stub")


def test_unicode_image_io_annotation_and_json(tmp_path):
    image = np.zeros((100, 200, 3), dtype=np.uint8)
    source = tmp_path / "تصویر.png"
    write_image(source, image)
    np.testing.assert_equal(read_image(source), image)
    output = tmp_path / "result.png"
    record = process_image(StubDetector(), source, output)
    assert read_image(output).any()
    assert json.loads(output.with_suffix(".json").read_text())["source"] == source.name
    assert record["detections"][0]["label"] == "person"
    assert not image.any(), "Annotation must not mutate the input"
    with pytest.raises(ValueError, match="differ"):
        process_image(StubDetector(), source, source)


def test_video_roundtrip_frame_limit_timestamps_and_cleanup(tmp_path):
    source = tmp_path / "source.avi"
    writer = cv2.VideoWriter(str(source), cv2.VideoWriter_fourcc(*"MJPG"), 10, (160, 100))
    assert writer.isOpened()
    for i in range(5):
        writer.write(np.full((100, 160, 3), i * 30, dtype=np.uint8))
    writer.release()
    output = tmp_path / "result.mp4"
    progress = []
    summary = process_video(
        StubDetector(),
        str(source),
        output,
        max_frames=3,
        on_progress=lambda n, total: progress.append((n, total)),
    )
    assert summary["frames"] == 3 and summary["source_fps"] == 10
    records = [json.loads(line) for line in output.with_suffix(".jsonl").read_text().splitlines()]
    assert [r["frame_index"] for r in records] == [0, 1, 2]
    assert [r["timestamp_seconds"] for r in records] == [0, 0.1, 0.2]
    assert progress == [(1, 3), (2, 3), (3, 3)]
    capture = cv2.VideoCapture(str(output))
    decoded = 0
    while capture.read()[0]:
        decoded += 1
    capture.release()
    assert decoded == 3
    output.rename(tmp_path / "closed.mp4")


def test_video_capture_released_on_inference_failure(monkeypatch):
    class Capture:
        released = False

        def isOpened(self):
            return True

        def get(self, prop):
            return 30

        def read(self):
            return True, np.zeros((100, 100, 3), dtype=np.uint8)

        def release(self):
            self.released = True

    class BrokenDetector:
        def predict(self, image):
            raise RuntimeError("inference failed")

    capture = Capture()
    monkeypatch.setattr(cv2, "VideoCapture", lambda _: capture)
    with pytest.raises(RuntimeError):
        process_video(BrokenDetector(), "test.avi")
    assert capture.released


def test_corrupt_input_and_invalid_video_options(tmp_path):
    broken = tmp_path / "broken.png"
    broken.write_bytes(b"not an image")
    with pytest.raises(ValueError, match="decode"):
        read_image(broken)
    with pytest.raises(FileNotFoundError):
        read_image(tmp_path / "missing.png")
    with pytest.raises(ValueError, match="positive"):
        process_video(StubDetector(), "missing.avi", max_frames=0)
    with pytest.raises(ValueError, match="differ"):
        process_video(StubDetector(), str(broken), broken)
    with pytest.raises(ValueError, match=".mp4 or .avi"):
        process_video(StubDetector(), "input.avi", tmp_path / "out.gif")


def test_empty_annotation_is_identity():
    image = np.zeros((20, 20, 3), dtype=np.uint8)
    result = annotate(image, Prediction((), 20, 20, 1, "stub"))
    np.testing.assert_equal(image, result)
    assert result is not image
