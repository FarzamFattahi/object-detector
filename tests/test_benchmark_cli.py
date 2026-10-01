import numpy as np
import pytest

from object_detector.benchmark import benchmark, compare_predictions
from object_detector.cli import build_parser, image_paths
from object_detector.types import Detection, Prediction


def prediction(detections):
    return Prediction(tuple(detections), 100, 100, 1, "stub")


def test_parity_uses_unique_same_class_matches():
    a = Detection(0, "person", 0.9, (0, 0, 10, 10))
    b = Detection(1, "car", 0.8, (0, 0, 10, 10))
    assert compare_predictions(prediction([a, b]), prediction([b, a]))["all_matched"]
    assert not compare_predictions(prediction([a, a]), prediction([a]))["all_matched"]
    assert not compare_predictions(prediction([a]), prediction([b]))["all_matched"]


def test_benchmark_validates_measurement_counts():
    with pytest.raises(ValueError):
        benchmark(None, [])
    with pytest.raises(ValueError):
        benchmark(None, [np.zeros((1, 1, 3))], repeats=0)


def test_cli_parses_onnx_and_class_filter():
    args = build_parser().parse_args(
        [
            "detect",
            "--source",
            "webcam:0",
            "--backend",
            "onnx",
            "--classes",
            "0",
            "2",
            "--max-frames",
            "20",
        ]
    )
    assert args.classes == [0, 2] and args.max_frames == 20


def test_directory_selects_only_supported_files(tmp_path):
    (tmp_path / "image.JPG").touch()
    (tmp_path / "readme.txt").touch()
    (tmp_path / "folder.png").mkdir()
    assert [p.name for p in image_paths(tmp_path)] == ["image.JPG"]
