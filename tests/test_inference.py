import sys
from types import SimpleNamespace

import numpy as np
import pytest

from object_detector import Detector, DetectorConfig
from object_detector.backends import OnnxBackend
from object_detector.types import Detection, Prediction


@pytest.mark.parametrize(
    "settings",
    [
        {"confidence": -1},
        {"confidence": float("nan")},
        {"iou": 2},
        {"image_size": 639},
        {"image_size": 0},
        {"backend": "unknown"},
        {"threads": 0},
        {"classes": ()},
        {"classes": (-1,)},
        {"backend": "onnx", "device": "0"},
        {"model": ""},
    ],
)
def test_invalid_config(settings):
    with pytest.raises(ValueError):
        DetectorConfig(**settings)


class FakeSession:
    def __init__(self, output):
        self.output = output
        self.tensor = None

    def run(self, _, inputs):
        self.tensor = inputs["images"]
        return [self.output]


def make_backend(classes=None, confidence=0.25):
    backend = OnnxBackend.__new__(OnnxBackend)
    backend.names = {0: "person", 1: "car"}
    backend.config = DetectorConfig(
        backend="onnx", model="test.onnx", image_size=640, confidence=confidence, classes=classes
    )
    backend.input = SimpleNamespace(name="images")
    rows = np.array(
        [
            [320, 320, 320, 160, 0.9, 0.1],
            [321, 321, 320, 160, 0.8, 0.1],
            [320, 320, 320, 160, 0.1, 0.7],
        ],
        dtype=np.float32,
    )
    backend.session = FakeSession(rows.T[None])
    return backend


def test_raw_onnx_decoding_preprocessing_and_class_aware_nms():
    image = np.zeros((100, 200, 3), dtype=np.uint8)
    image[:] = [10, 20, 30]
    backend = make_backend()
    results = backend.predict(image)
    assert [r.label for r in results] == ["person", "car"]
    np.testing.assert_allclose(results[0].xyxy, [50, 25, 150, 75])
    tensor = backend.session.tensor
    assert tensor.shape == (1, 3, 640, 640)
    assert tensor.dtype == np.float32 and tensor.flags.c_contiguous
    np.testing.assert_allclose(tensor[0, :, 320, 320], np.array([30, 20, 10]) / 255)
    np.testing.assert_allclose(tensor[0, :, 0, 0], 114 / 255)


def test_onnx_class_filter_and_empty_output():
    image = np.zeros((100, 200, 3), dtype=np.uint8)
    assert [r.class_id for r in make_backend(classes=(1,)).predict(image)] == [1]
    assert make_backend(confidence=0.99).predict(image) == ()


def test_rejects_wrong_export_shape():
    backend = make_backend()
    backend.session.output = np.zeros((1, 300, 6))
    with pytest.raises(ValueError, match="raw YOLO11"):
        backend.predict(np.zeros((100, 200, 3), dtype=np.uint8))


def test_constructor_validates_model_metadata_and_shape(monkeypatch, tmp_path):
    path = tmp_path / "test.onnx"
    path.write_bytes(b"fake")
    session = SimpleNamespace(
        get_inputs=lambda: [SimpleNamespace(type="tensor(float)", shape=[1, 3, 640, 640])],
        get_modelmeta=lambda: SimpleNamespace(custom_metadata_map={"names": "{0: 'person'}"}),
    )
    monkeypatch.setitem(
        sys.modules,
        "onnxruntime",
        SimpleNamespace(
            SessionOptions=SimpleNamespace,
            InferenceSession=lambda *a, **kw: session,
        ),
    )
    backend = OnnxBackend(DetectorConfig(model=str(path), backend="onnx"))
    assert backend.names == {0: "person"}
    with pytest.raises(ValueError, match="Class filter"):
        OnnxBackend(DetectorConfig(model=str(path), backend="onnx", classes=(1,)))
    session.get_inputs = lambda: [SimpleNamespace(type="tensor(float)", shape=[1, 3, 320, 320])]
    with pytest.raises(ValueError, match="static FP32"):
        OnnxBackend(DetectorConfig(model=str(path), backend="onnx"))


@pytest.mark.parametrize(
    "image", [None, np.zeros((2, 2)), np.zeros((2, 2, 3)), np.zeros((0, 2, 3), dtype=np.uint8)]
)
def test_detector_rejects_invalid_arrays_without_inference(image):
    detector = Detector.__new__(Detector)
    with pytest.raises(ValueError, match="BGR uint8"):
        detector.predict(image)


def test_prediction_serialization():
    record = Prediction((Detection(0, "person", 0.9, (1, 2, 3, 4)),), 100, 50, 5, "onnx").to_dict()
    assert record["detections"][0]["xyxy"] == (1, 2, 3, 4)


def test_ppe_output_uses_eleven_class_scores_and_person_id_six():
    from object_detector.ppe_dataset import NAMES

    backend = make_backend(classes=(6,))
    backend.names = dict(enumerate(NAMES))
    output = np.zeros((1, 15, 2), dtype=np.float32)
    output[0, :4, :] = np.array([[100, 400], [100, 400], [50, 80], [50, 160]])
    output[0, 4, 0] = 0.95  # helmet
    output[0, 4 + 6, 1] = 0.85  # Person
    backend.session.output = output
    results = backend.predict(np.zeros((640, 640, 3), dtype=np.uint8))
    assert len(results) == 1
    assert (results[0].class_id, results[0].label) == (6, "Person")
    np.testing.assert_allclose(results[0].xyxy, [360, 320, 440, 480])
