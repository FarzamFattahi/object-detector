import numpy as np
import pytest

from object_detector.geometry import box_iou, letterbox, nms, restore_boxes


@pytest.mark.parametrize("shape", [(100, 200), (201, 99), (99, 201), (640, 640), (1, 300)])
def test_letterbox_and_coordinate_roundtrip(shape):
    h, w = shape
    image = np.zeros((h, w, 3), dtype=np.uint8)
    padded, scale, padding = letterbox(image, 640)
    assert padded.shape == (640, 640, 3)
    original = np.array([[0, 0, w, h], [w * 0.1, h * 0.2, w * 0.7, h * 0.9]], dtype=np.float32)
    transformed = original * scale + np.array([*padding, *padding])
    np.testing.assert_allclose(
        restore_boxes(transformed, scale, padding, w, h), original, atol=1e-4
    )


def test_restore_clips_to_original_bounds():
    boxes = np.array([[-20.0, -20.0, 300.0, 200.0]])
    np.testing.assert_equal(restore_boxes(boxes, 1, (0, 0), 100, 50), [[0, 0, 100, 50]])


def test_nms_is_class_aware_and_sorted():
    boxes = np.array([[0, 0, 10, 10], [1, 1, 11, 11], [0, 0, 10, 10], [20, 20, 30, 30]])
    scores = np.array([0.9, 0.8, 0.7, 0.95])
    classes = np.array([0, 0, 1, 0])
    assert nms(boxes, scores, classes, 0.5) == [3, 0, 2]
    assert nms(boxes, scores, classes, 0.5, 2) == [3, 0]


def test_empty_nms_and_zero_area_iou():
    assert nms(np.empty((0, 4)), np.array([]), np.array([]), 0.5) == []
    assert box_iou(np.zeros(4), np.zeros((2, 4))).tolist() == [0, 0]
