"""Letterbox geometry and class-aware non-maximum suppression in NumPy."""

import cv2
import numpy as np


def letterbox(image: np.ndarray, size: int) -> tuple[np.ndarray, float, tuple[int, int]]:
    h, w = image.shape[:2]
    scale = min(size / h, size / w)
    new_w, new_h = round(w * scale), round(h * scale)
    resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    left = round((size - new_w) / 2 - 0.1)
    top = round((size - new_h) / 2 - 0.1)
    padded = cv2.copyMakeBorder(
        resized,
        top,
        size - new_h - top,
        left,
        size - new_w - left,
        cv2.BORDER_CONSTANT,
        value=(114, 114, 114),
    )
    return padded, scale, (left, top)


def box_iou(box: np.ndarray, boxes: np.ndarray) -> np.ndarray:
    overlap = np.maximum(0, np.minimum(box[2:], boxes[:, 2:]) - np.maximum(box[:2], boxes[:, :2]))
    intersection = overlap[:, 0] * overlap[:, 1]
    area = np.prod(np.maximum(0, box[2:] - box[:2]))
    areas = np.prod(np.maximum(0, boxes[:, 2:] - boxes[:, :2]), axis=1)
    return intersection / np.maximum(area + areas - intersection, 1e-9)


def nms(boxes, scores, class_ids, threshold: float, max_detections: int = 300) -> list[int]:
    """Suppress overlapping boxes of the same class, highest score first."""
    order = np.argsort(-scores, kind="stable")
    keep = []
    while order.size and len(keep) < max_detections:
        index = int(order[0])
        keep.append(index)
        rest = order[1:]
        overlap = box_iou(boxes[index], boxes[rest])
        order = rest[(class_ids[rest] != class_ids[index]) | (overlap <= threshold)]
    return keep


def restore_boxes(boxes, scale, padding, width, height):
    result = boxes.copy()
    result[:, [0, 2]] = (result[:, [0, 2]] - padding[0]) / scale
    result[:, [1, 3]] = (result[:, [1, 3]] - padding[1]) / scale
    result[:, [0, 2]] = result[:, [0, 2]].clip(0, width)
    result[:, [1, 3]] = result[:, [1, 3]].clip(0, height)
    return result
