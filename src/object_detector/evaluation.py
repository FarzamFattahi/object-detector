"""Confidence-ordered, one-to-one matching for readable operating-point error counts."""

import numpy as np

from .geometry import box_iou


def match_boxes(truth: np.ndarray, predictions: np.ndarray, iou_threshold=0.5) -> dict:
    """Rows are class,x1,y1,x2,y2[,confidence]; predictions require confidence.

    This computes TP/FP/FN at a fixed threshold, not interpolated AP. Ultralytics
    supplies the standard multi-IoU AP calculation in the case-study report.
    """
    matched_truth, matched_predictions = set(), set()
    for index in np.argsort(-predictions[:, 5], kind="stable"):
        candidates = [
            i
            for i in range(len(truth))
            if i not in matched_truth and truth[i, 0] == predictions[index, 0]
        ]
        if not candidates:
            continue
        overlaps = box_iou(predictions[index, 1:5], truth[candidates, 1:5])
        best = int(np.argmax(overlaps))
        if overlaps[best] >= iou_threshold:
            matched_truth.add(candidates[best])
            matched_predictions.add(int(index))
    tp = len(matched_truth)
    fp, fn = len(predictions) - tp, len(truth) - tp
    return {
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "precision": tp / (tp + fp) if tp + fp else 0.0,
        "recall": tp / (tp + fn) if tp + fn else 0.0,
        "f1": 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 0.0,
        "matched_truth": sorted(matched_truth),
        "matched_predictions": sorted(matched_predictions),
    }
