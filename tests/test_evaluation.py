import numpy as np

from object_detector.evaluation import match_boxes


def test_duplicate_and_wrong_class_detections_count_as_false_positives():
    truth = np.array([[0, 0, 0, 20, 20], [1, 30, 30, 50, 50]])
    predictions = np.array(
        [[0, 0, 0, 20, 20, 0.9], [0, 0, 0, 20, 20, 0.8], [0, 30, 30, 50, 50, 0.7]]
    )
    result = match_boxes(truth, predictions)
    assert (result["tp"], result["fp"], result["fn"]) == (1, 2, 1)
    assert result["matched_predictions"] == [0]


def test_no_predictions_and_no_truth_are_well_defined():
    empty_truth, empty_predictions = np.empty((0, 5)), np.empty((0, 6))
    assert match_boxes(empty_truth, empty_predictions)["f1"] == 0
    truth = np.array([[0, 0, 0, 20, 20]])
    assert match_boxes(truth, empty_predictions)["fn"] == 1
    assert match_boxes(empty_truth, np.array([[0, 0, 0, 20, 20, 0.9]]))["fp"] == 1
