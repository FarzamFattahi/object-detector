import json

import cv2
import numpy as np
import pytest
import yaml

from object_detector.ppe_dataset import prepare_dataset, read_labels, similarity_groups


def fixture_image(root, split, name, pixel, label="0 0.5 0.5 0.4 0.4\n"):
    images, labels = root / "images" / split, root / "labels" / split
    images.mkdir(parents=True, exist_ok=True)
    labels.mkdir(parents=True, exist_ok=True)
    image = np.random.default_rng(pixel).integers(0, 180, (64, 64, 3), dtype=np.uint8)
    cv2.imwrite(str(images / f"{name}.png"), image)
    (labels / f"{name}.txt").write_text(label, encoding="utf-8")


def test_test_pixels_cannot_leak_into_training_even_under_another_filename(tmp_path):
    root, output = tmp_path / "raw", tmp_path / "audited"
    fixture_image(root, "test", "heldout", 30)
    fixture_image(root, "val", "validation", 60)
    fixture_image(root, "train", "training", 90)
    fixture_image(root, "train", "renamed_test", 30)
    fixture_image(root, "train", "renamed_val", 60)
    fixture_image(root, "train", "repeat_training", 90)
    report = prepare_dataset(root, output)
    assert len(report["excluded"]) == 3
    assert all(report["splits"][s]["images"] == 1 for s in ("train", "val", "test"))
    manifests = [
        set((output / f"{s}.txt").read_text().splitlines()) for s in ("train", "val", "test")
    ]
    assert not (manifests[0] & manifests[1] | manifests[0] & manifests[2])
    assert json.loads((output / "audit.json").read_text())["excluded"] == report["excluded"]
    assert len(yaml.safe_load((output / "data.yaml").read_text())["names"]) == 11


@pytest.mark.parametrize(
    "row",
    [
        "11 .5 .5 .2 .2",
        "0.5 .5 .5 .2 .2",
        "0 nan .5 .2 .2",
        "0 .5 .5 0 .2",
        "0 .95 .5 .3 .2",
        "0 .5 .5 .2",
        "oops",
    ],
)
def test_bad_labels_fail_loudly(tmp_path, row):
    path = tmp_path / "bad.txt"
    path.write_text(row)
    with pytest.raises(ValueError):
        read_labels(path)


def test_negative_image_and_edge_rounding_are_valid(tmp_path):
    path = tmp_path / "empty.txt"
    path.write_text("")
    assert read_labels(path).shape == (0, 5)
    path.write_text("10 .1 .5 .2001 1")
    assert read_labels(path)[0, 0] == 10


def test_brightness_variant_of_test_image_is_excluded_from_train(tmp_path):
    root, output = tmp_path / "raw", tmp_path / "audited"
    for split, seed in [("test", 30), ("val", 60), ("train", 90)]:
        fixture_image(root, split, split, seed)
    fixture_image(root, "train", "variant", 120)
    original = cv2.imread(str(root / "images/test/test.png"))
    cv2.imwrite(str(root / "images/train/variant.png"), original + 30)
    report = prepare_dataset(root, output)
    assert report["splits"]["train"]["images"] == 1
    assert report["excluded"][0]["reason"] == "pHash group"
    assert report["excluded"][0]["retained_split"] == "test"


def test_related_frame_chains_form_one_group_even_when_endpoints_differ():
    hashes = np.zeros((4, 8), dtype=np.uint8)
    hashes[1, 0], hashes[2, 0], hashes[3, :] = 15, 255, 255
    groups = similarity_groups(hashes, 4)
    assert groups[0] == groups[1] == groups[2]
    assert groups[3] != groups[0]
