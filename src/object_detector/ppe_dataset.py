"""Validate YOLO labels and build disjoint manifests without modifying upstream data."""

import hashlib
import json
from pathlib import Path

import cv2
import numpy as np
import yaml

NAMES = [
    "helmet",
    "gloves",
    "vest",
    "boots",
    "goggles",
    "none",
    "Person",
    "no_helmet",
    "no_goggle",
    "no_gloves",
    "no_boots",
]
DATASET_URL = "https://github.com/ultralytics/assets/releases/download/v0.0.0/construction-ppe.zip"
DATASET_SHA256 = "bef8dcb599aa4e9d9f5e602cb6fa7143d3c84d7f6a0ff40463d7f2a4c2632ccc"


def read_labels(path: Path) -> np.ndarray:
    """Reject malformed annotations rather than silently dropping them during training."""
    rows = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            row = [float(value) for value in line.split()]
        except ValueError as error:
            raise ValueError(f"{path}:{line_number}: nonnumeric label") from error
        if len(row) != 5 or not np.isfinite(row).all():
            raise ValueError(f"{path}:{line_number}: expected five finite values")
        cls, x, y, width, height = row
        if cls != int(cls) or not 0 <= cls < len(NAMES):
            raise ValueError(f"{path}:{line_number}: invalid class ID")
        if not (0 <= x <= 1 and 0 <= y <= 1 and 0 < width <= 1 and 0 < height <= 1):
            raise ValueError(f"{path}:{line_number}: invalid normalized box")
        # Some source boxes touch edges with rounded coordinates; allow 0.1% rounding.
        if (
            min(x - width / 2, y - height / 2) < -0.001
            or max(x + width / 2, y + height / 2) > 1.001
        ):
            raise ValueError(f"{path}:{line_number}: box extends outside image")
        rows.append(row)
    return np.array(rows, dtype=np.float32).reshape(-1, 5)


def perceptual_hash(image: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    coefficients = cv2.dct(cv2.resize(gray, (32, 32)).astype(np.float32))[:8, :8].ravel()
    return np.packbits(coefficients > np.median(coefficients[1:]))


def similarity_groups(hashes: np.ndarray, distance: int) -> list[int]:
    """Connected components of 64-bit DCT pHash similarity (Hamming distance)."""
    lookup = np.array([bin(i).count("1") for i in range(256)], dtype=np.uint8)
    parents = list(range(len(hashes)))

    def find(index):
        while parents[index] != index:
            parents[index] = parents[parents[index]]
            index = parents[index]
        return index

    # Compare one row at a time to avoid an N x N x 8 memory allocation.
    for i, value in enumerate(hashes):
        distances = lookup[np.bitwise_xor(value, hashes[i + 1 :])].sum(axis=1)
        for j in np.flatnonzero(distances <= distance) + i + 1:
            parents[find(int(j))] = find(i)
    return [find(i) for i in range(len(hashes))]


def prepare_dataset(root: Path, output: Path, near_distance: int = 8) -> dict:
    """Keep test, then val, then train when decoded pixels are duplicated.

    Exact duplicates are removed within/across splits. Near-similar connected
    groups are retained only in their highest-priority split, preserving test.
    pHash is a heuristic, not a guarantee of site/subject independence.
    """
    root, output = root.resolve(), output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    seen = {}
    hashes, all_records = [], []
    report = {"source_url": DATASET_URL, "names": NAMES, "splits": {}, "excluded": []}
    for split in ("test", "val", "train"):
        files = sorted((root / "images" / split).glob("*"))
        files = [p for p in files if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp"}]
        if not files:
            raise ValueError(f"Empty or missing split: {split}")
        counts = np.zeros(len(NAMES), dtype=int)
        retained = []
        records = []
        for path in files:
            label = root / "labels" / split / (path.stem + ".txt")
            boxes = read_labels(label)
            image = cv2.imdecode(np.fromfile(path, dtype=np.uint8), cv2.IMREAD_COLOR)
            if image is None:
                raise ValueError(f"Unreadable image: {path}")
            digest = hashlib.sha256(str(image.shape).encode() + image.tobytes()).hexdigest()
            relative = path.relative_to(root).as_posix()
            if digest in seen:
                report["excluded"].append({"image": relative, "duplicate_of": seen[digest]})
                continue
            seen[digest] = relative
            hashes.append(perceptual_hash(image))
            retained.append(str(path))
            counts += np.bincount(boxes[:, 0].astype(int), minlength=len(NAMES))
            records.append(
                {
                    "image": relative,
                    "pixel_sha256": digest,
                    "label_sha256": hashlib.sha256(label.read_bytes()).hexdigest(),
                }
            )
            all_records.append((split, records[-1]))
        if not retained:
            raise ValueError(f"No distinct images remain in {split}")
        (output / f"{split}.txt").write_text("\n".join(retained) + "\n", encoding="utf-8")
        report["splits"][split] = {
            "original_images": len(files),
            "images": len(retained),
            "instances": counts.tolist(),
            "records": records,
        }
    groups = similarity_groups(np.array(hashes), near_distance)
    priority = {"test": 0, "val": 1, "train": 2}
    group_splits = {}
    for group, (split, _) in zip(groups, all_records, strict=True):
        group_splits.setdefault(group, set()).add(split)
    winners = {g: min(splits, key=priority.get) for g, splits in group_splits.items()}
    filtered = {s: [] for s in priority}
    for group, (split, record) in zip(groups, all_records, strict=True):
        record["similarity_group"] = group
        if split == winners[group]:
            filtered[split].append(record)
        else:
            report["excluded"].append(
                {
                    "image": record["image"],
                    "similarity_group": group,
                    "retained_split": winners[group],
                    "reason": "pHash group",
                }
            )
    for split, records in filtered.items():
        if not records:
            raise ValueError(f"No images remain after similarity filtering: {split}")
        counts = np.zeros(len(NAMES), dtype=int)
        for record in records:
            label = root / record["image"].replace("images/", "labels/")
            boxes = read_labels(label.with_suffix(".txt"))
            counts += np.bincount(boxes[:, 0].astype(int), minlength=len(NAMES))
        report["splits"][split].update(
            images=len(records),
            instances=counts.tolist(),
            records=records,
            similarity_groups=len({r["similarity_group"] for r in records}),
        )
        (output / f"{split}.txt").write_text(
            "\n".join(str(root / r["image"]) for r in records) + "\n", encoding="utf-8"
        )
    report["similarity_policy"] = {
        "method": "64-bit DCT pHash connected components",
        "max_hamming_distance": near_distance,
        "priority": ["test", "val", "train"],
        "limitation": "May miss related scenes or group unrelated images; no site IDs available. "
        "Related images within each retained split remain. "
        "This is not site-independent evaluation.",
    }
    data = {
        "path": str(root),
        **{s: str(output / f"{s}.txt") for s in ("train", "val", "test")},
        "names": dict(enumerate(NAMES)),
    }
    (output / "data.yaml").write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")
    (output / "audit.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report
