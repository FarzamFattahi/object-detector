"""Summarize PPE annotation sizes at the actual 640-pixel letterbox resolution."""

import json
from pathlib import Path

import matplotlib
import numpy as np
from PIL import Image

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from object_detector.ppe_dataset import NAMES, read_labels  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]


def main():
    report = {
        "resolution": 640,
        "size_definition": "sqrt(box area) after aspect-preserving "
        "resize to fit 640x640; small <=32 px, medium <=96 px, large >96 px",
        "splits": {},
    }
    for split in ("train", "val", "test"):
        values = [[] for _ in NAMES]
        for line in (ROOT / f"data/ppe-audited/{split}.txt").read_text().splitlines():
            path = Path(line)
            with Image.open(path) as image:
                width, height = image.size
            scale = 640 / max(width, height)
            labels = read_labels(path.parents[2] / "labels" / split / (path.stem + ".txt"))
            for cls, _, _, w, h in labels:
                values[int(cls)].append(float(np.sqrt(w * width * h * height) * scale))
        report["splits"][split] = [
            {
                "name": name,
                "instances": len(sizes),
                "median_equivalent_side_px": float(np.median(sizes)) if sizes else None,
                "small": sum(v <= 32 for v in sizes),
                "medium": sum(32 < v <= 96 for v in sizes),
                "large": sum(v > 96 for v in sizes),
            }
            for name, sizes in zip(NAMES, values, strict=True)
        ]
    output = ROOT / "assets/ppe"
    output.mkdir(exist_ok=True)
    (output / "box_sizes.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    fig, axes = plt.subplots(1, 2, figsize=(13, 5), layout="constrained", sharey=True)
    for ax, split in zip(axes, ("train", "test"), strict=True):
        left = np.zeros(len(NAMES))
        for category, color in zip(
            ("small", "medium", "large"), ("#f59e0b", "#2563eb", "#64748b"), strict=True
        ):
            fraction = [r[category] / max(r["instances"], 1) for r in report["splits"][split]]
            ax.barh(NAMES, fraction, left=left, label=category, color=color)
            left += fraction
        ax.set_title(split.capitalize())
        ax.set_xlabel("Fraction of annotations at 640 px letterbox resolution")
        ax.set_xlim(0, 1)
    axes[1].legend(loc="lower right")
    fig.suptitle("PPE detection difficulty / small boxes carry fewer visual cues", weight="bold")
    fig.savefig(output / "box_sizes.png", dpi=160)
    plt.close(fig)
    print(json.dumps(report["splits"]["test"], indent=2))


if __name__ == "__main__":
    main()
