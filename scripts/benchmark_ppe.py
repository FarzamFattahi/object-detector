"""Measure PPE deployment latency and independent ONNX agreement on real test inputs."""

import argparse
import json
from dataclasses import replace
from pathlib import Path

from object_detector import Detector, DetectorConfig
from object_detector.benchmark import benchmark, compare_predictions, environment
from object_detector.media import read_image

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--output", type=Path, default=ROOT / "assets/ppe/deployment.json")
    args = parser.parse_args()
    config = DetectorConfig(model=str(ROOT / "models/ppe-yolo11n.pt"), device=args.device)
    torch_detector = Detector(config)
    onnx_detector = Detector(
        replace(config, model=str(ROOT / "models/ppe-yolo11n.onnx"), backend="onnx", device="cpu")
    )
    paths = [Path(line) for line in (ROOT / "data/ppe-audited/test.txt").read_text().splitlines()][
        :20
    ]
    images = [read_image(p) for p in paths]
    parity = [
        {
            "image": path.name,
            **compare_predictions(torch_detector.predict(image), onnx_detector.predict(image)),
        }
        for path, image in zip(paths, images, strict=True)
    ]
    report = {
        "environment": environment(),
        "selection": "First 20 sorted test filenames",
        "parity": parity,
        "torch": benchmark(torch_detector, images, repeats=40, warmup=5),
        "onnx_cpu": benchmark(onnx_detector, images, repeats=40, warmup=5),
    }
    # Keep published paths portable.
    report["torch"]["config"]["model"] = "models/ppe-yolo11n.pt"
    report["onnx_cpu"]["config"]["model"] = "models/ppe-yolo11n.onnx"
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    if not all(row["all_matched"] for row in parity):
        raise RuntimeError("Backend differences found; inspect the published parity report")
    print(
        json.dumps(
            {
                "matched_detections": sum(p["matched_count"] for p in parity),
                "torch_median_ms": report["torch"]["median_ms"],
                "onnx_median_ms": report["onnx_cpu"]["median_ms"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
