"""Command-line entry point. Run cv-detect --help for all commands."""

import argparse
import json
import shutil
import sys
from pathlib import Path

from .config import DetectorConfig
from .detector import Detector
from .media import IMAGE_SUFFIXES, process_image, process_video, read_image


def add_settings(parser):
    parser.add_argument("--model", default="yolo11n.pt")
    parser.add_argument("--backend", choices=["torch", "onnx"], default="torch")
    parser.add_argument("--device", default="cpu", help="cpu or CUDA device ID, e.g. 0")
    parser.add_argument("--image-size", type=int, default=640)
    parser.add_argument("--confidence", type=float, default=0.25)
    parser.add_argument("--iou", type=float, default=0.45)
    parser.add_argument(
        "--classes", type=int, nargs="+", help="IDs from the selected model's labels"
    )
    parser.add_argument("--max-detections", type=int, default=300)
    parser.add_argument("--threads", type=int, default=4)


def build_parser():
    parser = argparse.ArgumentParser(description="YOLO11 detection and deployment toolkit")
    commands = parser.add_subparsers(dest="command", required=True)
    detect = commands.add_parser("detect", help="Process an image, directory, video or webcam")
    add_settings(detect)
    detect.add_argument("--source", required=True, help="Local path or webcam:0")
    detect.add_argument("--output", type=Path, default=Path("outputs"), help="Output directory")
    detect.add_argument("--show", action="store_true", help="Open live OpenCV preview; Q to stop")
    detect.add_argument("--max-frames", type=int)
    export = commands.add_parser("export", help="Export static FP32 YOLO11 to ONNX without NMS")
    export.add_argument("--model", default="yolo11n.pt")
    export.add_argument("--image-size", type=int, default=640)
    export.add_argument("--output", type=Path, default=Path("models/yolo11n.onnx"))
    bench = commands.add_parser("benchmark", help="Measure warmed latency on local images")
    add_settings(bench)
    bench.add_argument("--source", type=Path, required=True)
    bench.add_argument("--repeats", type=int, default=30)
    bench.add_argument("--warmup", type=int, default=5)
    bench.add_argument("--output", type=Path, default=Path("outputs/benchmark.json"))
    evaluate = commands.add_parser("evaluate", help="Ultralytics mAP evaluation on a dataset YAML")
    evaluate.add_argument("--model", default="yolo11n.pt")
    evaluate.add_argument("--data", default="coco8.yaml")
    evaluate.add_argument("--split", choices=["train", "val", "test"], default="val")
    evaluate.add_argument("--device", default="cpu")
    evaluate.add_argument("--image-size", type=int, default=640)
    evaluate.add_argument("--output", type=Path, default=Path("outputs/evaluation"))
    return parser


def image_paths(source: Path) -> list[Path]:
    if source.is_dir():
        return sorted(
            p for p in source.iterdir() if p.suffix.lower() in IMAGE_SUFFIXES and p.is_file()
        )
    return [source]


def run(args):
    if args.command == "export":
        from ultralytics import YOLO

        DetectorConfig(image_size=args.image_size)
        if args.output.suffix.lower() != ".onnx":
            raise ValueError("Export output must end in .onnx.")
        model = YOLO(args.model, task="detect")
        if model.task != "detect":
            raise ValueError("Use a YOLO11 detection model.")
        exported = Path(
            model.export(
                format="onnx",
                imgsz=args.image_size,
                opset=17,
                dynamic=False,
                simplify=False,
                nms=False,
                batch=1,
                device="cpu",
            )
        )
        args.output.parent.mkdir(parents=True, exist_ok=True)
        if exported.resolve() != args.output.resolve():
            shutil.copy2(exported, args.output)
        return {"model": str(args.output), "image_size": args.image_size}
    if args.command == "evaluate":
        from ultralytics import YOLO

        DetectorConfig(image_size=args.image_size)
        metrics = YOLO(args.model, task="detect").val(
            data=args.data,
            split=args.split,
            imgsz=args.image_size,
            device=args.device,
            batch=1,
            workers=0,
            project=str(args.output),
            name="validation",
            exist_ok=True,
            plots=True,
        )
        report = {
            "model": args.model,
            "data": args.data,
            "split": args.split,
            "image_size": args.image_size,
            "mAP50": float(metrics.box.map50),
            "mAP50_95": float(metrics.box.map),
            "precision": float(metrics.box.mp),
            "recall": float(metrics.box.mr),
            "note": "Interpret accuracy using the dataset split and provenance. "
            "COCO8/COCO128 are smoke datasets from COCO train, not held-out accuracy.",
        }
        args.output.mkdir(parents=True, exist_ok=True)
        (args.output / "metrics.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
        return report
    config = DetectorConfig(
        model=args.model,
        backend=args.backend,
        device=args.device,
        image_size=args.image_size,
        confidence=args.confidence,
        iou=args.iou,
        classes=tuple(args.classes) if args.classes else None,
        max_detections=args.max_detections,
        threads=args.threads,
    )
    detector = Detector(config)
    if args.command == "benchmark":
        from .benchmark import benchmark, environment

        images = [read_image(p) for p in image_paths(args.source)]
        report = {
            "environment": environment(),
            "measurement": benchmark(detector, images, args.repeats, args.warmup),
        }
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
        return report
    args.output.mkdir(parents=True, exist_ok=True)
    if args.source.startswith("webcam:"):
        return process_video(
            detector,
            int(args.source.split(":", 1)[1]),
            args.output / "webcam.mp4",
            max_frames=args.max_frames,
            show=args.show,
        )
    source = Path(args.source)
    if not source.exists():
        raise FileNotFoundError(f"Source not found: {source}")
    if source.is_dir() or source.suffix.lower() in IMAGE_SUFFIXES:
        paths = image_paths(source)
        if not paths:
            raise ValueError("Directory contains no supported images (nonrecursive).")
        return {
            "images": [
                process_image(detector, p, args.output / f"{p.name}_detected.png") for p in paths
            ]
        }
    summary = process_video(
        detector,
        str(source),
        args.output / f"{source.stem}_detected.mp4",
        max_frames=args.max_frames,
        show=args.show,
    )
    (args.output / f"{source.stem}_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    return summary


def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        print(json.dumps(run(args), indent=2))
    except (ValueError, FileNotFoundError, OSError, RuntimeError) as error:
        print(f"Error: {error}", file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        print("Stopped; video and record handles released.", file=sys.stderr)
        return 130
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
