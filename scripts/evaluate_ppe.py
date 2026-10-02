"""Evaluate the selected PPE checkpoint on test and publish measured evidence."""

import argparse
import csv
import hashlib
import json
import os
import shutil
from pathlib import Path

import cv2
import matplotlib
import numpy as np
import torch
import yaml
from PIL import Image, ImageDraw, ImageFont
from ultralytics import YOLO

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from object_detector import Detector, DetectorConfig  # noqa: E402
from object_detector.annotation import annotate  # noqa: E402
from object_detector.benchmark import environment  # noqa: E402
from object_detector.evaluation import match_boxes  # noqa: E402
from object_detector.media import read_image  # noqa: E402
from object_detector.ppe_dataset import NAMES, read_labels  # noqa: E402
from object_detector.types import Detection, Prediction  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]


def truth_prediction(path, image):
    label_path = path.parents[2] / "labels" / path.parent.name / (path.stem + ".txt")
    labels = read_labels(label_path)
    h, w = image.shape[:2]
    xywh = labels[:, 1:] * np.array([w, h, w, h])
    boxes = np.column_stack((xywh[:, :2] - xywh[:, 2:] / 2, xywh[:, :2] + xywh[:, 2:] / 2))
    rows = np.column_stack((labels[:, 0], boxes))
    detections = tuple(
        Detection(int(row[0]), NAMES[int(row[0])], 1.0, tuple(row[1:])) for row in rows
    )
    return rows, Prediction(detections, w, h, 0.0, "ground truth")


def font(size):
    try:
        return ImageFont.truetype("DejaVuSans.ttf", size)
    except OSError:
        try:
            return ImageFont.truetype("C:/Windows/Fonts/arial.ttf", size)
        except OSError:
            return ImageFont.load_default(size=size)


def gallery(records, destination):
    # Selection is disclosed: strongest, median and weakest equipment F1, two each.
    eligible = [r for r in records if r["equipment"]["tp"] + r["equipment"]["fn"] > 0]
    ranked = sorted(eligible, key=lambda r: (r["equipment"]["f1"], r["image"]))
    chosen = ranked[-2:] + ranked[len(ranked) // 2 - 1 : len(ranked) // 2 + 1] + ranked[:2]
    canvas = Image.new("RGB", (1440, 6 * 410 + 130), "#0f172a")
    draw = ImageDraw.Draw(canvas)
    draw.text((30, 20), "CONSTRUCTION PPE / HELD-OUT TEST", font=font(30), fill="white")
    draw.text(
        (30, 65),
        "Strong / median / weak equipment F1 · confidence 0.25 · matching IoU 0.50",
        font=font(19),
        fill="#cbd5e1",
    )
    sources = []
    for index, record in enumerate(chosen):
        path = Path(record["path"])
        image = read_image(path)
        _, truth = truth_prediction(path, image)
        predicted = Prediction(
            tuple(
                Detection(d["class_id"], d["label"], d["confidence"], tuple(d["xyxy"]))
                for d in record["detections"]
            ),
            image.shape[1],
            image.shape[0],
            0.0,
            "torch",
        )
        top = 130 + index * 410
        for column, frame in enumerate(
            (annotate(image, truth, show_confidence=False), annotate(image, predicted))
        ):
            panel = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            panel.thumbnail((680, 330), Image.Resampling.LANCZOS)
            x = 30 + column * 710
            canvas.paste(panel, (x + (680 - panel.width) // 2, top + 34))
            title = "DATASET ANNOTATIONS" if column == 0 else "MODEL PREDICTIONS"
            draw.text((x, top), title, font=font(19), fill="#93c5fd")
        score = record["equipment"]
        draw.text(
            (30, top + 365),
            f"{['STRONG', 'STRONG', 'MEDIAN', 'MEDIAN', 'WEAK', 'WEAK'][index]}"
            f" / {record['image']}"
            f" / equipment TP {score['tp']} · FP {score['fp']} · FN {score['fn']}"
            f" · F1 {score['f1']:.2f}",
            font=font(18),
            fill="white",
        )
        sample = destination.parent / "samples" / f"test-{index + 1}.jpg"
        sample.parent.mkdir(exist_ok=True)
        shutil.copy2(path, sample)
        sample.with_suffix(".json").write_text(
            json.dumps(truth.to_dict(), indent=2), encoding="utf-8"
        )
        sources.append(
            {
                "sample": sample.name,
                "dataset_image": record["image"],
                "sha256": hashlib.sha256(sample.read_bytes()).hexdigest(),
                "selection": "equipment F1 excludes Person and ambiguous none; "
                "images must have annotated equipment",
            }
        )
    canvas.save(destination, quality=90)
    canvas.crop((0, 0, 1440, 540)).save(destination.parent / "test_preview.jpg", quality=92)
    (destination.parent / "sample_sources.json").write_text(
        json.dumps(sources, indent=2), encoding="utf-8"
    )


def charts(report, run, output):
    plt.rcParams.update({"font.size": 11, "axes.spines.top": False, "axes.spines.right": False})
    fig, axes = plt.subplots(1, 2, figsize=(13, 5), layout="constrained")
    positions = np.arange(len(NAMES))
    for i, split in enumerate(("train", "val", "test")):
        axes[0].barh(
            positions + (i - 1) * 0.25,
            report["dataset"]["splits"][split]["instances"],
            height=0.25,
            label=split,
        )
    axes[0].set_yticks(positions, NAMES)
    axes[0].set_xlabel("Annotated instances (not images)")
    axes[0].legend()
    values = [r["mAP50_95"] for r in report["per_class"]]
    axes[1].barh(NAMES, values, color="#2563eb")
    axes[1].set_xlim(0, 1)
    axes[1].set_xlabel("Test AP @ IoU 0.50:0.95")
    for i, value in enumerate(values):
        axes[1].text(value + 0.01, i, f"{value:.3f}", va="center")
    fig.suptitle("Construction-PPE · data imbalance and held-out performance", weight="bold")
    fig.savefig(output / "class_performance.png", dpi=160)
    plt.close(fig)
    with (run / "results.csv").open(encoding="utf-8") as f:
        rows = [{k.strip(): float(v) for k, v in row.items()} for row in csv.DictReader(f)]
    fig, axes = plt.subplots(1, 2, figsize=(12, 4), layout="constrained")
    for key in ("train/box_loss", "train/cls_loss", "train/dfl_loss"):
        axes[0].plot([r["epoch"] for r in rows], [r[key] for r in rows], label=key)
    for key in ("metrics/mAP50(B)", "metrics/mAP50-95(B)"):
        axes[1].plot([r["epoch"] for r in rows], [r[key] for r in rows], label=key)
    for ax in axes:
        ax.set_xlabel("Training epoch")
        ax.legend(frameon=False)
    axes[0].set_ylabel("Training loss")
    axes[1].set_ylabel("Validation mAP (test unused)")
    fig.savefig(output / "learning_curves.png", dpi=160)
    plt.close(fig)


def person_baseline(data, device):
    """Compare the common Person class; COCO has no matching PPE categories."""
    root = ROOT / "data/ppe-person-baseline"
    images, labels = root / "images/test", root / "labels/test"
    images.mkdir(parents=True, exist_ok=True)
    labels.mkdir(parents=True, exist_ok=True)
    selected = []
    for line in (data.parent / "test.txt").read_text(encoding="utf-8").splitlines():
        source = Path(line)
        destination = images / source.name
        selected.append(str(destination.resolve()))
        if not destination.exists():
            try:
                os.link(source, destination)
            except OSError:
                shutil.copy2(source, destination)
        boxes = read_labels(
            source.parents[2] / "labels" / source.parent.name / (source.stem + ".txt")
        )
        person = boxes[boxes[:, 0] == 6]
        text = "\n".join("0 " + " ".join(f"{v:.8f}" for v in box[1:]) for box in person)
        (labels / (source.stem + ".txt")).write_text(text, encoding="utf-8")
    manifest = root / "test.txt"
    manifest.write_text("\n".join(selected) + "\n", encoding="utf-8")
    config = root / "data.yaml"
    config.write_text(
        yaml.safe_dump(
            {
                "path": str(root.resolve()),
                "train": str(manifest.resolve()),
                "val": str(manifest.resolve()),
                "test": str(manifest.resolve()),
                "names": {0: "Person"},
            }
        ),
        encoding="utf-8",
    )
    metrics = YOLO(str(ROOT / "yolo11n.pt")).val(
        data=str(config),
        split="test",
        classes=[0],
        single_cls=True,
        imgsz=640,
        batch=8,
        workers=0,
        device=device,
        conf=0.001,
        iou=0.7,
        rect=False,
        plots=False,
        project=str(ROOT / "outputs"),
        name="ppe-person-baseline",
        exist_ok=True,
    )
    return {
        "model": "COCO-pretrained YOLO11n",
        "class": "Person only",
        "images": len((data.parent / "test.txt").read_text().splitlines()),
        "mAP50": float(metrics.box.map50),
        "mAP50_95": float(metrics.box.map),
        "note": "COCO person class 0 compared against PPE Person class 6 on identical test images. "
        "Other categories excluded; this is not an 11-class baseline.",
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, default=ROOT / "runs/ppe-yolo11n")
    parser.add_argument("--data", type=Path, default=ROOT / "data/ppe-audited/data.yaml")
    parser.add_argument("--output", type=Path, default=ROOT / "assets/ppe")
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    output = args.output
    output.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(4)
    weights = args.run / "weights/best.pt"
    metrics = YOLO(str(weights)).val(
        data=str(args.data),
        split="test",
        imgsz=640,
        batch=8,
        workers=0,
        device=args.device,
        conf=0.001,
        iou=0.7,
        rect=False,
        plots=True,
        project=str(ROOT / "outputs"),
        name="ppe-test",
        exist_ok=True,
    )
    audit = json.loads((args.data.parent / "audit.json").read_text(encoding="utf-8"))
    with (args.run / "results.csv").open(encoding="utf-8") as file:
        training = [
            {key.strip(): float(value) for key, value in row.items()}
            for row in csv.DictReader(file)
        ]
    best_epoch = max(training, key=lambda row: row["metrics/mAP50-95(B)"])
    # Store portable evidence without absolute workstation paths.
    summary = {
        s: {k: v for k, v in r.items() if k != "records"} for s, r in audit["splits"].items()
    }
    report = {
        "dataset": {
            "splits": summary,
            "excluded": audit["excluded"],
            "source_url": audit["source_url"],
            "similarity_policy": audit["similarity_policy"],
        },
        "split": "test",
        "checkpoint_selection": "validation only",
        "training_summary": {
            "completed_epochs": len(training),
            "best_validation_epoch": int(best_epoch["epoch"]),
            "best_validation_mAP50": best_epoch["metrics/mAP50(B)"],
            "best_validation_mAP50_95": best_epoch["metrics/mAP50-95(B)"],
            "elapsed_seconds": training[-1]["time"],
        },
        "evaluation_environment": environment(),
        "evaluation_settings": {
            "device": args.device,
            "image_size": 640,
            "batch": 8,
            "rect": False,
            "half": False,
            "confidence": 0.001,
            "nms_iou": 0.7,
        },
        "weights_sha256": hashlib.sha256(weights.read_bytes()).hexdigest(),
        "mAP50": float(metrics.box.map50),
        "mAP50_95": float(metrics.box.map),
        "precision": float(metrics.box.mp),
        "recall": float(metrics.box.mr),
        "metric_note": "Ultralytics precision/recall use its best-F1 operating point; "
        "AP uses conf .001, NMS IoU .7. Fixed .25 counts below are separate.",
        "per_class": [],
    }
    for i, cls in enumerate(metrics.box.ap_class_index):
        precision, recall, ap50, ap = metrics.box.class_result(i)
        report["per_class"].append(
            {
                "class_id": int(cls),
                "name": NAMES[cls],
                "instances": summary["test"]["instances"][cls],
                "precision": float(precision),
                "recall": float(recall),
                "mAP50": float(ap50),
                "mAP50_95": float(ap),
            }
        )
    report["person_baseline"] = person_baseline(args.data, args.device)
    report["semantic_groups"] = {}
    for group_name, ids in (("worn_equipment", range(5)), ("missing_equipment", range(7, 11))):
        rows = [row for row in report["per_class"] if row["class_id"] in ids]
        report["semantic_groups"][group_name] = {
            "classes": [row["name"] for row in rows],
            "instances": sum(row["instances"] for row in rows),
            "mAP50": float(np.mean([row["mAP50"] for row in rows])),
            "mAP50_95": float(np.mean([row["mAP50_95"] for row in rows])),
            "note": "Unweighted mean of the same per-class AP values; descriptive subset, "
            "not a separately selected or trained model.",
        }
    detector = Detector(
        DetectorConfig(model=str(weights), device=args.device, confidence=0.25, iou=0.45)
    )
    records = []
    for line in (args.data.parent / "test.txt").read_text(encoding="utf-8").splitlines():
        path = Path(line)
        image = read_image(path)
        truth, _ = truth_prediction(path, image)
        prediction = detector.predict(image)
        preds = np.array(
            [[d.class_id, *d.xyxy, d.confidence] for d in prediction.detections]
        ).reshape(-1, 6)
        equipment_truth = truth[~np.isin(truth[:, 0], [5, 6])]
        equipment_preds = preds[~np.isin(preds[:, 0], [5, 6])]
        records.append(
            {
                "path": str(path),
                "image": path.name,
                "all_classes": match_boxes(truth, preds),
                "equipment": match_boxes(equipment_truth, equipment_preds),
                "detections": prediction.to_dict()["detections"],
            }
        )
    totals = {key: sum(r["all_classes"][key] for r in records) for key in ("tp", "fp", "fn")}
    report["fixed_operating_point"] = {
        "confidence": 0.25,
        "nms_iou": 0.45,
        "match_iou": 0.5,
        **totals,
    }
    gallery(records, output / "test_gallery.jpg")
    for record in records:
        record.pop("path")
    (output / "test_predictions.json").write_text(json.dumps(records, indent=2), encoding="utf-8")
    (output / "metrics.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    shutil.copy2(args.run / "experiment.json", output / "experiment.json")
    resolved = yaml.safe_load((args.run / "args.yaml").read_text(encoding="utf-8"))
    resolved.update(model="yolo11n.pt", data="data/ppe-audited/data.yaml", project="runs")
    resolved.pop("save_dir", None)
    (output / "resolved_train_args.yaml").write_text(
        yaml.safe_dump(resolved, sort_keys=False), encoding="utf-8"
    )
    shutil.copy2(args.run / "results.csv", output / "training.csv")
    shutil.copy2(ROOT / "data/ppe-download.json", output / "download.json")
    shutil.copy2(args.data.parent / "audit.json", output / "data_audit.json")
    for name in ("BoxPR_curve.png", "confusion_matrix_normalized.png"):
        source = ROOT / "outputs/ppe-test" / name
        if source.exists():
            shutil.copy2(source, output / name)
    charts(report, args.run, output)
    print(
        json.dumps({k: report[k] for k in ("mAP50", "mAP50_95", "fixed_operating_point")}, indent=2)
    )


if __name__ == "__main__":
    main()
