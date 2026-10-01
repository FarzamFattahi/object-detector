"""Reproduce the actual inference gallery, ONNX parity report, latency plot and GIF.

Run download_samples.py --video and cv-detect export before this script.
No simulated detections or invented measurements are used.
"""

import argparse
import hashlib
import json
from dataclasses import replace
from pathlib import Path

import cv2
import matplotlib
from PIL import Image, ImageDraw, ImageFont

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from object_detector import Detector, DetectorConfig  # noqa: E402
from object_detector.annotation import annotate  # noqa: E402
from object_detector.benchmark import benchmark, compare_predictions, environment  # noqa: E402
from object_detector.media import process_video, read_image  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]


def font(size):
    try:
        return ImageFont.truetype("DejaVuSans.ttf", size)
    except OSError:
        try:
            return ImageFont.truetype("C:/Windows/Fonts/arial.ttf", size)
        except OSError:
            return ImageFont.load_default(size=size)


def gallery(images, predictions, destination):
    canvas = Image.new("RGB", (1440, 1070), "#0f172a")
    draw = ImageDraw.Draw(canvas)
    draw.text((40, 28), "OBJECT DETECTOR / YOLO11 + ONNX", font=font(32), fill="white")
    draw.text((40, 78), "Actual inference outputs · Farzam Fattahi", font=font(20), fill="#cbd5e1")
    for index, (name, image) in enumerate(images.items()):
        prediction = predictions[name]
        top = 136 + index * 446
        for column, frame in enumerate((image, annotate(image, prediction))):
            panel = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            panel.thumbnail((650, 368), Image.Resampling.LANCZOS)
            x = 40 + column * 700
            canvas.paste(panel, (x + (650 - panel.width) // 2, top + 38))
            title = "INPUT" if column == 0 else f"DETECTIONS / {len(prediction.detections)} objects"
            draw.text((x, top), title, font=font(20), fill="#93c5fd" if column else "#cbd5e1")
        labels = ", ".join(f"{d.label} {d.confidence:.2f}" for d in prediction.detections)
        draw.text((40, top + 412), f"{name}: {labels}", font=font(17), fill="white")
    canvas.save(destination)


def plot_benchmarks(measurements, destination):
    plt.rcParams.update({"font.size": 12, "axes.spines.top": False, "axes.spines.right": False})
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), layout="constrained")
    names = ["PyTorch CPU", "ONNX Runtime CPU"]
    colors = ["#475569", "#1d4ed8"]
    medians = [m["median_ms"] for m in measurements]
    axes[0].barh(names, medians, color=colors)
    axes[0].set_xlim(0, max(medians) * 1.35)
    axes[0].set_xlabel("Median prediction latency (ms) · lower is better")
    for i, value in enumerate(medians):
        axes[0].text(value + max(medians) * 0.025, i, f"{value:.1f} ms", va="center")
    for name, color, m in zip(names, colors, measurements, strict=True):
        axes[1].plot(m["samples_ms"], label=name, color=color)
    axes[1].set_xlabel("Measured call after 5 warmup calls")
    axes[1].set_ylabel("Latency (ms)")
    axes[1].legend(frameon=False)
    fig.suptitle("YOLO11n · 640 × 640 · batch 1 · 4 CPU threads", fontsize=17, weight="bold")
    fig.savefig(destination, dpi=160, facecolor="white")
    plt.close(fig)


def make_gif(video, destination, max_frames, stride=2):
    capture = cv2.VideoCapture(str(video))
    frames = []
    fps = capture.get(cv2.CAP_PROP_FPS)
    try:
        for index in range(max_frames):
            ok, frame = capture.read()
            if not ok:
                break
            if index % stride:
                continue
            image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            image.thumbnail((512, 384), Image.Resampling.LANCZOS)
            frames.append(image)
    finally:
        capture.release()
    if not frames:
        raise ValueError("No video frames decoded for GIF.")
    frames[0].save(
        destination,
        save_all=True,
        append_images=frames[1:],
        duration=round(1000 * stride / fps),
        loop=0,
        optimize=True,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repeats", type=int, default=40)
    parser.add_argument("--video-frames", type=int, default=60)
    args = parser.parse_args()
    assets = ROOT / "assets"
    assets.mkdir(exist_ok=True)
    config = DetectorConfig()
    torch_detector = Detector(config)
    onnx_detector = Detector(
        replace(config, model=str(ROOT / "models/yolo11n.onnx"), backend="onnx")
    )
    images = {name: read_image(ROOT / "data" / name) for name in ("bus.jpg", "zidane.jpg")}
    predictions = {name: torch_detector.predict(image) for name, image in images.items()}
    parity = {
        name: compare_predictions(predictions[name], onnx_detector.predict(image))
        for name, image in images.items()
    }
    # Also compare real video frames, rather than restricting parity to two demo photos.
    capture = cv2.VideoCapture(str(ROOT / "data/vtest.avi"))
    try:
        for index in (0, 15, 30, 45):
            capture.set(cv2.CAP_PROP_POS_FRAMES, index)
            ok, image = capture.read()
            if not ok:
                raise ValueError(f"Cannot read video frame {index}.")
            parity[f"vtest.avi:frame{index}"] = compare_predictions(
                torch_detector.predict(image), onnx_detector.predict(image)
            )
    finally:
        capture.release()
    if not all(item["all_matched"] for item in parity.values()):
        raise RuntimeError(f"Backend parity check failed: {parity}")
    measurements = [
        benchmark(detector, list(images.values()), args.repeats, warmup=5)
        for detector in (torch_detector, onnx_detector)
    ]
    gallery(images, predictions, assets / "detections.jpg")
    plot_benchmarks(measurements, assets / "benchmark.png")
    output = ROOT / "outputs/pedestrians.mp4"
    summary = process_video(
        onnx_detector, str(ROOT / "data/vtest.avi"), output, max_frames=args.video_frames
    )
    make_gif(output, assets / "demo.gif", args.video_frames)
    hashes = {
        str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in (ROOT / "yolo11n.pt", ROOT / "models/yolo11n.onnx")
    }
    report = {
        "environment": environment(),
        "model_sha256": hashes,
        "image_predictions": {name: p.to_dict() for name, p in predictions.items()},
        "parity": parity,
        "benchmarks": measurements,
        "video": summary,
        "sources": json.loads((ROOT / "data/sources.json").read_text(encoding="utf-8")),
        "limitations": [
            "Pretrained weights; no new training performed.",
            "Two demo photos and four real video frames for parity; not an accuracy benchmark.",
            "CPU only; GPU and physical webcam not verified in this report.",
            "Video output is silent; GIF playback FPS is unrelated to processing FPS.",
        ],
    }
    (assets / "results.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(
        json.dumps(
            {
                "parity": parity,
                "benchmarks": [
                    {k: m[k] for k in ("mean_ms", "median_ms", "p95_ms", "prediction_fps")}
                    for m in measurements
                ],
                "video": summary,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
