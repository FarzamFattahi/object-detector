"""Launch with: streamlit run app.py"""

import json
import subprocess
import tempfile
import threading
from collections import Counter
from pathlib import Path

import cv2
import imageio_ffmpeg
import numpy as np
import streamlit as st
from PIL import Image, ImageOps

from object_detector import Detector, DetectorConfig
from object_detector.annotation import annotate
from object_detector.media import process_video, read_image
from object_detector.ppe_dataset import NAMES as PPE_NAMES
from object_detector.types import Detection, Prediction

ROOT = Path(__file__).resolve().parent
st.set_page_config(
    page_title="Object Detector | Farzam Fattahi", layout="wide", page_icon=":material/scan:"
)
st.markdown(
    """
<style>
.block-container {max-width: 1400px; padding-top: 2.5rem;}
h1 {letter-spacing: -.035em;}
[data-testid="stMetric"] {background: white; border: 1px solid #cbd5e1;
 border-radius: 10px; padding: 1rem;}
</style>
""",
    unsafe_allow_html=True,
)


@st.cache_resource(show_spinner="Loading detector…", max_entries=4)
def load_detector(config):
    # Ultralytics predictors mutate internal state; serialize shared-resource calls.
    return Detector(config), threading.Lock()


def png_bytes(bgr):
    ok, encoded = cv2.imencode(".png", bgr)
    if not ok:
        raise OSError("Image encoding failed.")
    return encoded.tobytes()


st.caption("FARZAM FATTAHI / COMPUTER VISION PORTFOLIO / PROJECT 02")
st.title("See what the model sees.")
st.write(
    "Detect objects in photos and video. Inspect every box, tune the threshold, "
    "and export the results."
)

with st.sidebar:
    st.header("Detection settings")
    profile = st.selectbox("Model task", ["Construction PPE", "General objects (COCO)"])
    ppe = profile == "Construction PPE"
    backend = st.selectbox("Inference engine", ["PyTorch", "ONNX Runtime"])
    model_path = (
        ROOT
        / "models"
        / (
            ("ppe-yolo11n.onnx" if backend == "ONNX Runtime" else "ppe-yolo11n.pt")
            if ppe
            else "yolo11n.onnx"
        )
    )
    if ppe and not model_path.is_file():
        st.info("Download the trained PPE model: python scripts/download_ppe_model.py")
    elif backend == "ONNX Runtime" and not model_path.is_file():
        st.info("Export the ONNX model first: cv-detect export")
    confidence = st.slider("Minimum confidence", 0.05, 0.95, 0.25, 0.05)
    iou = st.slider(
        "NMS overlap threshold",
        0.10,
        0.90,
        0.45,
        0.05,
        help="Higher values retain more overlapping boxes of the same class.",
    )
    selection = st.multiselect(
        "Filter classes",
        PPE_NAMES
        if ppe
        else [
            "person",
            "bicycle",
            "car",
            "motorcycle",
            "bus",
            "cat",
            "dog",
            "bottle",
            "chair",
            "laptop",
            "cell phone",
        ],
    )
    st.caption(
        "Leave empty to detect all 11 PPE classes."
        if ppe
        else "Leave empty to detect all 80 COCO classes."
    )
    st.divider()
    st.write("YOLO11 nano · 640 px · CPU")
    st.caption(
        "Inference runs on the machine hosting this app. The first PyTorch run downloads weights."
    )

ids = {
    "person": 0,
    "bicycle": 1,
    "car": 2,
    "motorcycle": 3,
    "bus": 5,
    "cat": 15,
    "dog": 16,
    "bottle": 39,
    "chair": 56,
    "laptop": 63,
    "cell phone": 67,
}
config = DetectorConfig(
    model=str(model_path) if ppe or backend == "ONNX Runtime" else "yolo11n.pt",
    backend="onnx" if backend == "ONNX Runtime" else "torch",
    confidence=confidence,
    iou=iou,
    classes=tuple((PPE_NAMES.index(name) if ppe else ids[name]) for name in selection)
    if selection
    else None,
)

if ppe:
    st.caption(
        "Fine-tuned on Construction-PPE. Labels describe model predictions, not verified "
        "worker compliance. Missing detections do not establish missing equipment."
    )
    with st.expander("Dataset experiment and held-out evidence"):
        metrics_path = ROOT / "assets/ppe/metrics.json"
        if metrics_path.is_file():
            measured = json.loads(metrics_path.read_text(encoding="utf-8"))
            cols = st.columns(3)
            cols[0].metric("Test mAP50", f"{measured['mAP50']:.1%}")
            cols[1].metric("Test mAP50–95", f"{measured['mAP50_95']:.1%}")
            cols[2].metric("Held-out images", measured["dataset"]["splits"]["test"]["images"])
            st.image(str(ROOT / "assets/ppe/class_performance.png"))
            st.caption(
                "Checkpoint selected using validation only. See docs/PPE_CASE_STUDY.md "
                "for class imbalance, error examples, and evaluation settings."
            )

mode = st.radio("Input source", ["Image", "Camera snapshot", "Video"], horizontal=True)
image = None
reference = None
video = None
input_key = None
if mode == "Image":
    uploaded = st.file_uploader("Upload a photo", type=["jpg", "jpeg", "png", "webp"])
    examples = (
        ["PPE test: strong", "PPE test: median", "PPE test: weak", "None"]
        if ppe
        else ["Street scene", "Two people", "None"]
    )
    example = st.selectbox("Or try an example", examples)
    if uploaded:
        try:
            pil = ImageOps.exif_transpose(Image.open(uploaded)).convert("RGB")
            if pil.width * pil.height > 25_000_000:
                raise ValueError("Please use an image smaller than 25 megapixels.")
            image = cv2.cvtColor(np.array(pil), cv2.COLOR_RGB2BGR)
            input_key = uploaded.getvalue()
        except (OSError, ValueError) as error:
            st.error(f"Cannot read photo: {error}")
    elif example != "None":
        path = (
            ROOT
            / "assets/ppe/samples"
            / {
                "PPE test: strong": "test-1.jpg",
                "PPE test: median": "test-3.jpg",
                "PPE test: weak": "test-5.jpg",
            }[example]
            if ppe
            else ROOT / "data" / ("bus.jpg" if example == "Street scene" else "zidane.jpg")
        )
        if path.is_file():
            image = read_image(path)
            input_key = str(path)
            annotation_path = path.with_suffix(".json")
            if (
                ppe
                and annotation_path.is_file()
                and st.checkbox("Show dataset annotations for this example")
            ):
                annotation = json.loads(annotation_path.read_text(encoding="utf-8"))
                reference = Prediction(
                    tuple(
                        Detection(d["class_id"], d["label"], 1.0, tuple(d["xyxy"]))
                        for d in annotation["detections"]
                        if config.classes is None or d["class_id"] in config.classes
                    ),
                    annotation["width"],
                    annotation["height"],
                    0.0,
                    "dataset reference",
                )
        else:
            st.info(
                "Run python scripts/download_samples.py to enable the examples, or upload a photo."
            )
elif mode == "Camera snapshot":
    photo = st.camera_input("Take a photo")
    if photo:
        pil = ImageOps.exif_transpose(Image.open(photo)).convert("RGB")
        image = cv2.cvtColor(np.array(pil), cv2.COLOR_RGB2BGR)
        input_key = photo.getvalue()
    st.caption("For a continuous local webcam feed: cv-detect detect --source webcam:0 --show")
else:
    video = st.file_uploader("Upload a video", type=["mp4", "avi", "mov", "mkv"])
    limit = st.slider("Maximum frames to process", 30, 900, 150, 30)
    st.caption(
        "Dashboard processing is bounded to this many frames. "
        "The CLI can process full videos. Audio is omitted."
    )
    if video:
        input_key = video.getvalue()

# A result belongs to its input AND settings. Downloads remain available across reruns.
result_key = (mode, input_key, config, limit if mode == "Video" else None)
if st.session_state.get("result_key") != result_key:
    st.session_state.pop("result", None)

if st.button("Run detection", type="primary", disabled=image is None and video is None):
    try:
        detector, lock = load_detector(config)
        if image is not None:
            with st.spinner("Finding objects…"), lock:
                prediction = detector.predict(image)
            st.session_state.result = {
                "kind": "image",
                "prediction": prediction,
                "annotated": annotate(image, prediction),
            }
        else:
            bar = st.progress(0.0, text="Processing video…")
            with tempfile.TemporaryDirectory(prefix="cv-detector-") as directory:
                source = Path(directory) / ("input" + Path(video.name).suffix)
                source.write_bytes(video.getvalue())
                output = Path(directory) / "detected.mp4"
                with lock:
                    summary = process_video(
                        detector,
                        str(source),
                        output,
                        max_frames=limit,
                        on_progress=lambda n, total: bar.progress(
                            min(n / (total if total > 0 else limit), 1.0),
                            text=f"Processed {n} frames",
                        ),
                    )
                playable = Path(directory) / "preview.mp4"
                subprocess.run(
                    [
                        imageio_ffmpeg.get_ffmpeg_exe(),
                        "-y",
                        "-i",
                        str(output),
                        "-an",
                        "-c:v",
                        "libx264",
                        "-pix_fmt",
                        "yuv420p",
                        "-vf",
                        "pad=ceil(iw/2)*2:ceil(ih/2)*2",
                        "-movflags",
                        "+faststart",
                        str(playable),
                    ],
                    check=True,
                    capture_output=True,
                    timeout=180,
                )
                st.session_state.result = {
                    "kind": "video",
                    "video": playable.read_bytes(),
                    "records": output.with_suffix(".jsonl").read_bytes(),
                    "summary": summary,
                }
            bar.empty()
        st.session_state.result_key = result_key
    except (ValueError, OSError, RuntimeError, subprocess.SubprocessError) as error:
        st.error(f"Detection failed: {error}. Check the model, input file and selected engine.")

result = st.session_state.get("result")
if result and result["kind"] == "image":
    p = result["prediction"]
    a, b, c = st.columns(3)
    a.metric("Objects detected", len(p.detections))
    b.metric("Classes found", len({d.class_id for d in p.detections}))
    c.metric("Prediction time", f"{p.latency_ms:.0f} ms")
    st.caption(
        "Prediction time includes preprocessing, inference and NMS. "
        "First-run timing includes warmup."
    )
    left, right = st.columns(2)
    left.image(
        annotate(image, reference, show_confidence=False) if reference else image,
        channels="BGR",
        caption="Dataset annotations" if reference else "Original image",
        use_container_width=True,
    )
    right.image(
        result["annotated"], channels="BGR", caption="Detected objects", use_container_width=True
    )
    rows = [
        {
            "Class": d.label,
            "Confidence": round(d.confidence, 4),
            "x1": round(d.xyxy[0], 1),
            "y1": round(d.xyxy[1], 1),
            "x2": round(d.xyxy[2], 1),
            "y2": round(d.xyxy[3], 1),
        }
        for d in p.detections
    ]
    if rows:
        st.dataframe(rows, use_container_width=True, hide_index=True)
        st.caption(
            "Counts: "
            + ", ".join(
                f"{name}: {count}" for name, count in Counter(d.label for d in p.detections).items()
            )
        )
    else:
        st.info(
            "No objects passed the current filter. "
            "Try lowering confidence or clearing class filters."
        )
    a, b = st.columns(2)
    a.download_button(
        "Download annotated PNG", png_bytes(result["annotated"]), "detections.png", "image/png"
    )
    b.download_button(
        "Download detections JSON",
        json.dumps(p.to_dict(), indent=2),
        "detections.json",
        "application/json",
    )
elif result:
    st.video(result["video"])
    st.json(result["summary"])
    a, b = st.columns(2)
    a.download_button("Download annotated video", result["video"], "detections.mp4", "video/mp4")
    b.download_button(
        "Download frame records", result["records"], "detections.jsonl", "application/x-ndjson"
    )
elif image is not None:
    st.image(image, channels="BGR", caption="Ready for detection", width=700)
else:
    st.info("Choose an input above to begin.")

st.divider()
st.caption(
    "General detector: COCO-pretrained Ultralytics YOLO11n. PPE detector: fine-tuned on "
    "Construction-PPE. See the case study for data provenance and evaluation limitations."
)
