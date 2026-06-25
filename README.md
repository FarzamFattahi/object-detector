# Real-Time Object Detector

A high-performance object detection system built on YOLOv11, capable of processing images, video files, and live webcam feeds with sub-30ms inference latency.

## Features

- Real-time detection on webcam at 30+ FPS
- Support for 80 COCO object classes out of the box
- ONNX export for framework-agnostic deployment
- Confidence threshold and NMS tuning via config
- Annotated output with bounding boxes, labels, and scores
- Streamlit dashboard for interactive demos

## Demo

![Demo](assets/demo.gif)

## Tech Stack

| Component | Technology |
|-----------|-----------|
| Detection model | YOLOv11 (Ultralytics) |
| Inference runtime | ONNX Runtime |
| Video processing | OpenCV |
| Web UI | Streamlit |

## Installation

```bash
git clone https://github.com/FarzamFattahi/object-detector.git
cd object-detector
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

## Usage

**Run on image**
```bash
python src/detect.py --source image.jpg
```

**Run on video**
```bash
python src/detect.py --source video.mp4 --output output.mp4
```

**Live webcam**
```bash
python src/detect.py --source webcam
```

**Launch dashboard**
```bash
streamlit run src/app.py
```

## Performance

| Mode | Device | FPS |
|------|--------|-----|
| YOLOv11n | CPU | ~30 |
| YOLOv11n | GPU (CUDA) | ~120 |
| ONNX Runtime | CPU | ~40 |

## License

MIT
