# v1.0.0 — Object detection application and ONNX deployment

The repository previously contained a project plan and README without executable code.
This release provides a tested YOLO11n inference application for images, directories,
recorded video and explicitly opened local webcams.

- Installable Python package, CLI, typed prediction records and configurable filters.
- Independent ONNX Runtime backend with letterboxing, normalization, raw-output
  decoding, class-aware NMS and original-coordinate boxes.
- Streamlit dashboard with camera snapshots, image/video upload, counts, detection
  records and PNG/JSON/H.264 video/JSONL downloads.
- Streaming video I/O with cleanup on failures; no audio preservation or tracking.
- Reproducible real-image/video evidence, CPU benchmarks and backend parity report.
- Offline tests, Linux/Windows CI, package builds, practical learning guide and source credits.

## Validation and measured results

37 tests pass locally; lint, format, dependency and package build checks pass.
Real-model PyTorch/ONNX checks match all 32 detections on two photos and four video
frames, minimum IoU 0.9999938. On an AMD Ryzen 7 7435HS with 640-pixel FP32 inputs and
four CPU threads, median prediction latency is 79.6 ms PyTorch and 72.3 ms ONNX.
The 60-frame video pipeline processes approximately 12.5 FPS including drawing/encoding.
Both image engines, JSON download and a real 30-frame browser video test were exercised.

COCO8 mAP is included solely as a tiny labeled smoke test, not held-out accuracy.
Physical webcam and CUDA are implemented entry points but were not verified in this run.
The model weights are pretrained upstream weights; this release does not claim new training.

## Release assets

The ONNX graph corresponds to the SHA-256 in `assets/results.json`. Place it in
`models/yolo11n.onnx` to use the dashboard's ONNX option without exporting locally.
The wheel installs the Python API/CLI; clone the repository for the dashboard and scripts.
The source archive includes the dashboard, documentation and evidence. See README for setup.

Code and the model dependency use AGPL-3.0; see LICENSE and evidence sources.
