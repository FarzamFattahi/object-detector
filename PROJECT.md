# Project 2: Object Detector

**Status:** Complete
**Updated:** 1 October 2026

## Purpose

Deploy pretrained YOLO11n for images, video and an explicitly opened local webcam.
Learn PyTorch tensors, detection outputs, geometric preprocessing and ONNX deployment.

## Completed

- [x] Package, validated configuration, CLI and Python API
- [x] PyTorch inference and original-coordinate prediction records
- [x] Annotations, image directories, video output and frame JSONL
- [x] Local webcam CLI with Q/Ctrl+C cleanup
- [x] FP32 static ONNX export
- [x] Independent ONNX preprocessing, decoding and class-aware NMS
- [x] Streamlit image/camera-snapshot/video dashboard and downloads
- [x] Offline tests, packaging and Linux/Windows CI configuration
- [x] Real outputs, latency chart, parity report and COCO8 smoke evaluation
- [x] README, evidence provenance and practical learning guide

## Validation

37 offline tests passed locally. Real-model checks matched all 32 detections between
PyTorch and ONNX across two photos and four video frames. CPU median prediction
latency: 79.6 ms PyTorch, 72.3 ms ONNX, with raw calls in `assets/results.json`.
60 real video frames processed and encoded successfully. Both dashboard image engines
and JSON download were exercised in the browser. A 30-frame dashboard video test
produced a browser-decoded H.264 preview and frame records. Responsive page widths
were checked at 375, 768, 1024 and 1440 pixels. Physical webcam and CUDA were not tested.

## Repository

[FarzamFattahi/object-detector](https://github.com/FarzamFattahi/object-detector)

[Version 1.0.0](https://github.com/FarzamFattahi/object-detector/releases/tag/v1.0.0)

Read [README.md](README.md), [learning guide](docs/LEARNING_GUIDE.md) and
[selection audit](docs/PROJECT_SELECTION.md).
