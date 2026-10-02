# Project 2: Object Detector

**Status:** Complete · dataset case study added in v1.1.0
**Updated:** 2 October 2026

## Purpose and completed work

General image/video/webcam detection plus a fine-tuned Construction-PPE model.
The project covers typed prediction records, CLI and Streamlit UI, independent
ONNX preprocessing/decoding/NMS, dataset auditing, transfer learning, held-out
evaluation, failure analysis and reproducible evidence.

The audited split has 967 training, 129 validation and 141 test images. Similarity
groups stay within one split. Fine-tuning completed 40 epochs on an
RTX 4050, selecting epoch 38 using validation only.
All-class test mAP50 is 0.536, mAP50–95 0.272; worn-equipment
AP50 is 0.807. Rare missing-equipment labels remain weak.
All per-class scores, test predictions, audit decisions and hashes are published.

## Validation

53 offline tests, lint/format, dependency consistency and packaging pass locally.
CI covers Linux and Windows on Python 3.10 and 3.12. PPE deployment matches all
150 detections between PyTorch and ONNX across twenty test images.
Median warmed CPU prediction: 74.8 ms PyTorch and
57.7 ms ONNX. CUDA fine-tuning and both PPE dashboard
engines/reference overlays were exercised. Physical webcam behavior remains untested.

The original v1.0 evidence includes 60 video frames, browser-decoded H.264 preview,
JSON downloads and responsive widths 375/768/1024/1440. Those measurements remain
separate from the new PPE test evidence.

## Repository and learning

[GitHub](https://github.com/FarzamFattahi/object-detector) ·
[v1.1.0](https://github.com/FarzamFattahi/object-detector/releases/tag/v1.1.0)

Read [README](README.md), [dataset experiment](docs/PPE_CASE_STUDY.md),
[model card](docs/PPE_MODEL_CARD.md), [learning guide](docs/LEARNING_GUIDE.md)
and [selection audit](docs/PROJECT_SELECTION.md).
