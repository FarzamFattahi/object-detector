# v1.1.0 — Construction-PPE dataset case study

Adds an audited, fine-tuned eleven-class construction-equipment model to the
existing general detector. pHash grouping reduces cross-split related-image
leakage, leaving 967/129/141 train/validation/test images. Validation alone
selects the checkpoint after 40 training epochs.

- All-class held-out mAP50 **0.536**, mAP50–95 **0.272**.
- Worn-equipment AP50 **0.807**; missing-equipment
  labels remain weak. Full per-class support and failure examples are published.
- Person-only COCO baseline on identical images; all 141 test predictions,
  dataset/source hashes, resolved settings, learning curves and actual galleries.
- Trained PyTorch and FP32 ONNX release artifacts with hash-verified download.
- Independent ONNX agreement: **150 detections / twenty images**.
  Warmed CPU medians: **74.8 ms** PyTorch,
  **57.7 ms** ONNX.
- PPE dashboard profile, correct class filters and optional reference annotations.
- Practical PyTorch explanation, model card and reproducibility instructions.
- 53 offline tests plus Linux/Windows CI on Python 3.10 and 3.12.

This is a research demonstration. Missing detections do not prove missing
equipment, and image-level evaluation does not establish unseen-site performance.
Model and code are AGPL-3.0-only. Dataset authors are credited in the case study.

See [the model card](https://github.com/FarzamFattahi/object-detector/blob/master/docs/PPE_MODEL_CARD.md)
and [the case study](https://github.com/FarzamFattahi/object-detector/blob/master/docs/PPE_CASE_STUDY.md).
