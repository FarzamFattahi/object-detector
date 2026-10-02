# Evidence provenance

The detection gallery and GIF are rendered from actual model predictions by
`scripts/make_evidence.py`. The benchmark chart is plotted from actual measured
calls. The raw report includes package versions, settings, input URLs and SHA-256
hashes. Prediction boxes come from actual model outputs.

Inputs are downloaded separately rather than bundled as a training dataset:

- `bus.jpg` and `zidane.jpg`: upstream
  [Ultralytics assets](https://github.com/ultralytics/assets/tree/main/im).
- `vtest.avi`: upstream
  [OpenCV video sample](https://github.com/opencv/opencv/blob/4.x/samples/data/vtest.avi),
  the pedestrian sequence distributed with OpenCV (PETS imagery).
- COCO8 sanity evaluation: upstream
  [COCO8](https://docs.ultralytics.com/datasets/detect/coco8/) distributed by Ultralytics.

The examples remain third-party imagery; credit and underlying image rights remain
with their sources. The code and YOLO11 model dependency are AGPL-3.0. The evidence
does not claim authorship of the input photographs or pretrained model weights.

`demo.gif` plays a short real video sequence at the source's playback rate. Its visual
playback speed does not demonstrate inference FPS. Read `results.json` for measured
processing speed. The GIF samples every second frame and retains the source duration.
`dashboard.jpg` is a screenshot of the running app.

## Construction-PPE evidence (v1.1.0)

The dataset is [Construction-PPE](https://docs.ultralytics.com/datasets/detect/construction-ppe),
by Mrunmayee Dalvi, Niyati Singh, Sahil Bhingarde and Ketaki Chalke, published by
Ultralytics in 2025 under AGPL-3.0. Its pinned archive digest is in `ppe/download.json`.
The archive includes a copy of the dataset license. Dataset imagery and annotations
remain credited to their authors; Farzam's contribution is the audit, filtering,
fine-tuning experiment, evaluation and deployment implementation.

`ppe/leakage_examples.jpg` displays actual related frames found across original
splits; their source images and hashes are in `ppe/leakage_sources.json`.
`ppe/test_gallery.jpg` places dataset annotations beside actual checkpoint
predictions. `ppe/sample_sources.json` records the six included demonstration
images and their disclosed strong/median/weak selection method. These examples
are illustrative; the metric report uses all retained test images. All test
prediction records are published in `ppe/test_predictions.json`.

Class-count, annotation-size and learning plots are computed from the audited
labels and training/evaluation records. The model architecture and initial COCO
weights are from Ultralytics YOLO11; the released PPE weights are fine-tuned.
