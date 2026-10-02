# Construction-PPE YOLO11n model card

Fine-tuned by Farzam Fattahi from COCO-pretrained YOLO11n using Ultralytics
8.3.203 and PyTorch 2.8.0+cu128. This model predicts eleven upstream categories;
it supports equipment-detection research and inspection of model errors. It does
not establish worker compliance. The architecture is upstream YOLO11n; this
project contributes data auditing, fine-tuning, evaluation and deployment.

## Dataset and protocol

Source: [Construction-PPE](https://docs.ultralytics.com/datasets/detect/construction-ppe),
credited to Mrunmayee Dalvi, Niyati Singh, Sahil Bhingarde and Ketaki Chalke (2025).
The archive includes AGPL-3.0. Its original 1,132/143/141 train/validation/test
images contain visually related frames across splits. Exact decoded-pixel hashes
found no duplicates. A predeclared 64-bit DCT pHash distance ≤8 connected-component
policy keeps groups in test, otherwise validation, otherwise train. Excluding
165 training and 14 validation images leaves **967/129/141 images**. Retained test
images contain 1,251 annotation rows in 131 similarity groups.

Labels retain their original meanings and IDs. `none` is ambiguous and is not
interpreted as a particular condition. One duplicate training label row is removed
by the upstream loader. No repeated rows were found in retained validation/test.
The full archive, pixel/label hashes, grouping decisions and excluded records are
in `assets/ppe/download.json` and `assets/ppe/data_audit.json`.

Test images were inspected for split construction and annotation-size diagnostics.
Test predictions were not used to select the checkpoint, training recipe or
operating threshold. Semantic group summaries were declared after validation
diagnostics and before test evaluation. They use the same selected checkpoint.

## Training

AdamW, initial learning rate 0.001, weight decay 0.0005, 640 px, microbatch 4,
gradient accumulation toward nominal batch 64, two workers, seed 42, deterministic
settings, mixed precision, maximum 40 epochs and patience 12. Both backbone and
head were fine-tuned; 448/499 state entries transferred from COCO initialization.
The training architecture has 2,591,985 parameters. Hardware: NVIDIA GeForce RTX 4050 Laptop GPU.
Completed **40 epochs**, selecting **epoch 38**
by all-class validation mAP50–95 (0.271).
Recorded training duration: 1.98 hours, including
validation and shared-laptop resource contention; it is not a throughput benchmark.

The exact initial checkpoint, source/configuration hashes, source Git revision,
CUDA/cuDNN/runtime versions and dataset audit hash are in `experiment.json`.
`resolved_train_args.yaml` captures upstream-resolved settings with portable paths;
`training.csv` records every completed epoch. Different runtimes can change results
despite seeded settings. See [the reproducibility guide](PPE_CASE_STUDY.md).

## Held-out test results

FP32, 640 px, batch 8, CPU, fixed square letterbox, confidence floor 0.001,
class-aware NMS IoU 0.7. AP50–95 averages IoU thresholds 0.50 through 0.95.

| Categories | AP50 | AP50–95 |
|---|---:|---:|
| All eleven (primary result) | 0.536 | 0.272 |
| Five worn-equipment classes | 0.807 | 0.423 |
| Four missing-equipment labels | 0.141 | 0.048 |

Subsets are unweighted means of the same per-class AP values, not separately
trained models. Every category is reported below. TP/FP/FN use a separate,
predeclared deployment operating point: confidence 0.25, NMS IoU 0.45,
same-class matching IoU 0.50, confidence-ordered one-to-one matching.

| Category | Test boxes | AP50 | AP50–95 | TP | FP | FN |
|---|---:|---:|---:|---:|---:|---:|
| helmet | 192 | 0.926 | 0.506 | 172 | 10 | 20 |
| gloves | 163 | 0.745 | 0.350 | 116 | 31 | 47 |
| vest | 178 | 0.898 | 0.569 | 152 | 25 | 26 |
| boots | 211 | 0.715 | 0.376 | 145 | 52 | 66 |
| goggles | 52 | 0.754 | 0.315 | 35 | 6 | 17 |
| none | 65 | 0.459 | 0.166 | 32 | 11 | 33 |
| Person | 236 | 0.831 | 0.521 | 192 | 36 | 44 |
| no_helmet | 40 | 0.228 | 0.098 | 4 | 3 | 36 |
| no_goggle | 33 | 0.156 | 0.046 | 0 | 1 | 33 |
| no_gloves | 58 | 0.075 | 0.020 | 1 | 10 | 57 |
| no_boots | 23 | 0.104 | 0.028 | 0 | 0 | 23 |

The COCO-pretrained Person-only baseline scores AP50 0.719
and AP50–95 0.301. The fine-tuned model's Person
scores are 0.831/0.521, using identical images and
236 person annotations. COCO Person ID 0 is mapped to PPE Person ID 6; other COCO
classes are excluded. This comparison cannot establish improvement on all eleven
categories. Summary precision/recall in `metrics.json` use Ultralytics' best-F1
point; a precision of 1 with recall 0 means no positive detections at that point.

![Per-class performance and annotation support](../assets/ppe/class_performance.png)

## Deployment and evidence

Both PyTorch and the independent NumPy/ONNX Runtime decoder support the eleven
class names from model metadata. Static FP32 ONNX has output `[1, 15, 8400]`.
All **150 detections across twenty specified test images** matched
one-to-one at IoU ≥0.99 (minimum observed 0.999997). Backend agreement
checks export/postprocessing, not accuracy against annotations.

Warmed CPU median prediction: PyTorch **74.8 ms**,
ONNX **57.7 ms**, four threads, batch one, five warmups,
forty calls cycling over twenty images. Timing covers preprocessing, inference
and NMS, excluding loading, file I/O and drawing. Machine-specific raw timings,
input/model hashes and environment are in `deployment.json`; these observations
are not a universal real-time guarantee.

[Release artifacts](https://github.com/FarzamFattahi/object-detector/releases/tag/v1.1.0)
include trained `.pt` and `.onnx` models. `scripts/download_ppe_model.py` verifies
their SHA-256 hashes using `assets/ppe/models.json`. The gallery includes two
strong, two median and two weak test images ranked by equipment F1, excluding
Person/ambiguous none and requiring an equipment annotation. This is a disclosed
illustrative selection. All 141 test prediction records are published.

![Annotations and strong, median and weak predictions](../assets/ppe/test_gallery.jpg)

## Limits

The gallery's `image1120.jpg` misses both annotated missing-equipment regions.
`image1125.jpg` has three unmatched equipment predictions and five missed equipment
boxes at the fixed threshold. These errors are published without changing the
selected model after testing. The dataset includes non-construction scenes;
its category labels do not verify protective certification.

Worn-equipment and missing-equipment categories have substantially different
performance. Rare missing labels have weak support: validation has only four
`no_boots` boxes, and test has 23. Small objects, occlusion, label ambiguity and
out-of-domain images require separate review. Missing detections do not establish
missing equipment; confidence scores are not calibrated probabilities.
No equipment-to-person association or temporal tracking is performed.

pHash filtering can miss related images or over-group unrelated images. Site and
session identifiers are absent, related frames remain within each retained split,
and uncertainty intervals are not reported. This is image-level performance on
the filtered dataset, not demonstrated generalization to unseen sites. Independent
annotation review, site-held-out validation and group-level uncertainty are
needed before any deployment decisions. Physical webcam behavior was not tested.

Code, upstream dependency and released model use AGPL-3.0-only. Dataset photographs
are third-party material attributed to their upstream source. The full training
dataset is downloaded separately rather than redistributed in this repository.
