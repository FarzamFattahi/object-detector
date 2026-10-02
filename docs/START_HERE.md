# Your local learning session

On Windows, double-click **Start Demo.cmd** in the project folder. It uses this
project's virtual environment, verifies the released model checksums and starts
the dashboard at http://localhost:8501. Keep the terminal open while using the
demo; Ctrl+C stops it. The launcher downloads models only if they are missing.
If another copy is already running, use its browser tab or stop that copy first.

## First session: predictions and mistakes (20 minutes)

1. Select **Construction PPE**, **ONNX Runtime** and **Image**.
2. Choose **PPE test: strong** and enable dataset annotations. Click **Run detection**.
   The left image contains human-provided labels; the right contains predictions.
   Read the class, confidence and original-pixel box coordinates in the table.
3. Repeat with **PPE test: median** and **PPE test: weak**. Record missed objects,
   extra boxes and incorrect classes. These examples were deliberately selected
   to illustrate different outcomes; the gallery is not an accuracy estimate.
4. Change confidence from 0.25 to 0.50, then 0.10. Higher thresholds usually discard
   more detections. Lower thresholds may recover objects and add false positives.
   Confidence is a model score, not a guarantee that the label is correct.
5. Switch to **PyTorch** and repeat. It should produce nearly identical detections.
   Backend agreement checks deployment correctness, not agreement with annotations.
6. Upload your own photo and download its annotated image and JSON. A new photo
   has no reference labels, so viewing its predictions cannot produce an AP score.

## What was built

The existing application ran a general COCO-pretrained detector. This extension
adapts YOLO11n to the Construction-PPE dataset's eleven labels, including helmet,
gloves, vest, boots, goggles, Person and missing-equipment categories.

The workflow is **audit data → train → select on validation → evaluate on test →
export → compare deployment backends**. Exact pixel checks found no duplicates,
but perceptual hashes found related frames across splits. Grouping these frames
and prioritizing test, then validation, left 967 training, 129 validation and 141
test images. This is a heuristic audit, not proof of unseen-site generalization.

Fine-tuning starts from learned COCO features. The detection head adapts from 80
classes to eleven; the backbone also updates. PyTorch differentiates box,
classification and distribution-focal losses, and AdamW updates parameters.
Training ran for 40 epochs on the local RTX 4050. Epoch 38 had the best validation
mAP50–95 and became the released checkpoint. The test split did not select it.

## How to read the outputs

| Output | Meaning |
|---|---|
| `models/ppe-yolo11n.pt` | Fine-tuned PyTorch weights |
| `models/ppe-yolo11n.onnx` | Export for the independent ONNX Runtime backend |
| `assets/ppe/training.csv` | Losses and validation metrics for every epoch |
| `assets/ppe/learning_curves.png` | Training and validation trends |
| `assets/ppe/metrics.json` | Held-out test metrics, baseline and evaluation protocol |
| `assets/ppe/test_predictions.json` | Predictions for all 141 test images |
| `assets/ppe/test_gallery.jpg` | Selected strong, median and weak examples |
| `assets/ppe/deployment.json` | Timings and PyTorch/ONNX agreement |

All eleven test categories gave **mAP50 0.536 / mAP50–95 0.272**. AP summarizes
the precision–recall curve for one class; mAP averages classes. mAP50 accepts
matched boxes at intersection-over-union (IoU) 0.50. mAP50–95 averages stricter
IoU thresholds, so localization errors matter more. These values are not the
percentage of images that are correct.

The five worn-equipment classes averaged mAP50 **0.807**, while the four missing-
equipment classes averaged **0.141**. Person AP50 improved from **0.719** with the
general pretrained model to **0.831** with fine-tuning on the same test images.
The missing-equipment results show a real limitation: the demo cannot certify
worker safety. It does not associate equipment with particular workers.

Measured warmed CPU medians were **74.8 ms PyTorch / 57.7 ms ONNX** on this machine.
All 150 compared detections matched at IoU ≥0.99. The application displays timing
for individual calls, including first-run warmup, so its number can differ.

## Second session: understand the code (45 minutes)

Read the [full case study](PPE_CASE_STUDY.md) and run its tensor example. Follow:

1. `src/object_detector/ppe_dataset.py`: label checks and split grouping.
2. `configs/ppe_train.yaml` and `scripts/train_ppe.py`: fixed training recipe.
3. `scripts/evaluate_ppe.py`: validation-selected model tested against annotations.
4. `src/object_detector/backends.py`: image preprocessing, prediction decoding and NMS.
5. `app.py`: the model's predictions become images, tables and downloadable results.

Your first exercise: convert one normalized YOLO label into pixel coordinates.
Next, inspect the `[1, 3, 640, 640]` input and `[1, 15, 8400]` output in the case
study. Finally, explain one weak prediction using the reference boxes rather than
the confidence score alone. Retraining is optional; it is not required to use the
released model. Use validation data for future tuning and a fresh test set for a
new final comparison.
