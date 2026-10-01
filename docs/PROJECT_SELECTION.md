# Portfolio audit and selection — 1 October 2026

This audit distinguishes local executable code from plans and README promises. It
does not claim to have run or re-audited every previously completed project.

| Folder | State before this work | Decision |
|---|---|---|
| `document_scanner` | Marked complete; scanner, OCR, app and demo files present | Continue to the next project |
| `object_detector` | Not started; only PROJECT.md and README.md in its repository | Selected: next planned project and useful inference/deployment foundation |
| `face_anonymizer` | Not started; plan and README only | Later privacy/video project |
| `visual_search` | Not started; plan and README only | Later embedding/retrieval project |
| `pose_tracker` | Not started; plan and README only | Later landmarks/state-machine project |
| `image_segmentation` | Not started; plan and README only | Later segmentation project |
| `defect_detector` | Not started; plan and README only | Later anomaly-detection project |
| `depth_estimator` | Not started; plan and README only | Later geometry/depth project |
| `action_recognizer` | Not started; plan and README only | Later temporal-model project |
| `vision_assistant` | Not started; plan and README only | Later multimodal-serving project |
| `aircursor-open-source` | Separate implementation, tests and assets already present | Preserve existing work |

The selected repository already existed at
[FarzamFattahi/object-detector](https://github.com/FarzamFattahi/object-detector).
It had no implementation files and advertised unverified FPS. This release replaces
those promises with executable features, reproducible measurements, visual outputs,
and a guide explaining exactly which parts were built versus supplied pretrained.

YOLO11n is intentionally retained from the existing brief as a compact, established
baseline. It is not described as the latest available detector. The release emphasizes
correct deployment and credible evidence over changing the original model name.
