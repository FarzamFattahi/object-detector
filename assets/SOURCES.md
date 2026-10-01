# Evidence provenance

The detection gallery and GIF are rendered from actual model predictions by
`scripts/make_evidence.py`. The benchmark chart is plotted from actual measured
calls. The raw report includes package versions, settings, input URLs and SHA-256
hashes. No generated or manually drawn detection boxes are used as performance evidence.

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
