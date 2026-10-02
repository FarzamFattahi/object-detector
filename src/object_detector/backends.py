"""PyTorch inference and independent ONNX preprocessing/decoding."""

import ast
from pathlib import Path

import numpy as np

from .config import DetectorConfig
from .geometry import letterbox, nms, restore_boxes
from .types import Detection


class TorchBackend:
    def __init__(self, config: DetectorConfig):
        import torch

        from ultralytics import YOLO

        torch.set_num_threads(config.threads)
        self.config = config
        self.model = YOLO(config.model, task="detect")
        self.names = self.model.names
        if config.classes and any(c not in self.names for c in config.classes):
            raise ValueError("Class filter contains IDs outside this model's classes.")

    def predict(self, image: np.ndarray) -> tuple[Detection, ...]:
        c = self.config
        result = self.model.predict(
            image,
            imgsz=c.image_size,
            conf=c.confidence,
            iou=c.iou,
            classes=c.classes,
            max_det=c.max_detections,
            device=c.device,
            rect=False,
            verbose=False,
        )[0]
        return tuple(
            Detection(
                int(row[5]),
                self.names[int(row[5])],
                float(row[4]),
                tuple(float(x) for x in row[:4]),
            )
            for row in result.boxes.data.cpu().numpy()
        )


class OnnxBackend:
    """YOLO11 FP32 raw detection output [1, 4 + classes, anchors], batch size one.

    Ultralytics/PyTorch are not imported by this backend. Preprocessing, decoding,
    NMS and restoration to original coordinates are implemented here explicitly.
    """

    def __init__(self, config: DetectorConfig):
        import onnxruntime as ort

        if not Path(config.model).is_file():
            raise FileNotFoundError("ONNX file missing. Run cv-detect export first.")
        options = ort.SessionOptions()
        options.intra_op_num_threads = config.threads
        options.inter_op_num_threads = 1
        self.session = ort.InferenceSession(
            config.model, sess_options=options, providers=["CPUExecutionProvider"]
        )
        self.config = config
        self.input = self.session.get_inputs()[0]
        if self.input.type != "tensor(float)" or self.input.shape != [
            1,
            3,
            config.image_size,
            config.image_size,
        ]:
            raise ValueError("Use a static FP32 batch-1 export with matching --image-size.")
        raw_names = self.session.get_modelmeta().custom_metadata_map.get("names")
        if raw_names is None:
            raise ValueError("ONNX model must contain Ultralytics 'names' metadata.")
        self.names = ast.literal_eval(raw_names)
        if isinstance(self.names, list):
            self.names = dict(enumerate(self.names))
        if not isinstance(self.names, dict) or set(self.names) != set(range(len(self.names))):
            raise ValueError("Model names metadata must use contiguous integer IDs.")
        if config.classes and any(c not in self.names for c in config.classes):
            raise ValueError("Class filter contains IDs outside this model's classes.")

    def predict(self, image: np.ndarray) -> tuple[Detection, ...]:
        c = self.config
        padded, scale, padding = letterbox(image, c.image_size)
        # OpenCV BGR HWC uint8 -> RGB BCHW float32 in [0, 1].
        tensor = (
            np.ascontiguousarray(padded[:, :, ::-1].transpose(2, 0, 1)[None], dtype=np.float32)
            / 255.0
        )
        output = self.session.run(None, {self.input.name: tensor})[0]
        if output.ndim != 3 or output.shape[0] != 1 or output.shape[1] != 4 + len(self.names):
            raise ValueError("Expected raw YOLO11 output; export with nms=False.")
        rows = output[0].T
        class_ids = rows[:, 4:].argmax(axis=1)
        scores = rows[np.arange(len(rows)), class_ids + 4]
        valid = np.isfinite(rows).all(axis=1) & (scores > c.confidence)
        if c.classes is not None:
            valid &= np.isin(class_ids, c.classes)
        rows, scores, class_ids = rows[valid], scores[valid], class_ids[valid]
        # Bound worst-case NMS work before quadratic suppression.
        order = np.argsort(-scores, kind="stable")[:30000]
        rows, scores, class_ids = rows[order], scores[order], class_ids[order]
        boxes = np.empty((len(rows), 4), dtype=np.float32)
        boxes[:, :2] = rows[:, :2] - rows[:, 2:4] / 2
        boxes[:, 2:] = rows[:, :2] + rows[:, 2:4] / 2
        keep = nms(boxes, scores, class_ids, c.iou, c.max_detections)
        h, w = image.shape[:2]
        boxes = restore_boxes(boxes[keep], scale, padding, w, h)
        return tuple(
            Detection(
                int(class_ids[index]),
                self.names[int(class_ids[index])],
                float(scores[index]),
                tuple(float(x) for x in box),
            )
            for index, box in zip(keep, boxes, strict=True)
        )
