"""Fine-tune YOLO11n on audited PPE data; test images are never used here."""

import argparse
import hashlib
import json
import platform
import subprocess
from pathlib import Path

import torch
import ultralytics
import yaml
from ultralytics import YOLO

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=ROOT / "data/ppe-audited/data.yaml")
    parser.add_argument("--config", type=Path, default=ROOT / "configs/ppe_train.yaml")
    parser.add_argument("--device", default="0")
    parser.add_argument("--name", default="ppe-yolo11n")
    args = parser.parse_args()
    if not args.data.is_file():
        raise FileNotFoundError("Run scripts/prepare_ppe.py first")
    if args.device != "cpu" and not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable. Install CUDA PyTorch or explicitly use --device cpu")
    parameters = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    torch.set_num_threads(4)
    run = ROOT / "runs" / args.name
    if run.exists():
        raise FileExistsError(f"Choose a new --name; refusing to overwrite {run}")
    model = YOLO(str(ROOT / "yolo11n.pt"))
    metadata = {
        "initial_weights": "COCO-pretrained YOLO11n",
        "initial_sha256": hashlib.sha256((ROOT / "yolo11n.pt").read_bytes()).hexdigest(),
        "selection": "best checkpoint by validation fitness (mAP50-95); test unused",
        "parameters": parameters,
        "device": args.device,
        "torch": torch.__version__,
        "ultralytics": ultralytics.__version__,
        "python": platform.python_version(),
        "gpu": torch.cuda.get_device_name(0) if args.device != "cpu" else None,
        "cuda": torch.version.cuda,
        "cudnn": torch.backends.cudnn.version(),
        "git_revision": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT)
        .decode()
        .strip(),
        "audit_sha256": hashlib.sha256((args.data.parent / "audit.json").read_bytes()).hexdigest(),
        "source_sha256": {
            str(p.relative_to(ROOT)).replace("\\", "/"): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in (
                Path(__file__),
                args.config,
                ROOT / "scripts/prepare_ppe.py",
                ROOT / "src/object_detector/ppe_dataset.py",
            )
        },
        "working_tree_dirty": bool(
            subprocess.check_output(["git", "status", "--porcelain"], cwd=ROOT).strip()
        ),
    }
    model.train(
        data=str(args.data),
        device=args.device,
        project=str(ROOT / "runs"),
        name=args.name,
        exist_ok=False,
        **parameters,
    )
    (run / "experiment.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(f"Selected checkpoint: {run / 'weights/best.pt'}")


if __name__ == "__main__":
    main()
