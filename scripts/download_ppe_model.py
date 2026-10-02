"""Fetch the released, fine-tuned PPE models with pinned SHA-256 verification."""

import argparse
import hashlib
import json
from pathlib import Path

from object_detector.download import download

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend", choices=["torch", "onnx", "both"], default="both")
    args = parser.parse_args()
    manifest = json.loads((ROOT / "assets/ppe/models.json").read_text(encoding="utf-8"))
    for backend, record in manifest.items():
        if args.backend not in ("both", backend):
            continue
        path = ROOT / "models" / record["file"]
        if path.exists() and hashlib.sha256(path.read_bytes()).hexdigest() == record["sha256"]:
            print(f"Verified existing {path.name}")
            continue
        download(record["url"], path, record["sha256"])
        print(f"Downloaded and verified {path.name}")


if __name__ == "__main__":
    main()
