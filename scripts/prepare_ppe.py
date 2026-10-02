"""Download the official PPE archive and build audited train/val/test manifests."""

import argparse
import hashlib
import json
import zipfile
from pathlib import Path

from object_detector.download import download
from object_detector.ppe_dataset import DATASET_SHA256, DATASET_URL, prepare_dataset

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT / "data/construction-ppe")
    parser.add_argument("--output", type=Path, default=ROOT / "data/ppe-audited")
    args = parser.parse_args()
    if not args.root.exists():
        archive = ROOT / "data/construction-ppe.zip"
        archive.parent.mkdir(parents=True, exist_ok=True)
        if not archive.exists() or not zipfile.is_zipfile(archive):
            if archive.exists():
                archive.replace(archive.with_suffix(archive.suffix + ".partial"))
            download(DATASET_URL, archive, DATASET_SHA256)
        if hashlib.sha256(archive.read_bytes()).hexdigest() != DATASET_SHA256:
            raise ValueError("Archive differs from the pinned Construction-PPE dataset release")
        target = args.root.resolve()
        with zipfile.ZipFile(archive) as bundle:
            for entry in bundle.infolist():
                destination = (target / entry.filename).resolve()
                if (
                    not destination.is_relative_to(target)
                    or (entry.external_attr >> 16) & 0o170000 == 0o120000
                ):
                    raise ValueError("Unsafe archive member")
            bundle.extractall(target)
        provenance = {
            "url": DATASET_URL,
            "bytes": archive.stat().st_size,
            "sha256": hashlib.sha256(archive.read_bytes()).hexdigest(),
        }
        (archive.parent / "ppe-download.json").write_text(
            json.dumps(provenance, indent=2), encoding="utf-8"
        )
    report = prepare_dataset(args.root, args.output)
    print(
        json.dumps(
            {
                "splits": {
                    s: {k: v for k, v in r.items() if k != "records"}
                    for s, r in report["splits"].items()
                },
                "excluded_similar_images": len(report["excluded"]),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
