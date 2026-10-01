"""Download public upstream demo inputs. No user data or credentials are needed."""

import argparse
import hashlib
import json
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SOURCES = {
    "bus.jpg": "https://raw.githubusercontent.com/ultralytics/assets/main/im/bus.jpg",
    "zidane.jpg": "https://raw.githubusercontent.com/ultralytics/assets/main/im/zidane.jpg",
    "vtest.avi": "https://raw.githubusercontent.com/opencv/opencv/4.x/samples/data/vtest.avi",
}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--video", action="store_true", help="Also download OpenCV's pedestrian video"
    )
    args = parser.parse_args()
    directory = ROOT / "data"
    directory.mkdir(exist_ok=True)
    provenance = []
    for name, url in SOURCES.items():
        if name.endswith(".avi") and not args.video:
            continue
        path = directory / name
        if not path.exists():
            request = urllib.request.Request(
                url, headers={"User-Agent": "farzam-object-detector/1.0"}
            )
            with urllib.request.urlopen(request, timeout=90) as response:
                payload = response.read()
            temporary = path.with_suffix(path.suffix + ".download")
            temporary.write_bytes(payload)
            temporary.replace(path)
        provenance.append(
            {
                "file": name,
                "url": url,
                "bytes": path.stat().st_size,
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            }
        )
        print(f"Ready: {path}")
    (directory / "sources.json").write_text(json.dumps(provenance, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
