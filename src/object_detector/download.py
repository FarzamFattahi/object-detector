"""Length-checked, resumable downloads with optional pinned SHA-256 verification."""

import hashlib
import time
import urllib.request
from pathlib import Path


def download(url: str, destination: Path, sha256: str | None = None) -> Path:
    destination.parent.mkdir(parents=True, exist_ok=True)
    partial = destination.with_suffix(destination.suffix + ".partial")
    for attempt in range(30):
        offset = partial.stat().st_size if partial.exists() else 0
        request = urllib.request.Request(
            url, headers={"Range": f"bytes={offset}-", "User-Agent": "farzam-object-detector/1.1"}
        )
        try:
            with urllib.request.urlopen(request, timeout=120) as response:
                if response.status == 206:
                    content_range = response.headers.get("Content-Range", "")
                    if not content_range.startswith(f"bytes {offset}-"):
                        raise OSError("Server returned an incorrect byte range")
                    total = int(content_range.split("/")[-1])
                    mode = "ab"
                elif response.status == 200:
                    total = int(response.headers["Content-Length"])
                    mode = "wb"
                else:
                    raise OSError(f"Unexpected response: {response.status}")
                with partial.open(mode) as stream:
                    while chunk := response.read(1024 * 1024):
                        stream.write(chunk)
            if partial.stat().st_size == total:
                break
            print(f"Resuming {destination.name}: {partial.stat().st_size}/{total}", flush=True)
        except OSError as error:
            print(f"Download retry {attempt + 1}: {error}", flush=True)
        time.sleep(min(attempt + 1, 5))
    else:
        raise OSError(f"Incomplete download: {url}")
    if sha256:
        hasher = hashlib.sha256()
        with partial.open("rb") as stream:
            while chunk := stream.read(1024 * 1024):
                hasher.update(chunk)
        digest = hasher.hexdigest()
        if digest != sha256:
            partial.unlink()
            raise ValueError(f"SHA-256 mismatch for {destination.name}: {digest}")
    partial.replace(destination)
    return destination
