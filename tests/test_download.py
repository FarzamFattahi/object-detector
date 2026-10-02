import hashlib
import io
from contextlib import nullcontext

import pytest

from object_detector.download import download


class Response(io.BytesIO):
    def __init__(self, body, status, headers):
        super().__init__(body)
        self.status, self.headers = status, headers


def test_truncated_body_is_resumed_and_verified(tmp_path, monkeypatch):
    payload = b"complete public model"
    calls = []

    def open_response(request, timeout):
        calls.append(request.headers["Range"])
        if len(calls) == 1:
            return Response(payload[:8], 200, {"Content-Length": str(len(payload))})
        return Response(
            payload[8:], 206, {"Content-Range": f"bytes 8-{len(payload) - 1}/{len(payload)}"}
        )

    monkeypatch.setattr("urllib.request.urlopen", open_response)
    monkeypatch.setattr("time.sleep", lambda _: None)
    path = download(
        "https://example.org/model", tmp_path / "model.pt", hashlib.sha256(payload).hexdigest()
    )
    assert path.read_bytes() == payload
    assert calls == ["bytes=0-", "bytes=8-"]
    assert not path.with_suffix(".pt.partial").exists()


def test_wrong_hash_cannot_be_published_as_completed_download(tmp_path, monkeypatch):
    response = Response(b"wrong", 200, {"Content-Length": "5"})
    monkeypatch.setattr("urllib.request.urlopen", lambda *a, **k: nullcontext(response))
    path = tmp_path / "model.pt"
    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        download("https://example.org/model", path, "0" * 64)
    assert not path.exists()
