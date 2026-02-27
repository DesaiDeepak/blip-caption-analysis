import io
import os
import pytest
from unittest.mock import MagicMock
from fastapi.testclient import TestClient
from PIL import Image

# Set test API key BEFORE importing the app
os.environ["API_KEY"] = "test-api-key"

from api.inference import app
import api.inference as inference_module

# Inject mock model so tests don't load the 1GB BLIP model from disk
_mock_processor = MagicMock()
_mock_model = MagicMock()
_mock_model.generate.return_value = [[1, 2, 3]]
_mock_processor.decode.return_value = "a dog sitting on a bench"

inference_module.model = _mock_model
inference_module.processor = _mock_processor
inference_module.device = "cpu"

client = TestClient(app)
HEADERS = {"X-API-Key": "test-api-key"}


def _make_jpeg() -> bytes:
    buf = io.BytesIO()
    Image.new("RGB", (64, 64), color=(100, 150, 200)).save(buf, format="JPEG")
    return buf.getvalue()


def test_health_ok():
    resp = client.get("/health")
    assert resp.status_code == 200
    assert resp.json()["status"] == "ok"
    assert resp.json()["model_loaded"] is True


def test_caption_returns_text():
    resp = client.post(
        "/caption",
        files={"file": ("photo.jpg", _make_jpeg(), "image/jpeg")},
        headers=HEADERS,
    )
    assert resp.status_code == 200
    assert "caption" in resp.json()


def test_caption_wrong_api_key():
    resp = client.post(
        "/caption",
        files={"file": ("photo.jpg", _make_jpeg(), "image/jpeg")},
        headers={"X-API-Key": "wrong-key"},
    )
    assert resp.status_code == 403


def test_caption_no_api_key():
    resp = client.post(
        "/caption",
        files={"file": ("photo.jpg", _make_jpeg(), "image/jpeg")},
    )
    assert resp.status_code == 401  # APIKeyHeader returns 401 when header is missing


def test_caption_unsupported_type():
    resp = client.post(
        "/caption",
        files={"file": ("image.gif", b"GIF89a\x01\x00\x01\x00", "image/gif")},
        headers=HEADERS,
    )
    assert resp.status_code == 400


def test_caption_oversized_file():
    big = b"x" * (6 * 1024 * 1024)  # 6MB — limit is 5MB
    resp = client.post(
        "/caption",
        files={"file": ("big.jpg", big, "image/jpeg")},
        headers=HEADERS,
    )
    assert resp.status_code == 413


def test_caption_corrupt_image():
    resp = client.post(
        "/caption",
        files={"file": ("bad.jpg", b"this is not a real image", "image/jpeg")},
        headers=HEADERS,
    )
    assert resp.status_code == 400
