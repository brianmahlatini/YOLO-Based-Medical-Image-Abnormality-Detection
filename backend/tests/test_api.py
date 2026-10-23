import io
from pathlib import Path

from fastapi.testclient import TestClient
from PIL import Image, ImageDraw

from app.main import app


def _demo_png() -> bytes:
    image = Image.new("RGB", (160, 160), color=(52, 52, 52))
    draw = ImageDraw.Draw(image)
    draw.ellipse((62, 55, 98, 92), fill=(230, 230, 230))
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return buffer.getvalue()


def test_health() -> None:
    client = TestClient(app)
    response = client.get("/api/health")

    assert response.status_code == 200
    assert response.json()["experimental"] is True


def test_predict_returns_detections_and_prediction() -> None:
    client = TestClient(app)
    response = client.post(
        "/api/predict",
        files={"file": ("demo.png", _demo_png(), "image/png")},
    )

    assert response.status_code == 200
    payload = response.json()
    assert "Detection results do not establish a medical diagnosis" in payload["disclaimer"]
    assert "outcome" in payload
    assert "abnormality_probability" in payload["outcome"]
    assert payload["annotated_image"].startswith("data:image/png;base64,")
