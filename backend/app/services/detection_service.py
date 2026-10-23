from __future__ import annotations

import io
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw, ImageFont

from app.core.config import get_settings
from app.ml.features import propose_abnormal_regions
from app.ml.prediction_model import ImageOutcomeModel


class DetectionService:
    def __init__(self, artifacts_dir: Path | None = None) -> None:
        self.settings = get_settings()
        self.outcome_model = ImageOutcomeModel(artifacts_dir=artifacts_dir)
        self.yolo_model = self._load_yolo_model(artifacts_dir or self.settings.artifacts_dir)

    def _load_yolo_model(self, artifacts_dir: Path) -> Any | None:
        yolo_path = artifacts_dir / "yolo_abnormality.pt"
        if not yolo_path.exists():
            return None
        try:
            from ultralytics import YOLO

            return YOLO(str(yolo_path))
        except Exception:
            return None

    def predict(self, image_bytes: bytes) -> dict[str, Any]:
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        outcome = self.outcome_model.predict(image)
        boxes = self._detect_boxes(image, outcome["abnormality_probability"])
        annotated = self._annotate(image, boxes)
        return {
            "disclaimer": self.settings.experimental_disclaimer,
            "image_size": {"width": image.width, "height": image.height},
            "detections": boxes,
            "outcome": outcome,
            "metrics": self.outcome_model.metrics,
            "annotated_image": annotated,
        }

    def _detect_boxes(self, image: Image.Image, probability: float) -> list[dict[str, Any]]:
        if self.yolo_model is not None:
            return self._detect_with_yolo(image)

        if probability < 0.5:
            return []

        detections: list[dict[str, Any]] = []
        for rank, box in enumerate(propose_abnormal_regions(image)[:3], start=1):
            confidence = max(0.05, min(0.99, probability - ((rank - 1) * 0.08)))
            detections.append(
                {
                    "label": "suspected_abnormality",
                    "confidence": round(confidence, 4),
                    "bbox": box.to_dict(),
                    "source": "supervised-risk-guided localizer",
                }
            )
        return detections

    def _detect_with_yolo(self, image: Image.Image) -> list[dict[str, Any]]:
        results = self.yolo_model(image, verbose=False)
        detections: list[dict[str, Any]] = []
        for result in results:
            names = getattr(result, "names", {})
            for box in result.boxes:
                x_min, y_min, x_max, y_max = [int(value) for value in box.xyxy[0].tolist()]
                class_id = int(box.cls[0])
                detections.append(
                    {
                        "label": names.get(class_id, "abnormality"),
                        "confidence": round(float(box.conf[0]), 4),
                        "bbox": {"x_min": x_min, "y_min": y_min, "x_max": x_max, "y_max": y_max},
                        "source": "ultralytics-yolo",
                    }
                )
        return detections

    def _annotate(self, image: Image.Image, detections: list[dict[str, Any]]) -> str:
        canvas = image.copy()
        draw = ImageDraw.Draw(canvas)
        try:
            font = ImageFont.truetype("arial.ttf", 14)
        except OSError:
            font = ImageFont.load_default()

        for detection in detections:
            box = detection["bbox"]
            label = f'{detection["label"]} {detection["confidence"]:.2f}'
            xy = [box["x_min"], box["y_min"], box["x_max"], box["y_max"]]
            draw.rectangle(xy, outline=(255, 60, 60), width=3)
            text_bbox = draw.textbbox((xy[0], max(0, xy[1] - 18)), label, font=font)
            draw.rectangle(text_bbox, fill=(255, 60, 60))
            draw.text((xy[0], max(0, xy[1] - 18)), label, fill=(255, 255, 255), font=font)

        buffer = io.BytesIO()
        canvas.save(buffer, format="PNG")
        return "data:image/png;base64," + __import__("base64").b64encode(buffer.getvalue()).decode("ascii")
