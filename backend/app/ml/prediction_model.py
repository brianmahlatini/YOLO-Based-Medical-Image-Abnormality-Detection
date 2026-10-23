from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import joblib
from PIL import Image

from app.core.config import get_settings
from app.ml.features import extract_image_features, features_to_frame


class ImageOutcomeModel:
    def __init__(self, artifacts_dir: Path | None = None) -> None:
        self.artifacts_dir = artifacts_dir or get_settings().artifacts_dir
        self.model_path = self.artifacts_dir / "strong_xgboost_model.joblib"
        self.baseline_path = self.artifacts_dir / "baseline_random_forest.joblib"
        self.metrics_path = self.artifacts_dir / "metrics.json"
        self.model = self._load_model()
        self.metrics = self._load_metrics()

    def _load_model(self) -> Any:
        if self.model_path.exists():
            return joblib.load(self.model_path)
        if self.baseline_path.exists():
            return joblib.load(self.baseline_path)
        return None

    def _load_metrics(self) -> dict[str, Any]:
        if self.metrics_path.exists():
            return json.loads(self.metrics_path.read_text(encoding="utf-8"))
        return {}

    @property
    def is_ready(self) -> bool:
        return self.model is not None

    def predict(self, image: Image.Image) -> dict[str, Any]:
        features = extract_image_features(image)
        frame = features_to_frame([features])
        if self.model is None:
            probability = float(min(0.95, max(0.05, features["bright_area_ratio"] * 10)))
            model_name = "feature-threshold fallback"
        else:
            if hasattr(self.model, "predict_proba"):
                probability = float(self.model.predict_proba(frame)[0][1])
            else:
                probability = float(self.model.predict(frame)[0])
            model_name = self.model.__class__.__name__

        return {
            "model_name": model_name,
            "abnormality_probability": round(probability, 4),
            "predicted_label": "abnormality_detected" if probability >= 0.5 else "no_abnormality_detected",
            "features": {key: round(value, 5) for key, value in features.items()},
        }
