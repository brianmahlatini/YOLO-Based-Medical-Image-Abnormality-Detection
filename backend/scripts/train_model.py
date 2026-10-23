from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Iterable

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT / "backend") not in sys.path:
    sys.path.insert(0, str(ROOT / "backend"))

from app.ml.features import FEATURE_COLUMNS, build_feature_table, propose_abnormal_regions
from scripts.generate_dataset import generate_dataset


def _load_or_generate_dataset(data_dir: Path, samples: int, seed: int) -> pd.DataFrame:
    metadata_path = data_dir / "metadata.csv"
    if metadata_path.exists():
        return pd.read_csv(metadata_path)
    return generate_dataset(data_dir, samples=samples, seed=seed)


def _train_stronger_model(seed: int):
    try:
        from xgboost import XGBClassifier

        return XGBClassifier(
            n_estimators=90,
            max_depth=3,
            learning_rate=0.08,
            subsample=0.9,
            colsample_bytree=0.9,
            eval_metric="logloss",
            random_state=seed,
        )
    except Exception:
        return GradientBoostingClassifier(random_state=seed)


def _classification_metrics(model, x: pd.DataFrame, y: pd.Series) -> dict[str, float | str]:
    probabilities = model.predict_proba(x)[:, 1]
    predictions = (probabilities >= 0.5).astype(int)
    metrics = {
        "model": model.__class__.__name__,
        "accuracy": round(float(accuracy_score(y, predictions)), 4),
        "precision": round(float(precision_score(y, predictions, zero_division=0)), 4),
        "recall": round(float(recall_score(y, predictions, zero_division=0)), 4),
        "f1": round(float(f1_score(y, predictions, zero_division=0)), 4),
        "average_precision": round(float(average_precision_score(y, probabilities)), 4),
    }
    if len(set(y)) > 1:
        metrics["roc_auc"] = round(float(roc_auc_score(y, probabilities)), 4)
    return metrics


def _parse_yolo_boxes(label_path: Path, image_size: int = 256) -> list[tuple[int, int, int, int]]:
    boxes: list[tuple[int, int, int, int]] = []
    if not label_path.exists():
        return boxes
    for line in label_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        _, x_center, y_center, width, height = [float(part) for part in line.split()]
        x_min = int((x_center - width / 2) * image_size)
        y_min = int((y_center - height / 2) * image_size)
        x_max = int((x_center + width / 2) * image_size)
        y_max = int((y_center + height / 2) * image_size)
        boxes.append((x_min, y_min, x_max, y_max))
    return boxes


def _iou(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    intersection = max(0, ix2 - ix1) * max(0, iy2 - iy1)
    area_a = max(0, ax2 - ax1) * max(0, ay2 - ay1)
    area_b = max(0, bx2 - bx1) * max(0, by2 - by1)
    union = area_a + area_b - intersection
    return intersection / union if union else 0.0


def _ap_from_scores(scores: Iterable[tuple[float, int]]) -> float:
    ordered = sorted(scores, key=lambda item: item[0], reverse=True)
    positives = sum(label for _, label in ordered)
    if positives == 0:
        return 0.0
    tp = 0
    fp = 0
    precisions: list[float] = []
    recalls: list[float] = []
    for _, label in ordered:
        if label:
            tp += 1
        else:
            fp += 1
        precisions.append(tp / max(1, tp + fp))
        recalls.append(tp / positives)

    ap = 0.0
    previous_recall = 0.0
    for precision, recall in zip(precisions, recalls):
        ap += precision * max(0.0, recall - previous_recall)
        previous_recall = recall
    return ap


def _detection_metrics(metadata: pd.DataFrame, data_dir: Path, model) -> dict[str, float]:
    test_rows = metadata[metadata["split"] == "test"]
    thresholds = [round(value, 2) for value in np.arange(0.5, 1.0, 0.05)]
    matched_at_50 = 0
    predicted = 0
    ground_truth = 0
    map_scores: dict[float, list[tuple[float, int]]] = {threshold: [] for threshold in thresholds}

    for _, row in test_rows.iterrows():
        image_path = data_dir / row["split"] / "images" / row["file_name"]
        label_path = data_dir / row["split"] / "labels" / row["file_name"].replace(".png", ".txt")
        gt_boxes = _parse_yolo_boxes(label_path)
        proposed = propose_abnormal_regions(image_path)
        features, _ = build_feature_table(pd.DataFrame([row.to_dict()]), data_dir)
        base_score = float(model.predict_proba(features)[:, 1][0])
        pred_boxes = [(box.x_min, box.y_min, box.x_max, box.y_max) for box in proposed[:1]] if base_score >= 0.5 else []

        ground_truth += len(gt_boxes)
        predicted += len(pred_boxes)
        used_gt: set[int] = set()
        for rank, pred in enumerate(pred_boxes):
            score = max(0.01, base_score - (rank * 0.08))
            best_iou = 0.0
            best_idx = -1
            for gt_idx, gt in enumerate(gt_boxes):
                if gt_idx in used_gt:
                    continue
                overlap = _iou(pred, gt)
                if overlap > best_iou:
                    best_iou = overlap
                    best_idx = gt_idx
            if best_iou >= 0.5 and best_idx >= 0:
                matched_at_50 += 1
                used_gt.add(best_idx)
            for threshold in thresholds:
                map_scores[threshold].append((score, int(best_iou >= threshold)))

        if not pred_boxes and gt_boxes:
            for threshold in thresholds:
                map_scores[threshold].append((0.0, 0))

    precision = matched_at_50 / predicted if predicted else 0.0
    recall = matched_at_50 / ground_truth if ground_truth else 0.0
    ap_by_threshold = {threshold: _ap_from_scores(scores) for threshold, scores in map_scores.items()}
    return {
        "precision": round(float(precision), 4),
        "recall": round(float(recall), 4),
        "mAP@50": round(float(ap_by_threshold[0.5]), 4),
        "mAP@50-95": round(float(np.mean(list(ap_by_threshold.values()))), 4),
    }


def _maybe_train_yolo(data_dir: Path, artifacts_dir: Path, epochs: int) -> dict[str, object]:
    try:
        from ultralytics import YOLO
    except Exception as exc:
        return {"status": "skipped", "reason": f"Ultralytics is not installed: {exc}"}

    model = YOLO("yolov8n.pt")
    results = model.train(data=str(data_dir / "data.yaml"), epochs=epochs, imgsz=256, project=str(artifacts_dir), name="yolo_run")
    weights = Path(results.save_dir) / "weights" / "best.pt"
    target = artifacts_dir / "yolo_abnormality.pt"
    if weights.exists():
        target.write_bytes(weights.read_bytes())
    return {"status": "trained", "weights": str(target)}


def train(data_dir: Path, artifacts_dir: Path, samples: int, seed: int, train_yolo: bool, yolo_epochs: int) -> dict[str, object]:
    metadata = _load_or_generate_dataset(data_dir, samples=samples, seed=seed)
    artifacts_dir.mkdir(parents=True, exist_ok=True)

    train_metadata = metadata[metadata["split"] == "train"]
    test_metadata = metadata[metadata["split"] == "test"]
    x_train, y_train = build_feature_table(train_metadata, data_dir)
    x_test, y_test = build_feature_table(test_metadata, data_dir)

    baseline = RandomForestClassifier(n_estimators=180, max_depth=8, class_weight="balanced", random_state=seed)
    baseline.fit(x_train, y_train)

    stronger = _train_stronger_model(seed)
    stronger.fit(x_train, y_train)

    baseline_metrics = _classification_metrics(baseline, x_test, y_test)
    stronger_metrics = _classification_metrics(stronger, x_test, y_test)
    selected_model = stronger if stronger_metrics["average_precision"] >= baseline_metrics["average_precision"] else baseline
    detection_metrics = _detection_metrics(metadata, data_dir, selected_model)
    yolo_result = _maybe_train_yolo(data_dir, artifacts_dir, yolo_epochs) if train_yolo else {
        "status": "not_requested",
        "reason": "Run backend/scripts/train_model.py --train-yolo after installing ultralytics and torch.",
    }

    joblib.dump(baseline, artifacts_dir / "baseline_random_forest.joblib")
    joblib.dump(stronger, artifacts_dir / "strong_xgboost_model.joblib")
    (artifacts_dir / "feature_schema.json").write_text(json.dumps({"features": FEATURE_COLUMNS}, indent=2), encoding="utf-8")

    metrics: dict[str, object] = {
        "dataset": {
            "name": "synthetic_vindr_style_demo",
            "task": "suspected lung abnormality detection in chest X-ray-like images",
            "samples": int(len(metadata)),
            "train": int((metadata["split"] == "train").sum()),
            "validation": int((metadata["split"] == "val").sum()),
            "test": int((metadata["split"] == "test").sum()),
            "target": "has_abnormality",
        },
        "features": FEATURE_COLUMNS,
        "models": {
            "baseline": baseline_metrics,
            "stronger": stronger_metrics,
            "selected_runtime_model": selected_model.__class__.__name__,
            "yolo": yolo_result,
        },
        "detection": detection_metrics,
    }
    (artifacts_dir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=ROOT / "data" / "synthetic")
    parser.add_argument("--artifacts-dir", type=Path, default=ROOT / "backend" / "app" / "ml" / "artifacts")
    parser.add_argument("--samples", type=int, default=180)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--train-yolo", action="store_true")
    parser.add_argument("--yolo-epochs", type=int, default=5)
    args = parser.parse_args()
    metrics = train(args.data_dir, args.artifacts_dir, args.samples, args.seed, args.train_yolo, args.yolo_epochs)
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
