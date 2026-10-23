from pathlib import Path

from scripts.train_model import train


def test_training_pipeline_writes_artifacts(tmp_path: Path) -> None:
    metrics = train(
        data_dir=tmp_path / "data",
        artifacts_dir=tmp_path / "artifacts",
        samples=36,
        seed=7,
        train_yolo=False,
        yolo_epochs=1,
    )

    assert (tmp_path / "artifacts" / "baseline_random_forest.joblib").exists()
    assert (tmp_path / "artifacts" / "strong_xgboost_model.joblib").exists()
    assert metrics["models"]["baseline"]["model"] == "RandomForestClassifier"
    assert "precision" in metrics["detection"]
