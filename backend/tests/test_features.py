from pathlib import Path

import pandas as pd
from PIL import Image, ImageDraw

from app.ml.features import FEATURE_COLUMNS, build_feature_table, extract_image_features, propose_abnormal_regions


def test_extract_image_features_has_expected_schema() -> None:
    image = Image.new("L", (128, 128), color=60)
    draw = ImageDraw.Draw(image)
    draw.ellipse((50, 50, 80, 80), fill=230)

    features = extract_image_features(image)

    assert list(features.keys()) == FEATURE_COLUMNS
    assert features["bright_area_ratio"] > 0
    assert features["p99_intensity"] > features["mean_intensity"]


def test_propose_abnormal_regions_finds_bright_region() -> None:
    image = Image.new("L", (128, 128), color=55)
    draw = ImageDraw.Draw(image)
    draw.rectangle((40, 45, 72, 78), fill=235)

    boxes = propose_abnormal_regions(image)

    assert boxes
    assert boxes[0].x_min <= 45
    assert boxes[0].x_max >= 70


def test_build_feature_table_uses_metadata_paths(tmp_path: Path) -> None:
    image_dir = tmp_path / "train" / "images"
    image_dir.mkdir(parents=True)
    Image.new("L", (64, 64), color=80).save(image_dir / "sample.png")
    metadata = pd.DataFrame([{"split": "train", "file_name": "sample.png", "has_abnormality": 0}])

    x, y = build_feature_table(metadata, tmp_path)

    assert x.shape == (1, len(FEATURE_COLUMNS))
    assert y.tolist() == [0]
