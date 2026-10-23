from __future__ import annotations

import argparse
import math
import random
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFilter


CLASSES = ["suspected_abnormality"]


def _base_xray(size: int, rng: random.Random) -> Image.Image:
    y = np.linspace(-1.0, 1.0, size)[:, None]
    x = np.linspace(-1.0, 1.0, size)[None, :]
    lungs = np.exp(-(((x - 0.35) / 0.28) ** 2 + ((y + 0.05) / 0.62) ** 2))
    lungs += np.exp(-(((x + 0.35) / 0.28) ** 2 + ((y + 0.05) / 0.62) ** 2))
    ribs = 0.035 * np.sin((y + 1.0) * math.pi * 10)
    heart = 0.18 * np.exp(-(((x + 0.02) / 0.34) ** 2 + ((y - 0.38) / 0.34) ** 2))
    image = 0.28 + (0.36 * lungs) + ribs + heart
    image += rng.uniform(-0.025, 0.025) + np.random.default_rng(rng.randint(0, 999999)).normal(0, 0.025, (size, size))
    return Image.fromarray(np.uint8(np.clip(image, 0, 1) * 255), mode="L").filter(ImageFilter.GaussianBlur(0.7))


def _add_abnormality(image: Image.Image, rng: random.Random) -> tuple[Image.Image, tuple[int, int, int, int]]:
    canvas = image.convert("L")
    draw = ImageDraw.Draw(canvas, "L")
    size = canvas.size[0]
    radius_x = rng.randint(size // 18, size // 9)
    radius_y = rng.randint(size // 20, size // 8)
    cx = rng.randint(size // 4, 3 * size // 4)
    cy = rng.randint(size // 4, 3 * size // 4)
    box = (cx - radius_x, cy - radius_y, cx + radius_x, cy + radius_y)
    intensity = rng.randint(205, 245)
    draw.ellipse(box, fill=intensity)
    return canvas.filter(ImageFilter.GaussianBlur(rng.uniform(1.0, 2.4))), box


def _yolo_line(box: tuple[int, int, int, int], size: int) -> str:
    x_min, y_min, x_max, y_max = box
    x_center = ((x_min + x_max) / 2) / size
    y_center = ((y_min + y_max) / 2) / size
    width = (x_max - x_min) / size
    height = (y_max - y_min) / size
    return f"0 {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}"


def generate_dataset(output_dir: Path, samples: int = 180, image_size: int = 256, seed: int = 42) -> pd.DataFrame:
    rng = random.Random(seed)
    output_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, object]] = []
    splits = [("train", 0.72), ("val", 0.14), ("test", 0.14)]
    split_names = [name for name, ratio in splits for _ in range(round(ratio * samples))]
    split_names = (split_names + ["train"] * samples)[:samples]
    rng.shuffle(split_names)

    for split in {name for name, _ in splits}:
        (output_dir / split / "images").mkdir(parents=True, exist_ok=True)
        (output_dir / split / "labels").mkdir(parents=True, exist_ok=True)

    for idx, split in enumerate(split_names):
        has_abnormality = int(rng.random() < 0.58)
        image = _base_xray(image_size, rng)
        boxes: list[tuple[int, int, int, int]] = []
        if has_abnormality:
            image, box = _add_abnormality(image, rng)
            boxes.append(box)

        file_name = f"cxr_{idx:04d}.png"
        label_name = file_name.replace(".png", ".txt")
        image.save(output_dir / split / "images" / file_name)
        (output_dir / split / "labels" / label_name).write_text(
            "\n".join(_yolo_line(box, image_size) for box in boxes),
            encoding="utf-8",
        )
        rows.append(
            {
                "file_name": file_name,
                "split": split,
                "has_abnormality": has_abnormality,
                "boxes": len(boxes),
                "dataset": "synthetic_vindr_style_demo",
            }
        )

    metadata = pd.DataFrame(rows)
    metadata.to_csv(output_dir / "metadata.csv", index=False)
    (output_dir / "data.yaml").write_text(
        "\n".join(
            [
                f"path: {output_dir.resolve().as_posix()}",
                "train: train/images",
                "val: val/images",
                "test: test/images",
                "names:",
                "  0: suspected_abnormality",
            ]
        ),
        encoding="utf-8",
    )
    return metadata


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=Path("data/synthetic"))
    parser.add_argument("--samples", type=int, default=180)
    parser.add_argument("--image-size", type=int, default=256)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    metadata = generate_dataset(args.output_dir, args.samples, args.image_size, args.seed)
    print(f"Generated {len(metadata)} images at {args.output_dir}")


if __name__ == "__main__":
    main()
