from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
from PIL import Image, ImageFilter


FEATURE_COLUMNS = [
    "mean_intensity",
    "std_intensity",
    "p75_intensity",
    "p90_intensity",
    "p99_intensity",
    "bright_area_ratio",
    "high_contrast_ratio",
    "edge_density",
    "left_right_asymmetry",
    "upper_lower_asymmetry",
    "central_brightness",
    "max_component_area_ratio",
]


@dataclass(frozen=True)
class BoundingBox:
    x_min: int
    y_min: int
    x_max: int
    y_max: int

    @property
    def width(self) -> int:
        return max(0, self.x_max - self.x_min)

    @property
    def height(self) -> int:
        return max(0, self.y_max - self.y_min)

    @property
    def area(self) -> int:
        return self.width * self.height

    def to_dict(self) -> dict[str, int]:
        return {
            "x_min": self.x_min,
            "y_min": self.y_min,
            "x_max": self.x_max,
            "y_max": self.y_max,
        }


def load_grayscale_image(image: Image.Image | str | Path) -> Image.Image:
    if isinstance(image, (str, Path)):
        return Image.open(image).convert("L")
    return image.convert("L")


def image_to_array(image: Image.Image | str | Path, size: tuple[int, int] = (256, 256)) -> np.ndarray:
    gray = load_grayscale_image(image).resize(size)
    return np.asarray(gray, dtype=np.float32) / 255.0


def _edge_density(gray: np.ndarray) -> float:
    gy, gx = np.gradient(gray)
    mag = np.sqrt((gx * gx) + (gy * gy))
    return float((mag > 0.08).mean())


def _largest_component_area(mask: np.ndarray) -> int:
    visited = np.zeros(mask.shape, dtype=bool)
    best = 0
    height, width = mask.shape

    for y in range(height):
        for x in range(width):
            if not mask[y, x] or visited[y, x]:
                continue
            stack = [(x, y)]
            visited[y, x] = True
            area = 0
            while stack:
                cx, cy = stack.pop()
                area += 1
                for nx, ny in ((cx + 1, cy), (cx - 1, cy), (cx, cy + 1), (cx, cy - 1)):
                    if 0 <= nx < width and 0 <= ny < height and mask[ny, nx] and not visited[ny, nx]:
                        visited[ny, nx] = True
                        stack.append((nx, ny))
            best = max(best, area)
    return best


def extract_image_features(image: Image.Image | str | Path) -> dict[str, float]:
    gray = image_to_array(image)
    blurred = np.asarray(load_grayscale_image(image).resize((256, 256)).filter(ImageFilter.GaussianBlur(1.2)), dtype=np.float32) / 255.0
    p75, p90, p99 = np.percentile(gray, [75, 90, 99])
    bright_mask = blurred > max(float(p90), 0.62)
    high_contrast_mask = gray > (gray.mean() + (1.35 * gray.std()))
    h, w = gray.shape
    central = gray[h // 4 : 3 * h // 4, w // 4 : 3 * w // 4]

    features = {
        "mean_intensity": float(gray.mean()),
        "std_intensity": float(gray.std()),
        "p75_intensity": float(p75),
        "p90_intensity": float(p90),
        "p99_intensity": float(p99),
        "bright_area_ratio": float(bright_mask.mean()),
        "high_contrast_ratio": float(high_contrast_mask.mean()),
        "edge_density": _edge_density(gray),
        "left_right_asymmetry": float(abs(gray[:, : w // 2].mean() - gray[:, w // 2 :].mean())),
        "upper_lower_asymmetry": float(abs(gray[: h // 2, :].mean() - gray[h // 2 :, :].mean())),
        "central_brightness": float(central.mean()),
        "max_component_area_ratio": float(_largest_component_area(bright_mask) / bright_mask.size),
    }
    return features


def features_to_frame(rows: Iterable[dict[str, float]]) -> pd.DataFrame:
    frame = pd.DataFrame(rows)
    return frame.reindex(columns=FEATURE_COLUMNS, fill_value=0.0)


def build_feature_table(metadata: pd.DataFrame, image_root: Path) -> tuple[pd.DataFrame, pd.Series]:
    rows: list[dict[str, float]] = []
    targets: list[int] = []
    for row in metadata.itertuples(index=False):
        rows.append(extract_image_features(image_root / row.split / "images" / row.file_name))
        targets.append(int(row.has_abnormality))
    return features_to_frame(rows), pd.Series(targets, name="has_abnormality")


def propose_abnormal_regions(image: Image.Image | str | Path, min_area_ratio: float = 0.0015) -> list[BoundingBox]:
    gray_image = load_grayscale_image(image)
    original_width, original_height = gray_image.size
    gray = image_to_array(gray_image)
    threshold = max(float(np.percentile(gray, 96)), float(gray.mean() + (1.55 * gray.std())), 0.58)
    mask = gray > threshold
    visited = np.zeros(mask.shape, dtype=bool)
    height, width = mask.shape
    boxes: list[BoundingBox] = []
    min_area = int(mask.size * min_area_ratio)

    for y in range(height):
        for x in range(width):
            if not mask[y, x] or visited[y, x]:
                continue
            stack = [(x, y)]
            visited[y, x] = True
            xs: list[int] = []
            ys: list[int] = []
            while stack:
                cx, cy = stack.pop()
                xs.append(cx)
                ys.append(cy)
                for nx, ny in ((cx + 1, cy), (cx - 1, cy), (cx, cy + 1), (cx, cy - 1)):
                    if 0 <= nx < width and 0 <= ny < height and mask[ny, nx] and not visited[ny, nx]:
                        visited[ny, nx] = True
                        stack.append((nx, ny))

            if len(xs) < min_area:
                continue
            x_scale = original_width / width
            y_scale = original_height / height
            boxes.append(
                BoundingBox(
                    x_min=max(0, int(min(xs) * x_scale)),
                    y_min=max(0, int(min(ys) * y_scale)),
                    x_max=min(original_width, int((max(xs) + 1) * x_scale)),
                    y_max=min(original_height, int((max(ys) + 1) * y_scale)),
                )
            )

    return sorted(boxes, key=lambda box: box.area, reverse=True)[:5]
