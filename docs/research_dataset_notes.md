# Research Dataset Notes

Recommended real datasets to adapt after accepting their terms:

- VinDr-CXR: chest X-ray annotations including bounding boxes.
- RSNA Pneumonia Detection Challenge: chest X-ray pneumonia bounding boxes.

Expected conversion steps:

1. Place images under `data/research/raw`.
2. Convert source annotations into YOLO text labels.
3. Create `train`, `val`, and `test` folders with `images` and `labels` subfolders.
4. Update `data.yaml`.
5. Run `python backend/scripts/train_model.py --train-yolo`.

Keep patient data governance, licensing, and institutional approval requirements outside the repo.
