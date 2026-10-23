# Experimental Model Card

## Intended Use

Education and research demonstration of a full-stack computer-vision ML workflow for medical-image abnormality localisation.

## Not Intended For

Clinical diagnosis, triage, treatment planning, or ruling out disease.

## Models

- Baseline: RandomForestClassifier
- Stronger model: XGBClassifier when available, GradientBoostingClassifier fallback
- Object detection path: YOLO-format labels with optional Ultralytics YOLO training

## Data

The included generator creates synthetic chest X-ray-like images and bounding boxes. Replace this with a properly licensed research dataset before serious experimentation.

## Known Limitations

- Synthetic data is not representative of clinical acquisition conditions.
- Localisation fallback is not a substitute for a trained YOLO detector.
- Metrics from the generated demo dataset do not estimate clinical performance.
