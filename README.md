# YOLO-Based Medical Image Abnormality Detection

Experimental full-stack ML project for detecting and localising suspected abnormalities in chest X-ray-like images.

> This system is for research and education only. Detection results do not establish a medical diagnosis.

## What This Project Does

- Generates an authorised local research-style demo dataset with chest X-ray-like images and YOLO-format bounding-box annotations.
- Defines supervised ML inputs `X` as engineered image features and target `y` as `has_abnormality`.
- Trains a baseline `RandomForestClassifier`.
- Trains a stronger `XGBClassifier` when XGBoost is installed, otherwise `GradientBoostingClassifier`.
- Saves model artifacts with `joblib`.
- Loads the saved runtime model inside a FastAPI backend.
- Uploads an image from a React + TypeScript frontend.
- Returns abnormality probability, localized boxes, confidence scores, annotated image, and evaluation metrics.
- Provides an optional Ultralytics YOLO training path using the generated `data.yaml`.

## Project Structure

```text
backend/
  app/
    api/routes.py
    core/config.py
    ml/
      features.py
      outcome_model.py
      prediction_model.py
      artifacts/
    services/
      detection_service.py
      storage.py
    main.py
  scripts/
    generate_dataset.py
    train_model.py
  tests/
frontend/
  src/
    components/
    lib/
    App.tsx
    main.tsx
data/
  synthetic/
```

## Dataset

The default dataset is generated locally by `backend/scripts/generate_dataset.py`. It creates synthetic chest X-ray-like images with labelled bright suspected-abnormality regions and YOLO `.txt` bounding-box files.

This is intentionally not a diagnostic dataset. It is a reproducible, authorised teaching dataset that exercises the same data contracts used by a research dataset such as VinDr-CXR or RSNA pneumonia detection data after those datasets are obtained under their license terms.

## Features and Target

Input features `X` are extracted in `backend/app/ml/features.py`:

- intensity statistics: mean, standard deviation, 75th/90th/99th percentiles
- bright-region ratios
- high-contrast ratios
- edge density
- left/right and upper/lower asymmetry
- central brightness
- largest bright connected component ratio

Target `y`:

- `has_abnormality = 1` if the generated image has a bounding-box abnormality
- `has_abnormality = 0` otherwise

The object-localisation output uses YOLO-compatible labels and a localizer in the API. If `backend/app/ml/artifacts/yolo_abnormality.pt` exists, the API will load it through Ultralytics automatically.

## Train Models

```bash
python backend/scripts/train_model.py
```

Artifacts are written to:

```text
backend/app/ml/artifacts/
```

To train an Ultralytics YOLO model after installing `torch` and `ultralytics`:

```bash
python backend/scripts/train_model.py --train-yolo --yolo-epochs 20
```

## Backend

Install dependencies:

```bash
pip install -r backend/requirements.txt
```

Run API:

```bash
uvicorn app.main:app --app-dir backend --reload
```

Endpoints:

- `GET /api/health`
- `GET /api/metrics`
- `POST /api/predict`

## Frontend

```bash
cd frontend
npm install
npm run dev
```

The app uploads an image, calls `/api/predict`, and displays:

- original image with predicted bounding boxes
- detection confidence scores
- runtime model abnormality probability
- precision, recall, mAP@50, and mAP@50-95

## Tests

```bash
pytest
cd frontend
npm run build
```

## Storage

`backend/app/services/storage.py` includes local storage and an S3-compatible storage adapter. Add `boto3` and configure bucket credentials to enable S3-compatible uploads.

## Medical Safety

This project must be labelled experimental wherever deployed. It is not a medical device, has not been clinically validated, and must not be used to diagnose, treat, or rule out disease.
