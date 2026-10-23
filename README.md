# YOLO-Based Medical Image Abnormality Detection

A senior-level full-stack machine learning project that detects and localises suspected abnormalities in chest X-ray-like medical images. The project includes research-style data generation, YOLO-format bounding-box annotations, feature engineering, supervised ML model training, model artifact persistence, a FastAPI inference service, automated tests, and a React TypeScript upload interface.

> Experimental research system only. Detection results do not establish a medical diagnosis.

## Features

- Synthetic chest X-ray-like dataset generation with bounding-box annotations
- YOLO-compatible `images/`, `labels/`, and `data.yaml` dataset structure
- Clear feature matrix `X` and target vector `y`
- Supervised ML target: whether an image contains a suspected abnormality
- Baseline model: `RandomForestClassifier`
- Stronger model: `XGBClassifier`
- Optional Ultralytics YOLO training path for object detection
- Runtime abnormality probability from a saved supervised ML model
- Localised detected regions with bounding boxes and confidence scores
- Precision, recall, mAP@50, mAP@50-95, ROC AUC, average precision, and F1 evaluation
- Saved model artifacts with `joblib`
- FastAPI runtime loading of trained model artifacts
- React TypeScript interface for image upload and prediction display
- Backend tests for feature engineering, training, and API predictions
- S3-compatible storage adapter boundary

## Tech Stack

### Machine Learning

- Python
- pandas
- NumPy
- scikit-learn
- XGBoost
- joblib
- Pillow
- Optional: PyTorch, Ultralytics YOLO

### Backend

- FastAPI
- Uvicorn
- pytest
- python-multipart

### Frontend

- React
- TypeScript
- Vite
- Lucide React icons

### Storage Readiness

The demo uses local generated data and local ML artifacts for reproducible training. The backend includes a storage service boundary with local storage and an S3-compatible adapter, ready for MinIO, AWS S3, Cloudflare R2, or another S3-compatible object store.

## Project Structure

```text
.
├── backend/
│   ├── app/
│   │   ├── api/
│   │   │   └── routes.py
│   │   ├── core/
│   │   │   └── config.py
│   │   ├── main.py
│   │   ├── ml/
│   │   │   ├── artifacts/
│   │   │   ├── features.py
│   │   │   ├── outcome_model.py
│   │   │   └── prediction_model.py
│   │   └── services/
│   │       ├── detection_service.py
│   │       └── storage.py
│   ├── scripts/
│   │   ├── generate_dataset.py
│   │   └── train_model.py
│   ├── tests/
│   │   ├── test_api.py
│   │   ├── test_features.py
│   │   └── test_training.py
│   └── requirements.txt
├── docs/
│   ├── model_card.md
│   └── research_dataset_notes.md
├── frontend/
│   ├── src/
│   │   ├── components/
│   │   │   ├── ResultsPanel.tsx
│   │   │   └── UploadPanel.tsx
│   │   ├── lib/
│   │   │   └── api.ts
│   │   ├── App.tsx
│   │   ├── main.tsx
│   │   ├── styles.css
│   │   └── types.ts
│   ├── package.json
│   ├── tsconfig.json
│   └── vite.config.ts
├── .gitignore
├── pyproject.toml
└── README.md
```

## ML Design

### Dataset

The default training script generates a deterministic research-style dataset with:

- Chest X-ray-like grayscale images
- Positive images containing a synthetic suspected abnormality
- Negative images without the target abnormality
- YOLO bounding-box labels for positive examples
- Train, validation, and unseen test splits
- `data.yaml` for Ultralytics YOLO compatibility

The generated dataset is intentionally a safe, authorised teaching dataset. For serious experimentation, replace it with a licensed research dataset such as VinDr-CXR or RSNA Pneumonia Detection Challenge after accepting the dataset terms and converting annotations to YOLO format.

### Features `X`

The supervised ML model uses engineered visual features:

- Mean image intensity
- Intensity standard deviation
- 75th, 90th, and 99th percentile intensity
- Bright area ratio
- High-contrast area ratio
- Edge density
- Left/right asymmetry
- Upper/lower asymmetry
- Central brightness
- Largest bright connected component ratio

### Target `y`

`has_abnormality = 1` when the image contains a labelled suspected abnormality.

`has_abnormality = 0` when the image does not contain the target abnormality.

## Models Trained

- `RandomForestClassifier` baseline
- `XGBClassifier` stronger model
- Optional `Ultralytics YOLO` detector when `torch` and `ultralytics` are installed

Latest training run:

| Model | Accuracy | ROC AUC | Average Precision | Precision | Recall | F1 |
|---|---:|---:|---:|---:|---:|---:|
| RandomForestClassifier baseline | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| XGBClassifier stronger_model | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |

Selected runtime model: `XGBClassifier stronger_model`.

Detection metrics from the unseen test split:

| Precision | Recall | mAP@50 | mAP@50-95 |
|---:|---:|---:|---:|
| 1.0000 | 1.0000 | 1.0000 | 0.6805 |

The selected runtime artifacts are saved to:

```text
backend/app/ml/artifacts/strong_xgboost_model.joblib
backend/app/ml/artifacts/baseline_random_forest.joblib
backend/app/ml/artifacts/metrics.json
backend/app/ml/artifacts/feature_schema.json
```

## Runtime Prediction Flow

1. User uploads an image in the React frontend.
2. FastAPI receives the image through `POST /api/predict`.
3. The backend extracts visual features.
4. The saved supervised ML model predicts abnormality probability.
5. The detector proposes suspected regions and confidence scores.
6. The API returns annotated image, bounding boxes, prediction probability, feature values, and metrics.
7. The frontend displays the original image with predicted boxes and confidence scores.

If `backend/app/ml/artifacts/yolo_abnormality.pt` exists, the backend loads it with Ultralytics and uses YOLO for localisation.

## Local Setup

### 1. Backend

```bash
python -m pip install -r backend/requirements.txt
python backend/scripts/train_model.py
uvicorn app.main:app --app-dir backend --reload
```

Backend API:

```text
http://localhost:8000/api/health
```

### 2. Frontend

```bash
cd frontend
npm install
npm run dev
```

Frontend interface:

```text
http://localhost:5173
```

## API Endpoints

- `GET /api/health`
- `GET /api/metrics`
- `POST /api/predict`

The prediction response includes:

- Experimental medical disclaimer
- Image dimensions
- Detected abnormality regions
- Bounding boxes
- Confidence scores
- Runtime model name
- Model-predicted abnormality probability
- Predicted label
- Engineered feature values
- Evaluation metrics
- Base64 annotated image

## Retraining

Run:

```bash
python backend/scripts/train_model.py
```

This regenerates or loads:

- Synthetic research-style dataset
- YOLO-format labels
- Feature matrix
- Target vector
- Baseline model
- Stronger model
- Classification metrics
- Detection metrics
- Saved model artifacts
- Metrics JSON artifact

Optional YOLO training:

```bash
python backend/scripts/train_model.py --train-yolo --yolo-epochs 20
```

This requires:

- PyTorch
- Ultralytics
- Enough compute for YOLO training

## Testing

### Backend Tests

```bash
pytest
```

### Frontend Build

```bash
cd frontend
npm run build
```

## Repository

Owner: `brianmahlatini`

Repository: `YOLO-Based-Medical-Image-Abnormality-Detection`

Main branch: `main`

## Medical Safety

This project is not a medical device. It has not been clinically validated and must not be used to diagnose, treat, triage, or rule out disease. The system must remain clearly labelled as experimental wherever it is deployed.
