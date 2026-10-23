from __future__ import annotations

from fastapi import APIRouter, File, HTTPException, UploadFile

from app.core.config import get_settings
from app.services.detection_service import DetectionService

router = APIRouter()
settings = get_settings()
service = DetectionService()


@router.get("/health")
def health() -> dict[str, object]:
    return {
        "status": "ok",
        "experimental": True,
        "model_ready": service.outcome_model.is_ready,
        "disclaimer": settings.experimental_disclaimer,
    }


@router.get("/metrics")
def metrics() -> dict[str, object]:
    return service.outcome_model.metrics


@router.post("/predict")
async def predict(file: UploadFile = File(...)) -> dict[str, object]:
    if file.content_type and not file.content_type.startswith("image/"):
        raise HTTPException(status_code=415, detail="Upload must be an image file.")

    content = await file.read()
    if len(content) > settings.max_upload_mb * 1024 * 1024:
        raise HTTPException(status_code=413, detail="Image exceeds upload size limit.")

    try:
        return service.predict(content)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Could not process image: {exc}") from exc
