from functools import lru_cache
from pathlib import Path

from pydantic import BaseModel


class Settings(BaseModel):
    app_name: str = "Experimental Medical Abnormality Detection API"
    experimental_disclaimer: str = (
        "Experimental research system. Detection results do not establish a medical diagnosis."
    )
    artifacts_dir: Path = Path(__file__).resolve().parents[1] / "ml" / "artifacts"
    max_upload_mb: int = 12


@lru_cache
def get_settings() -> Settings:
    return Settings()
