"""
backend/routes/anomaly.py
POST /anomaly -- compare a reference image against a current image.
"""
import io

from fastapi import APIRouter, File, HTTPException, UploadFile
from PIL import Image, UnidentifiedImageError

from backend.schemas import AnomalyResponse
from backend.services.inference_service import run_anomaly_check

router = APIRouter(tags=["Anomaly"])


def _read_image(contents: bytes, label: str) -> Image.Image:
    if not contents:
        raise HTTPException(status_code=400, detail=f"{label} file is empty.")
    try:
        image = Image.open(io.BytesIO(contents))
        image.load()
        return image
    except UnidentifiedImageError:
        raise HTTPException(
            status_code=400,
            detail=f"{label} could not be read as an image. Upload a valid JPG or PNG.",
        )
    except Exception:
        raise HTTPException(status_code=400, detail=f"{label} appears corrupted or unsupported.")


@router.post("/anomaly", response_model=AnomalyResponse)
async def anomaly(
    reference_file: UploadFile = File(..., description="Baseline/reference sonar image."),
    current_file: UploadFile = File(..., description="Current sonar image to compare."),
):
    reference_contents = await reference_file.read()
    current_contents = await current_file.read()

    reference_image = _read_image(reference_contents, "Reference image")
    current_image = _read_image(current_contents, "Current image")

    try:
        result = run_anomaly_check(reference_image, current_image)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Anomaly comparison failed: {exc}")

    return result
