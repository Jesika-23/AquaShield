"""
backend/routes/detect.py
POST /detect -- upload a sonar image, run YOLO detection, return structured JSON.
"""
import io

from fastapi import APIRouter, File, Form, HTTPException, UploadFile
from PIL import Image, UnidentifiedImageError

from backend.schemas import DetectResponse
from backend.services.inference_service import run_detection

router = APIRouter(tags=["Detection"])


@router.post("/detect", response_model=DetectResponse)
async def detect(
    file: UploadFile = File(..., description="Sonar image (JPG/PNG)."),
    conf_threshold: float = Form(
        0.25, ge=0.05, le=0.95, description="Confidence threshold, 0.05-0.95."
    ),
):
    contents = await file.read()
    if not contents:
        raise HTTPException(status_code=400, detail="Uploaded file is empty.")

    try:
        image = Image.open(io.BytesIO(contents))
        image.load()
    except UnidentifiedImageError:
        raise HTTPException(
            status_code=400,
            detail="File could not be read as an image. Upload a valid JPG or PNG.",
        )
    except Exception:
        raise HTTPException(status_code=400, detail="File appears corrupted or unsupported.")

    try:
        result = run_detection(
            image, file_name=file.filename or "upload.jpg", conf_threshold=conf_threshold
        )
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Inference failed: {exc}")

    return result
