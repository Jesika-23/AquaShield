"""
backend/services/inference_service.py
Thin orchestration layer around backend/utils/inference.py.
"""
from __future__ import annotations

import base64
import io
import os
from pathlib import Path
from typing import Union

import numpy as np
from PIL import Image

from backend.schemas import (
    AnomalyResponse,
    BoundingBox,
    CLASS_ID_TO_NAME,
    Detection,
    DetectResponse,
    ImageSize,
)
from backend.utils.inference import load_model, run_inference, detect_visual_anomalies


ImageInput = Union[Image.Image, np.ndarray]


# ------------------------------------------------------------------ #
# Model path (adjusted to your project layout)
# ------------------------------------------------------------------ #
# best.pt is at:
#   - backend\model\best.pt
#   - model\best.pt
# We prefer backend\model\best.pt relative to project root.
# ------------------------------------------------------------------ #

BASE_DIR = Path(__file__).resolve().parent.parent.parent  # project root
MODEL_PATH = BASE_DIR / "backend" / "model" / "best.pt"


_model = None


def get_model():
    global _model
    if _model is None:
        _model = load_model(str(MODEL_PATH))
    return _model


def _to_pil(image: ImageInput) -> Image.Image:
    if isinstance(image, Image.Image):
        return image.convert("RGB")
    if isinstance(image, np.ndarray):
        return Image.fromarray(image).convert("RGB")
    raise TypeError(f"Unsupported image type: {type(image)}")


def _annotated_to_base64(annotated: ImageInput) -> str:
    pil_img = _to_pil(annotated)
    buf = io.BytesIO()
    pil_img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("utf-8")


def _bbox_to_schema(raw_bbox) -> BoundingBox:
    x1, y1, x2, y2 = raw_bbox
    return BoundingBox(x1=float(x1), y1=float(y1), x2=float(x2), y2=float(y2))


def run_detection(
    image: ImageInput,
    file_name: str,
    conf_threshold: float = 0.25,
) -> DetectResponse:
    model = get_model()
    pil_image = _to_pil(image)
    result = run_inference(model, pil_image, conf_threshold=conf_threshold)

    detections = [
        Detection(
            class_id=int(det["class_id"]),
            class_name=str(det.get("class") or CLASS_ID_TO_NAME.get(int(det["class_id"]), "Unknown")),
            confidence=float(det["confidence"]),
            bbox=_bbox_to_schema(det["bbox"]),
        )
        for det in result["detections"]
    ]

    width, height = result["image_size"]

    return DetectResponse(
        file_name=file_name,
        conf_threshold=conf_threshold,
        image_size=ImageSize(width=int(width), height=int(height)),
        inference_time_ms=float(result["inference_time_ms"]),
        detections=detections,
        annotated_image_base64=_annotated_to_base64(result["annotated_image"]),
    )


def run_anomaly_check(
    reference_image: ImageInput,
    current_image: ImageInput,
) -> AnomalyResponse:
    ref_pil = _to_pil(reference_image)
    cur_pil = _to_pil(current_image)
    result = detect_visual_anomalies(ref_pil, cur_pil)

    if not result.get("reliable", False):
        return AnomalyResponse(
            reliable=False,
            count=0,
            warning=result.get("warning", "Anomaly comparison was not reliable for this image pair."),
            annotated_image_base64=None,
        )

    return AnomalyResponse(
        reliable=True,
        count=int(result.get("count", 0)),
        warning=None,
        annotated_image_base64=_annotated_to_base64(result["annotated_image"]),
    )