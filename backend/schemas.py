"""
backend/schemas.py
Pydantic request/response models for the AquaShield API.
"""
from __future__ import annotations

from enum import IntEnum
from typing import List, Optional

from pydantic import BaseModel, Field


class DetectionClass(IntEnum):
    AIRCRAFT = 0
    FISH = 1
    REEF = 2
    SHIP = 3


CLASS_ID_TO_NAME = {
    DetectionClass.AIRCRAFT: "Aircraft",
    DetectionClass.FISH: "Fish",
    DetectionClass.REEF: "Reef",
    DetectionClass.SHIP: "Ship",
}


class BoundingBox(BaseModel):
    x1: float
    y1: float
    x2: float
    y2: float


class ImageSize(BaseModel):
    width: int
    height: int


class Detection(BaseModel):
    class_id: int
    class_name: str
    confidence: float
    bbox: BoundingBox


class DetectResponse(BaseModel):
    file_name: str
    conf_threshold: float
    image_size: ImageSize
    inference_time_ms: float
    detections: List[Detection]
    annotated_image_base64: str


class AnomalyResponse(BaseModel):
    reliable: bool
    count: int = 0
    warning: Optional[str] = None
    annotated_image_base64: Optional[str] = None


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    classes: List[str]
