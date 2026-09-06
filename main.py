from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from pydantic import BaseModel
from typing import List, Optional
import ultralytics
from ultralytics import YOLO
import cv2
import numpy as np
import os
import time
import uuid


app = FastAPI()


# ---------------------------
# Configuration
# ---------------------------
MODEL_PATH = os.path.join("backend", "model", "best.pt")
OUTPUT_DIR = "outputs"

os.makedirs(OUTPUT_DIR, exist_ok=True)


# ---------------------------
# UI class-name mapping (AquaShield narrative)
# ---------------------------
# Raw model classes: [aircraft, fish, ship, submarine]
# We display them as:    [Aircraft, Fish, Reef, Ship]
CLASS_ID_TO_NAME = {
    0: "Aircraft",
    1: "Fish",
    2: "Reef",      # raw: "ship"
    3: "Ship",      # raw: "submarine"
}


# ---------------------------
# Load model at startup
# ---------------------------
model = None


@app.on_event("startup")
def load_model():
    global model
    model = YOLO(MODEL_PATH)
    print("Model loaded successfully.")


# ---------------------------
# Pydantic models
# ---------------------------
class BBox(BaseModel):
    x1: float
    y1: float
    x2: float
    y2: float


class Detection(BaseModel):
    class_id: int
    class_name: str
    confidence: float
    bbox: BBox


class ImageSize(BaseModel):
    width: int
    height: int


class DetectResponse(BaseModel):
    file_name: str
    conf_threshold: float
    image_size: ImageSize
    inference_time_ms: float
    detections: List[Detection]
    annotated_image_path: Optional[str] = None


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    classes: List[str]


# ---------------------------
# Endpoints
# ---------------------------
@app.get("/health", response_model=HealthResponse)
def health_check():
    global model
    if model is None:
        return HealthResponse(status="error", model_loaded=False, classes=[])
    # Use our UI mapping for class names
    class_names = [CLASS_ID_TO_NAME[i] for i in range(len(CLASS_ID_TO_NAME))]
    return HealthResponse(status="ok", model_loaded=True, classes=class_names)


@app.post("/detect", response_model=DetectResponse)
def detect_objects(
    file: UploadFile = File(...),
    conf_threshold: float = Form(0.25)
):
    global model

    # Read image
    try:
        contents = file.file.read()
        nparr = np.frombuffer(contents, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if image is None:
            raise HTTPException(status_code=400, detail="Invalid image file")
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error reading image: {str(e)}")

    orig_h, orig_w = image.shape[:2]

    # Run inference
    start_time = time.time()
    results = model(image, conf=conf_threshold)
    end_time = time.time()
    inference_time_ms = (end_time - start_time) * 1000

    # Parse detections
    detections: List[Detection] = []
    result = results[0]
    boxes = result.boxes

    if boxes is not None:
        for i in range(len(boxes)):
            cls_id = int(boxes.cls[i].item())
            conf = float(boxes.conf[i].item())
            x1, y1, x2, y2 = boxes.xyxy[i].tolist()

            # Use UI-mapped class name
            class_name = CLASS_ID_TO_NAME.get(cls_id, f"Class{cls_id}")

            detections.append(
                Detection(
                    class_id=cls_id,
                    class_name=class_name,
                    confidence=conf,
                    bbox=BBox(x1=round(x1, 1), y1=round(y1, 1), x2=round(x2, 1), y2=round(y2, 1))
                )
            )

    # Draw annotations
    annotated = image.copy()
    for det in detections:
        x1, y1, x2, y2 = map(int, det.bbox.dict().values())
        label = f"{det.class_name} {det.confidence:.2f}"
        cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(annotated, label, (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # Save annotated image
    unique_id = uuid.uuid4().hex
    base_name = os.path.splitext(file.filename)[0] if file.filename else "image"
    out_filename = f"{base_name}_{unique_id}_annotated.jpg"
    out_path = os.path.join(OUTPUT_DIR, out_filename)
    cv2.imwrite(out_path, annotated)

    return DetectResponse(
        file_name=file.filename or "unknown",
        conf_threshold=conf_threshold,
        image_size=ImageSize(width=orig_w, height=orig_h),
        inference_time_ms=round(inference_time_ms, 2),
        detections=detections,
        annotated_image_path=out_path
    )