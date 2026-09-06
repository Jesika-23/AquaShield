"""
api_client.py
Calls the AquaShield FastAPI backend and reshapes the response to look
EXACTLY like the old utils.inference.run_inference() return value, so
app.py's downstream priority/quality/shadow code needs zero changes.
"""
import base64
import io
import os


import requests
from PIL import Image


BACKEND_URL = os.environ.get("AQUASHIELD_BACKEND_URL", "https://aquashield-backend-59kl.onrender.com")


def detect_via_backend(pil_image: Image.Image, filename: str, conf_threshold: float = 0.25, timeout: int = 30) -> dict:
    buf = io.BytesIO()
    pil_image.convert("RGB").save(buf, format="JPEG")
    buf.seek(0)

    try:
        response = requests.post(
            f"{BACKEND_URL}/detect",
            files={"file": (filename, buf, "image/jpeg")},
            data={"conf_threshold": conf_threshold},
            timeout=timeout,
        )
        response.raise_for_status()
    except requests.exceptions.RequestException as exc:
        raise RuntimeError(f"Could not reach detection backend at {BACKEND_URL}: {exc}")

    payload = response.json()

    detections = [
        {
            "class": det["class_name"],
            "class_id": det["class_id"],
            "confidence": det["confidence"],
            "bbox": (det["bbox"]["x1"], det["bbox"]["y1"], det["bbox"]["x2"], det["bbox"]["y2"]),
        }
        for det in payload["detections"]
    ]

    annotated_bytes = base64.b64decode(payload["annotated_image_base64"])
    annotated_image = Image.open(io.BytesIO(annotated_bytes)).convert("RGB")

    return {
        "detections": detections,
        "annotated_image": annotated_image,
        "image_size": (payload["image_size"]["width"], payload["image_size"]["height"]),
        "inference_time_ms": payload["inference_time_ms"],
    }