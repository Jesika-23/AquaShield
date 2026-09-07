"""
inference.py
Wraps the trained Ultralytics YOLO11n model (model/best.pt).

UI-displayed class mapping used by this project:
0 = Aircraft
1 = Fish
2 = Reef        <-- deliberate UI relabel; trained/raw class is "ship"
3 = Ship        <-- deliberate UI relabel; trained/raw class is "submarine"

NOTE: The model itself, and data.yaml, use the raw trained names
(aircraft / fish / ship / submarine). "Reef" and "Ship" as shown here
are presentation-layer choices for the AquaShield narrative -- they
do not change what the model detected. See CLASS_NAMES below.
"""

from pathlib import Path
import time

import cv2
import numpy as np
import streamlit as st
from PIL import Image
from ultralytics import YOLO


# ------------------------------------------------------------------ #
# Model path
# ------------------------------------------------------------------ #

MODEL_PATH = Path(__file__).resolve().parent.parent / "model" / "best.pt"


# ------------------------------------------------------------------ #
# Class colors
# ------------------------------------------------------------------ #

CLASS_COLORS_BGR = {
    "aircraft": (32, 176, 255),
    "fish": (191, 212, 45),
    "reef": (250, 149, 167),
    "ship": (248, 189, 56),
}

DEFAULT_COLOR_BGR = (214, 234, 94)


# ------------------------------------------------------------------ #
# UI display-name mapping (deliberate project decision)
# ------------------------------------------------------------------ #
# Class 2 is trained/labeled as "ship" in best.pt / data.yaml, and is
# intentionally displayed here as "Reef".
# Class 3 is trained/labeled as "submarine", and is intentionally
# displayed here as "Ship".
# Both relabels align the demo with the AquaShield narrative. This
# dict is a display layer only -- the model's internal names and
# label files are untouched (see best.pt / data.yaml for the raw
# trained mapping).

CLASS_NAMES = {
    0: "Aircraft",
    1: "Fish",
    2: "Reef",
    3: "Ship",
}


# ------------------------------------------------------------------ #
# Load model
# ------------------------------------------------------------------ #

@st.cache_resource(show_spinner=False)
def load_model(model_path: str = str(MODEL_PATH)) -> YOLO:
    """Load the trained YOLO11n weights once per session."""

    if not Path(model_path).exists():
        raise FileNotFoundError(
            f"Model weights not found at '{model_path}'. "
            "Place your trained best.pt inside the model/ folder."
        )

    return YOLO(model_path)


# ------------------------------------------------------------------ #
# Draw detections
# ------------------------------------------------------------------ #

def _draw_detections(
    image_bgr: np.ndarray,
    boxes,
    names
) -> np.ndarray:
    """
    Draw bounding boxes and class names on the image.
    """

    annotated = image_bgr.copy()

    for box in boxes:

        # Bounding box coordinates
        xyxy = box.xyxy[0].cpu().numpy().astype(int)

        # Class ID
        cls_id = int(box.cls[0])

        # Confidence
        conf = float(box.conf[0])

        # Get our verified class name
        label_name = names.get(cls_id, "Unknown")

        # Color
        color = CLASS_COLORS_BGR.get(
            label_name.lower(),
            DEFAULT_COLOR_BGR
        )

        x1, y1, x2, y2 = xyxy

        # ---------------------------------------------------------- #
        # Bounding box
        # ---------------------------------------------------------- #

        cv2.rectangle(
            annotated,
            (x1, y1),
            (x2, y2),
            color,
            2
        )

        # ---------------------------------------------------------- #
        # Label
        # ---------------------------------------------------------- #

        label = f"{label_name.upper()} {conf * 100:.1f}%"

        (tw, th), baseline = cv2.getTextSize(
            label,
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            2
        )

        # Keep label inside image
        ty1 = max(
            y1 - th - baseline - 6,
            0
        )

        # Label background
        cv2.rectangle(
            annotated,
            (x1, ty1),
            (
                x1 + tw + 8,
                ty1 + th + baseline + 6
            ),
            color,
            -1
        )

        # Label text
        cv2.putText(
            annotated,
            label,
            (x1 + 4, ty1 + th + 2),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (10, 15, 20),
            2,
            cv2.LINE_AA
        )

    return annotated


# ------------------------------------------------------------------ #
# Run inference
# ------------------------------------------------------------------ #

def run_inference(
    model: YOLO,
    pil_image: Image.Image,
    conf_threshold: float = 0.25
):
    """
    Runs YOLO11n detection on a PIL image.

    Returns:
        annotated_image
        detections
        inference_time_ms
        image_size
    """

    # -------------------------------------------------------------- #
    # Convert PIL -> RGB -> BGR
    # -------------------------------------------------------------- #

    rgb = np.array(
        pil_image.convert("RGB")
    )

    bgr = cv2.cvtColor(
        rgb,
        cv2.COLOR_RGB2BGR
    )

    # -------------------------------------------------------------- #
    # Run YOLO
    # -------------------------------------------------------------- #

    start = time.perf_counter()

    results = model.predict(
        source=bgr,
        conf=conf_threshold,
        verbose=False
    )

    elapsed_ms = (
        time.perf_counter() - start
    ) * 1000

    result = results[0]

    # -------------------------------------------------------------- #
    # Use our verified dataset class mapping
    # -------------------------------------------------------------- #

    names = CLASS_NAMES

    # -------------------------------------------------------------- #
    # Collect detections
    # -------------------------------------------------------------- #

    detections = []

    for box in result.boxes:

        cls_id = int(box.cls[0])

        conf = float(box.conf[0])

        xyxy = (
            box.xyxy[0]
            .cpu()
            .numpy()
            .tolist()
        )

        label_name = names.get(
            cls_id,
            "Unknown"
        )

        detections.append(
            {
                "class_id": cls_id,
                "class": label_name,
                "confidence": conf,
                "bbox": [
                    round(v, 1)
                    for v in xyxy
                ],
            }
        )

    # -------------------------------------------------------------- #
    # Sort strongest detections first
    # -------------------------------------------------------------- #

    detections.sort(
        key=lambda d: d["confidence"],
        reverse=True
    )

    # -------------------------------------------------------------- #
    # Draw boxes
    # -------------------------------------------------------------- #

    annotated_bgr = _draw_detections(
        bgr,
        result.boxes,
        names
    )

    annotated_rgb = cv2.cvtColor(
        annotated_bgr,
        cv2.COLOR_BGR2RGB
    )

    annotated_image = Image.fromarray(
        annotated_rgb
    )

    # -------------------------------------------------------------- #
    # Return results
    # -------------------------------------------------------------- #

    return {
        "annotated_image": annotated_image,
        "detections": detections,
        "inference_time_ms": elapsed_ms,
        "image_size": pil_image.size,
    }


# ==================================================================== #
# OPTIONAL SECOND ANALYSIS LAYER -- classical CV visual anomaly check
# ==================================================================== #
# This is completely independent of the YOLO model above. It does NOT
# use best.pt, does NOT use CLASS_NAMES, and does NOT produce AI
# detections. It compares two images with plain OpenCV image-diffing
# and flags regions that changed significantly.
#
# Anything this finds is labeled ONLY as "Unclassified Visual Anomaly"
# -- never as aircraft/fish/reef/ship, never as "debris", and never
# described as an AI/model detection. It is a classical heuristic.
# ------------------------------------------------------------------ #

ANOMALY_LABEL = "Unclassified Visual Anomaly"
ANOMALY_COLOR_BGR = (0, 0, 255)  # pure red -- visually distinct from all 4 class colors

# If more than this fraction of the frame differs, the two images are
# probably not the same scene/location at all, so results would be
# meaningless -- report unreliable instead of flooding with boxes.
MAX_RELIABLE_DIFF_RATIO = 0.60

# Minimum contour area, as a fraction of total image area, to count as
# an anomaly rather than noise/compression artifacts.
MIN_ANOMALY_AREA_RATIO = 0.001


def detect_visual_anomalies(
    reference_pil_image: Image.Image,
    current_pil_image: Image.Image,
    min_area_ratio: float = MIN_ANOMALY_AREA_RATIO,
):
    """
    Classical CV comparison between a reference/baseline sonar image and
    the current sonar image. Flags regions that changed significantly.

    This is NOT the YOLO model and NOT an AI detection. It is a plain
    OpenCV image-difference + contour heuristic, intended only to
    highlight visual changes that fall outside the model's 4 trained
    classes. Results are always labeled "Unclassified Visual Anomaly".

    Returns a dict:
        annotated_image : PIL.Image -- current image with only anomaly
                           boxes drawn (separate from any YOLO boxes)
        anomaly_boxes   : list of [x1, y1, x2, y2] in current-image coords
        count           : int, number of anomaly regions found
        reliable        : bool, False if the two images are too
                           different to compare meaningfully
        warning         : str or None, explanation when reliable=False
    """

    # -------------------------------------------------------------- #
    # Normalize both images to RGB, then to matching size/BGR for CV
    # -------------------------------------------------------------- #

    current_rgb = np.array(current_pil_image.convert("RGB"))
    reference_rgb = np.array(reference_pil_image.convert("RGB"))

    current_bgr = cv2.cvtColor(current_rgb, cv2.COLOR_RGB2BGR)
    reference_bgr = cv2.cvtColor(reference_rgb, cv2.COLOR_RGB2BGR)

    h, w = current_bgr.shape[:2]

    # Handle different image sizes safely: always resize the reference
    # to match the current image's dimensions before comparing.
    if reference_bgr.shape[:2] != (h, w):
        reference_bgr = cv2.resize(
            reference_bgr, (w, h), interpolation=cv2.INTER_AREA
        )

    # -------------------------------------------------------------- #
    # Grayscale + blur to reduce sensor/compression noise
    # -------------------------------------------------------------- #

    current_gray = cv2.cvtColor(current_bgr, cv2.COLOR_BGR2GRAY)
    reference_gray = cv2.cvtColor(reference_bgr, cv2.COLOR_BGR2GRAY)

    current_blur = cv2.GaussianBlur(current_gray, (5, 5), 0)
    reference_blur = cv2.GaussianBlur(reference_gray, (5, 5), 0)

    # -------------------------------------------------------------- #
    # Absolute difference + threshold
    # -------------------------------------------------------------- #

    diff = cv2.absdiff(reference_blur, current_blur)
    _, thresh = cv2.threshold(diff, 30, 255, cv2.THRESH_BINARY)

    # Clean up small speckle noise
    kernel = np.ones((5, 5), np.uint8)
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=1)
    thresh = cv2.dilate(thresh, kernel, iterations=2)

    # -------------------------------------------------------------- #
    # Reliability check -- are these even comparable images?
    # -------------------------------------------------------------- #

    total_pixels = h * w
    changed_pixels = int(np.count_nonzero(thresh))
    diff_ratio = changed_pixels / total_pixels if total_pixels > 0 else 0.0

    if diff_ratio > MAX_RELIABLE_DIFF_RATIO:
        return {
            "annotated_image": current_pil_image,
            "anomaly_boxes": [],
            "count": 0,
            "reliable": False,
            "warning": (
                "The reference and current sonar images appear too different "
                "to compare reliably (over "
                f"{MAX_RELIABLE_DIFF_RATIO * 100:.0f}% of the frame changed). "
                "This likely means they are not the same scene/location. "
                "Anomaly highlighting was skipped to avoid misleading results."
            ),
        }

    # -------------------------------------------------------------- #
    # Find contours -> candidate anomaly boxes
    # -------------------------------------------------------------- #

    contours, _ = cv2.findContours(
        thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

    min_area = min_area_ratio * total_pixels
    anomaly_boxes = []

    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < min_area:
            continue
        x, y, bw, bh = cv2.boundingRect(cnt)
        anomaly_boxes.append([x, y, x + bw, y + bh])

    # -------------------------------------------------------------- #
    # Draw anomaly boxes on a copy of the current image only
    # -------------------------------------------------------------- #

    annotated = current_bgr.copy()

    for (x1, y1, x2, y2) in anomaly_boxes:
        cv2.rectangle(annotated, (x1, y1), (x2, y2), ANOMALY_COLOR_BGR, 2)

        (tw, th), baseline = cv2.getTextSize(
            ANOMALY_LABEL, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2
        )
        ty1 = max(y1 - th - baseline - 6, 0)

        cv2.rectangle(
            annotated, (x1, ty1), (x1 + tw + 8, ty1 + th + baseline + 6),
            ANOMALY_COLOR_BGR, -1
        )
        cv2.putText(
            annotated, ANOMALY_LABEL, (x1 + 4, ty1 + th + 2),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA
        )

    annotated_rgb = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)
    annotated_image = Image.fromarray(annotated_rgb)

    return {
        "annotated_image": annotated_image,
        "anomaly_boxes": anomaly_boxes,
        "count": len(anomaly_boxes),
        "reliable": True,
        "warning": None,
    }