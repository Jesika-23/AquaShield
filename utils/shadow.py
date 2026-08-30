"""
utils/shadow.py
----------------
Prototype acoustic-shadow analysis for AquaShield.

Sonar imagery convention: a solid object on the seabed often casts a dark
"shadow" region behind it (opposite the sonar source). Analysts sometimes use
target/shadow size and shape as a secondary cue for object identification.

THIS MODULE IS A CLASSICAL-CV HEURISTIC, NOT REAL SONAR PHYSICS. It looks at
pixel-intensity darkness in a region adjacent to each YOLO bounding box and
estimates a plausible "shadow" footprint using simple thresholding. It is
NOT validated against real acoustic shadow physics, NOT part of the trained
model, and should be clearly labeled "prototype heuristic" in the UI.
"""

from dataclasses import dataclass
from typing import List, Tuple
import numpy as np
import cv2


@dataclass
class ShadowAnalysis:
    bbox: List[float]
    shadow_area_px: int
    shadow_length_px: float
    target_area_px: int
    target_shadow_ratio: float  # shadow_area / target_area
    reliability_score: float  # 0-1, heuristic confidence in this estimate
    reliability_label: str  # "Low" | "Moderate" | "High"
    note: str


def _shadow_search_region(
    x1: int, y1: int, x2: int, y2: int, img_w: int, img_h: int, direction: str = "below"
) -> Tuple[int, int, int, int]:
    """
    Define a region adjacent to the bounding box to search for a shadow.
    Default assumption: shadow extends "below" the target in image space
    (a common sonar convention when the sensor look-direction is roughly
    top-to-bottom). This is a simplifying assumption, not measured physics.
    """
    h = y2 - y1
    w = x2 - x1
    if direction == "below":
        sy1 = min(y2, img_h - 1)
        sy2 = min(y2 + int(h * 1.5) + 5, img_h)
        return x1, sy1, x2, sy2
    # fallback: right side
    sx1 = min(x2, img_w - 1)
    sx2 = min(x2 + int(w * 1.5) + 5, img_w)
    return sx1, y1, sx2, y2


def analyze_shadow(
    gray_image: np.ndarray,
    bbox: List[float],
    dark_percentile: float = 25.0,
) -> ShadowAnalysis:
    """
    Estimate a shadow region adjacent to a single detection bounding box.

    Args:
        gray_image: grayscale numpy array of the full sonar image.
        bbox: [x1, y1, x2, y2] in pixel coordinates.
        dark_percentile: pixels darker than this percentile (within the
            search region) are treated as candidate shadow pixels.

    Returns:
        ShadowAnalysis with area/length estimates and a reliability score.
    """
    img_h, img_w = gray_image.shape[:2]
    x1, y1, x2, y2 = [int(v) for v in bbox]
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(img_w, x2), min(img_h, y2)

    target_area_px = max(1, (x2 - x1) * (y2 - y1))

    sx1, sy1, sx2, sy2 = _shadow_search_region(x1, y1, x2, y2, img_w, img_h)
    region = gray_image[sy1:sy2, sx1:sx2]

    if region.size == 0:
        return ShadowAnalysis(
            bbox=bbox,
            shadow_area_px=0,
            shadow_length_px=0.0,
            target_area_px=target_area_px,
            target_shadow_ratio=0.0,
            reliability_score=0.0,
            reliability_label="Low",
            note="Shadow search region fell outside image bounds — no estimate possible.",
        )

    thresh_val = float(np.percentile(region, dark_percentile))
    shadow_mask = (region <= thresh_val).astype(np.uint8) * 255

    kernel = np.ones((3, 3), np.uint8)
    shadow_mask = cv2.morphologyEx(shadow_mask, cv2.MORPH_OPEN, kernel)

    shadow_area_px = int(np.count_nonzero(shadow_mask))

    # Estimate shadow "length" as the tallest contiguous dark run along the
    # search-region's primary axis (a coarse 1-D proxy, not a true measurement).
    col_sums = np.count_nonzero(shadow_mask, axis=0)
    shadow_length_px = float(np.max(col_sums)) if col_sums.size else 0.0

    ratio = shadow_area_px / target_area_px if target_area_px else 0.0

    # Reliability heuristic: favors cases where a clear, moderately-sized
    # dark region was found relative to the target, and search region wasn't
    # degenerate. Purely descriptive — not a statistically calibrated score.
    coverage = shadow_area_px / region.size if region.size else 0.0
    if coverage < 0.03 or ratio < 0.05:
        reliability_score, reliability_label = 0.25, "Low"
        note = "Little to no distinct dark region found adjacent to the target."
    elif ratio > 3.0 or coverage > 0.9:
        reliability_score, reliability_label = 0.4, "Low"
        note = "Shadow region estimate is unusually large relative to target — likely noise or seabed texture."
    elif 0.2 <= ratio <= 2.0:
        reliability_score, reliability_label = 0.75, "High"
        note = "Shadow footprint is plausible relative to target size."
    else:
        reliability_score, reliability_label = 0.55, "Moderate"
        note = "Shadow footprint found but ratio to target is atypical."

    return ShadowAnalysis(
        bbox=bbox,
        shadow_area_px=shadow_area_px,
        shadow_length_px=round(shadow_length_px, 1),
        target_area_px=target_area_px,
        target_shadow_ratio=round(ratio, 3),
        reliability_score=reliability_score,
        reliability_label=reliability_label,
        note=note,
    )


def analyze_all_shadows(rgb_image: np.ndarray, detections: List[dict]) -> List[ShadowAnalysis]:
    """Run analyze_shadow() for every detection dict with a 'bbox' key."""
    gray = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2GRAY) if rgb_image.ndim == 3 else rgb_image
    return [analyze_shadow(gray, det["bbox"]) for det in detections]