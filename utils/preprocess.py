"""
utils/preprocess.py
--------------------
Prototype preprocessing utilities for AquaShield.

These are classical OpenCV operations (denoising, CLAHE contrast enhancement,
optional shadow enhancement). They are OPTIONAL, operator-triggered, and never
change the trained model or its class IDs. Label all outputs as "prototype
preprocessing" in the UI — this is not a validated sonar-physics pipeline.
"""

import numpy as np
import cv2


def denoise(image: np.ndarray, strength: int = 7) -> np.ndarray:
    """Fast Non-Local Means denoising. Works on RGB or grayscale uint8 images."""
    img = image.astype(np.uint8)
    if img.ndim == 3:
        return cv2.fastNlMeansDenoisingColored(img, None, strength, strength, 7, 21)
    return cv2.fastNlMeansDenoising(img, None, strength, 7, 21)


def apply_clahe(image: np.ndarray, clip_limit: float = 2.5, tile_grid: int = 8) -> np.ndarray:
    """Contrast Limited Adaptive Histogram Equalization, applied on luminance only."""
    img = image.astype(np.uint8)
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(tile_grid, tile_grid))
    if img.ndim == 3:
        lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
        l, a, b = cv2.split(lab)
        l = clahe.apply(l)
        lab = cv2.merge((l, a, b))
        return cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)
    return clahe.apply(img)


def enhance_shadows(image: np.ndarray, gamma: float = 1.6) -> np.ndarray:
    """
    Gamma correction to lift shadow detail (gamma > 1 brightens dark regions).
    Purely a visualization/preprocessing aid — not a physical shadow model.
    """
    img = image.astype(np.float32) / 255.0
    corrected = np.power(img, 1.0 / gamma)
    return np.clip(corrected * 255.0, 0, 255).astype(np.uint8)


def enhance_image(
    image: np.ndarray,
    do_denoise: bool = True,
    do_clahe: bool = True,
    do_shadow: bool = False,
) -> np.ndarray:
    """
    Apply the selected prototype enhancement pipeline in sequence:
    denoise -> CLAHE -> shadow lift. Each step is optional/togglable.
    """
    result = image.copy()
    if do_denoise:
        result = denoise(result)
    if do_clahe:
        result = apply_clahe(result)
    if do_shadow:
        result = enhance_shadows(result)
    return result


def compare_detection_counts(original_results, enhanced_results) -> dict:
    """
    Small helper: given two Ultralytics Results objects (original vs enhanced
    image inference), summarize the count difference for side-by-side display.
    Does not alter either result — purely descriptive.
    """
    orig_count = len(original_results.boxes) if original_results.boxes is not None else 0
    enh_count = len(enhanced_results.boxes) if enhanced_results.boxes is not None else 0
    return {
        "original_detections": orig_count,
        "enhanced_detections": enh_count,
        "difference": enh_count - orig_count,
    }