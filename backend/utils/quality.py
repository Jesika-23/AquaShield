"""
utils/quality.py
-----------------
Image Quality Assessment layer for AquaShield.

IMPORTANT: This module is a QUALITY-AWARENESS layer only. It does NOT predict
model accuracy, does NOT modify detections, and is NOT a trained AI component.
It uses classical image statistics (brightness, contrast, noise, blur) to warn
the operator about conditions that commonly degrade detection reliability.
"""

from dataclasses import dataclass, field
from typing import List
import numpy as np
import cv2


@dataclass
class QualityReport:
    brightness: float
    contrast: float
    noise_estimate: float
    sharpness: float  # Laplacian variance
    rating: str  # "Good" | "Moderate" | "Poor"
    warnings: List[str] = field(default_factory=list)

    def as_dict(self):
        return {
            "Brightness (0-255)": round(self.brightness, 1),
            "Contrast (std dev)": round(self.contrast, 1),
            "Noise Estimate": round(self.noise_estimate, 2),
            "Sharpness (Laplacian Var)": round(self.sharpness, 1),
            "Rating": self.rating,
        }


def _to_gray(image: np.ndarray) -> np.ndarray:
    if image.ndim == 3:
        return cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    return image


def _estimate_noise(gray: np.ndarray) -> float:
    """
    Fast noise estimate using the Laplacian-of-Gaussian residual method
    (Immerkaer, 1996). Cheap, classical, no ML involved.
    """
    h, w = gray.shape
    laplacian_kernel = np.array([[1, -2, 1], [-2, 4, -2], [1, -2, 1]], dtype=np.float64)
    conv = cv2.filter2D(gray.astype(np.float64), -1, laplacian_kernel)
    sigma = np.sum(np.abs(conv))
    sigma = sigma * np.sqrt(0.5 * np.pi) / (6 * (w - 2) * (h - 2))
    return float(sigma)


def assess_image_quality(image: np.ndarray) -> QualityReport:
    """
    Compute classical image-quality statistics.

    Args:
        image: RGB or grayscale numpy array (as loaded via PIL/cv2 upstream).

    Returns:
        QualityReport with a Good/Moderate/Poor rating and plain-language
        warnings. This is a heuristic quality-awareness aid, not an
        accuracy predictor.
    """
    gray = _to_gray(image)

    brightness = float(np.mean(gray))
    contrast = float(np.std(gray))
    noise_estimate = _estimate_noise(gray)
    sharpness = float(cv2.Laplacian(gray, cv2.CV_64F).var())

    warnings: List[str] = []
    score = 0  # higher = better, 0-4 scale of passed checks

    # Brightness checks
    if brightness < 40:
        warnings.append("Image is very dark — low brightness may hide low-contrast targets.")
    elif brightness > 215:
        warnings.append("Image is very bright / overexposed — detail may be washed out.")
    else:
        score += 1

    # Contrast checks
    if contrast < 20:
        warnings.append("Low contrast may affect detection reliability.")
    else:
        score += 1

    # Noise checks
    if noise_estimate > 6.0:
        warnings.append("High estimated noise level — may increase false positives/negatives.")
    else:
        score += 1

    # Sharpness / blur checks
    if sharpness < 60:
        warnings.append("Image appears blurry or low-detail (low sharpness score).")
    else:
        score += 1

    if score >= 4:
        rating = "Good"
    elif score >= 2:
        rating = "Moderate"
    else:
        rating = "Poor"

    if not warnings:
        warnings.append("No significant quality issues detected.")

    return QualityReport(
        brightness=brightness,
        contrast=contrast,
        noise_estimate=noise_estimate,
        sharpness=sharpness,
        rating=rating,
        warnings=warnings,
    )