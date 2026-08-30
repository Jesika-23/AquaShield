"""
utils/priority.py
------------------
Explainable priority scoring — a DECISION-SUPPORT HEURISTIC for AquaShield.

Combines several already-computed signals (detection confidence, box size,
image quality, presence of a visual anomaly, shadow evidence) into a single
LOW / MEDIUM / HIGH / REQUIRES HUMAN VERIFICATION recommendation, with the
contributing factors shown alongside it.

This is a rules-based weighting, not a trained model, not a probability of
threat, and not scientifically validated. It exists to help a reviewer
triage a batch of results faster — final judgment always rests with a human.
"""

from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class PriorityResult:
    priority: str  # "LOW" | "MEDIUM" | "HIGH" | "REQUIRES HUMAN VERIFICATION"
    score: float  # 0-100 explainable composite score
    factors: List[str] = field(default_factory=list)

    def summary_line(self) -> str:
        return " -> ".join(self.factors + [f"Priority: {self.priority}"])


def _confidence_factor(confidence: float) -> (str, float):
    if confidence >= 0.70:
        return "Confidence: High", 30.0
    elif confidence >= 0.45:
        return "Confidence: Medium", 18.0
    else:
        return "Confidence: Low", 6.0


def _size_factor(bbox: List[float], image_size) -> (str, float):
    img_w, img_h = image_size
    x1, y1, x2, y2 = bbox
    box_area = max(0.0, (x2 - x1)) * max(0.0, (y2 - y1))
    img_area = max(1.0, img_w * img_h)
    ratio = box_area / img_area
    if ratio >= 0.08:
        return "Object size: Large", 15.0
    elif ratio >= 0.015:
        return "Object size: Medium", 10.0
    else:
        return "Object size: Small", 5.0


def _quality_factor(quality_rating: Optional[str]) -> (str, float):
    if quality_rating == "Good":
        return "Image quality: Good", 20.0
    elif quality_rating == "Moderate":
        return "Image quality: Moderate", 10.0
    elif quality_rating == "Poor":
        return "Image quality: Poor", 2.0
    return "Image quality: Unknown", 8.0


def _anomaly_factor(anomaly_present: bool) -> (str, float):
    if anomaly_present:
        return "Anomaly: Present", 15.0
    return "Anomaly: None", 0.0


def _shadow_factor(shadow_reliability_label: Optional[str]) -> (str, float):
    if shadow_reliability_label == "High":
        return "Shadow evidence: Strong", 20.0
    elif shadow_reliability_label == "Moderate":
        return "Shadow evidence: Moderate", 10.0
    elif shadow_reliability_label == "Low":
        return "Shadow evidence: Weak", 3.0
    return "Shadow evidence: Not assessed", 5.0


def compute_priority(
    confidence: float,
    bbox: List[float],
    image_size,
    quality_rating: Optional[str] = None,
    anomaly_present: bool = False,
    shadow_reliability_label: Optional[str] = None,
) -> PriorityResult:
    """
    Combine signals into an explainable 0-100 score and a categorical
    priority label. Every contributing factor is returned in plain language
    so the UI can show exactly why a score was assigned.
    """
    factors: List[str] = []
    score = 0.0

    label, pts = _confidence_factor(confidence)
    factors.append(label)
    score += pts

    label, pts = _size_factor(bbox, image_size)
    factors.append(label)
    score += pts

    label, pts = _quality_factor(quality_rating)
    factors.append(label)
    score += pts

    label, pts = _anomaly_factor(anomaly_present)
    factors.append(label)
    score += pts

    label, pts = _shadow_factor(shadow_reliability_label)
    factors.append(label)
    score += pts

    # Human-verification override: low confidence + poor image quality is
    # exactly the scenario where an automated label is least trustworthy.
    needs_human = confidence < 0.35 and quality_rating == "Poor"

    if needs_human:
        priority = "REQUIRES HUMAN VERIFICATION"
    elif score >= 65:
        priority = "HIGH"
    elif score >= 35:
        priority = "MEDIUM"
    else:
        priority = "LOW"

    return PriorityResult(priority=priority, score=round(score, 1), factors=factors)


def priority_badge_color(priority: str) -> str:
    """Hex color hint for UI badges — matches AquaShield's teal/navy palette."""
    return {
        "LOW": "#5eead4",
        "MEDIUM": "#ffb020",
        "HIGH": "#ff5c5c",
        "REQUIRES HUMAN VERIFICATION": "#a78bfa",
    }.get(priority, "#91aebe")