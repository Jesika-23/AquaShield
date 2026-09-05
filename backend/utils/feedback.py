"""
feedback.py
Lightweight, local CSV logger for human-in-the-loop detection feedback.

This does NOT retrain or modify the model in any way. It only records
what a human reviewer thought of each detection, so the log can be
used later (manually, by a person) to inform future retraining.

CSV columns:
    timestamp, image_filename, class_id, displayed_class,
    confidence, bbox, feedback, latitude, longitude

Latitude/longitude are optional and manually entered by the user in
the sidebar. They are NEVER generated, guessed, or looked up here --
if not provided, the literal string "Location not provided" is
stored instead of a coordinate.

NOTE: This module never deletes, renames, or migrates any existing
feedback_log.csv. If an older CSV (without latitude/longitude
columns) already exists at FEEDBACK_CSV_PATH, new rows will simply
have two extra columns; that file is left as-is unless the user
manually deletes or renames it.
"""

import csv
from pathlib import Path
from datetime import datetime

FEEDBACK_CSV_PATH = Path(__file__).resolve().parent.parent / "feedback_log.csv"

FIELDNAMES = [
    "timestamp",
    "image_filename",
    "class_id",
    "displayed_class",
    "confidence",
    "bbox",
    "feedback",
    "latitude",
    "longitude",
]

NOT_PROVIDED_TEXT = "Location not provided"


def save_feedback(
    image_filename: str,
    class_id: int,
    displayed_class: str,
    confidence: float,
    bbox: list,
    feedback: str,
    latitude: str = "",
    longitude: str = "",
    csv_path: Path = FEEDBACK_CSV_PATH,
) -> None:
    """
    Append one feedback row to the local feedback CSV.
    Creates the file with a header row if it doesn't exist yet.
    Never modifies, deletes, or rewrites any existing rows/file.

    latitude / longitude are optional, user-entered strings. If either
    is blank, "Location not provided" is stored -- coordinates are
    never invented or looked up automatically.

    Does not touch the model, weights, or label files in any way.
    """

    csv_path = Path(csv_path)
    file_exists = csv_path.exists()

    lat_value = latitude.strip() if latitude and latitude.strip() else NOT_PROVIDED_TEXT
    lon_value = longitude.strip() if longitude and longitude.strip() else NOT_PROVIDED_TEXT

    row = {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "image_filename": image_filename,
        "class_id": class_id,
        "displayed_class": displayed_class,
        "confidence": round(float(confidence), 4),
        "bbox": bbox,
        "feedback": feedback,
        "latitude": lat_value,
        "longitude": lon_value,
    }

    with open(csv_path, mode="a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)


# ==================================================================== #
# NEW -- Dataset-ready export
# ==================================================================== #
# Reads the existing feedback_log.csv (never modifies or deletes it) and
# produces a second, clean CSV formatted for manual review / future
# labeling work. This does NOT retrain the model and does NOT claim the
# model learned anything from feedback -- it only reformats what a human
# already recorded so a person can use it later, manually, if they choose.
# ------------------------------------------------------------------ #

import pandas as pd

EXPORT_CSV_PATH = Path(__file__).resolve().parent.parent / "feedback_export_dataset_ready.csv"

EXPORT_FIELDNAMES = [
    "image",
    "class_id",
    "displayed_class",
    "confidence",
    "bbox",
    "feedback",
    "latitude",
    "longitude",
]


def export_dataset_ready_csv(
    source_csv_path: Path = FEEDBACK_CSV_PATH,
    export_csv_path: Path = EXPORT_CSV_PATH,
) -> pd.DataFrame:
    """
    Read feedback_log.csv and write a clean, YOLO-label-ready CSV
    (renamed/reordered columns only -- no rows are added, changed, or
    dropped). Returns the exported DataFrame so the caller can preview it
    or offer it via st.download_button.

    Raises FileNotFoundError if source_csv_path does not exist yet.
    Does NOT retrain the model. Does NOT modify source_csv_path.
    """
    source_csv_path = Path(source_csv_path)
    if not source_csv_path.exists():
        raise FileNotFoundError(
            f"No feedback log found at '{source_csv_path}'. "
            "Collect at least one piece of feedback before exporting."
        )

    df = pd.read_csv(source_csv_path)

    export_df = pd.DataFrame({
        "image": df.get("image_filename", ""),
        "class_id": df.get("class_id", ""),
        "displayed_class": df.get("displayed_class", ""),
        "confidence": df.get("confidence", ""),
        "bbox": df.get("bbox", ""),
        "feedback": df.get("feedback", ""),
        "latitude": df.get("latitude", ""),
        "longitude": df.get("longitude", ""),
    })

    export_df.to_csv(export_csv_path, index=False, columns=EXPORT_FIELDNAMES)
    return export_df