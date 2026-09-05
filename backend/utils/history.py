"""
utils/history.py
-----------------
Local, in-memory session history for AquaShield.

No database, no disk persistence — this lives in st.session_state for the
duration of the browser session, exactly like the rest of the app's state.
Closing the tab / restarting Streamlit clears it, same as everything else
that isn't written to feedback_log.csv.
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import List, Optional
import pandas as pd
import streamlit as st

HISTORY_KEY = "ds_history"


@dataclass
class HistoryEntry:
    timestamp: str
    image_name: str
    detections: int
    top_confidence: float
    priority: str
    latitude: str
    longitude: str
    anomaly_count: Optional[int]
    feedback_status: str


def _ensure_history() -> List[dict]:
    if HISTORY_KEY not in st.session_state:
        st.session_state[HISTORY_KEY] = []
    return st.session_state[HISTORY_KEY]


def add_history_entry(
    image_name: str,
    detections: list,
    priority: str = "N/A",
    latitude: str = "",
    longitude: str = "",
    anomaly_count: Optional[int] = None,
    feedback_status: str = "Pending",
) -> None:
    """Append one analysis run to the session history list."""
    history = _ensure_history()
    top_conf = max((d["confidence"] for d in detections), default=0.0)

    entry = HistoryEntry(
        timestamp=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        image_name=image_name,
        detections=len(detections),
        top_confidence=round(top_conf, 3),
        priority=priority,
        latitude=latitude.strip() if latitude else "",
        longitude=longitude.strip() if longitude else "",
        anomaly_count=anomaly_count,
        feedback_status=feedback_status,
    )
    history.append(entry.__dict__)


def get_history_df() -> pd.DataFrame:
    """Return the session history as a DataFrame, newest first."""
    history = _ensure_history()
    if not history:
        return pd.DataFrame(
            columns=[
                "timestamp", "image_name", "detections", "top_confidence",
                "priority", "latitude", "longitude", "anomaly_count", "feedback_status",
            ]
        )
    df = pd.DataFrame(history)
    return df.iloc[::-1].reset_index(drop=True)


def clear_history() -> None:
    st.session_state[HISTORY_KEY] = []


def history_count() -> int:
    return len(_ensure_history())