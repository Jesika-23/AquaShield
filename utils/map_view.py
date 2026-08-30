"""
utils/map_view.py
------------------
Optional geospatial visualization for AquaShield.

Shows ONLY coordinates the user manually entered — nothing is ever invented,
geocoded, or looked up. If no valid lat/lon history exists, the caller should
skip rendering this section entirely (see app.py wiring notes).

Dependency: plotly (not currently in requirements.txt). Install with:
    pip install plotly
This module is optional — if plotly isn't installed, importing it will raise
ImportError and the calling code in app.py should catch that and simply hide
the map section (see wiring instructions).
"""

from typing import List, Dict
import pandas as pd

try:
    import plotly.graph_objects as go
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False


PRIORITY_COLORS = {
    "LOW": "#5eead4",
    "MEDIUM": "#ffb020",
    "HIGH": "#ff5c5c",
    "REQUIRES HUMAN VERIFICATION": "#a78bfa",
    "N/A": "#91aebe",
}


def _parse_coord(value: str):
    try:
        f = float(str(value).strip())
        return f
    except (ValueError, TypeError):
        return None


def build_location_dataframe(history_df: pd.DataFrame) -> pd.DataFrame:
    """
    Filter a history DataFrame (see utils/history.py) down to rows that have
    valid, user-entered latitude/longitude values. Never fabricates points.
    """
    if history_df.empty:
        return pd.DataFrame(columns=["lat", "lon", "image_name", "priority", "timestamp"])

    rows = []
    for _, r in history_df.iterrows():
        lat = _parse_coord(r.get("latitude", ""))
        lon = _parse_coord(r.get("longitude", ""))
        if lat is None or lon is None:
            continue
        if not (-90 <= lat <= 90 and -180 <= lon <= 180):
            continue
        rows.append({
            "lat": lat,
            "lon": lon,
            "image_name": r.get("image_name", ""),
            "priority": r.get("priority", "N/A"),
            "timestamp": r.get("timestamp", ""),
        })
    return pd.DataFrame(rows)


def render_map_figure(locations_df: pd.DataFrame):
    """
    Build a plotly Scattermapbox figure colored by priority.
    Returns None if plotly isn't installed or there's nothing to plot.
    Caller (app.py) is responsible for st.plotly_chart(...) and for
    checking PLOTLY_AVAILABLE before calling this.
    """
    if not PLOTLY_AVAILABLE or locations_df.empty:
        return None

    fig = go.Figure()
    for priority, group in locations_df.groupby("priority"):
        fig.add_trace(go.Scattermapbox(
            lat=group["lat"],
            lon=group["lon"],
            mode="markers",
            marker=dict(size=13, color=PRIORITY_COLORS.get(priority, "#91aebe")),
            text=group.apply(
                lambda r: f"{r['image_name']} — {priority} — {r['timestamp']}", axis=1
            ),
            hoverinfo="text",
            name=priority,
        ))

    center_lat = locations_df["lat"].mean()
    center_lon = locations_df["lon"].mean()

    fig.update_layout(
        mapbox=dict(
            style="carto-darkmatter",
            center=dict(lat=center_lat, lon=center_lon),
            zoom=4,
        ),
        margin=dict(l=0, r=0, t=0, b=0),
        paper_bgcolor="rgba(0,0,0,0)",
        legend=dict(font=dict(color="#eaf4f7"), bgcolor="rgba(0,0,0,0)"),
        height=420,
    )
    return fig