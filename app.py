"""
app.py
AquaShield -- AI Marine Debris & Object Detection
Streamlit front end wrapping the existing YOLO11n (model/best.pt) pipeline.
Custom HTML/CSS/JS is injected on top of Streamlit to deliver a full
product-grade marine-surveillance UI while all inference stays in Python.
"""


from api_client import detect_via_backend
from pathlib import Path



import pandas as pd
import streamlit as st
from PIL import Image, UnidentifiedImageError



from utils.inference import detect_visual_anomalies
from utils.report import build_text_report
from utils.feedback import save_feedback, FEEDBACK_CSV_PATH, export_dataset_ready_csv
from utils.quality import assess_image_quality
from utils.shadow import analyze_all_shadows
from utils.priority import compute_priority, priority_badge_color
from utils.history import add_history_entry, get_history_df, history_count
import numpy as np



try:
    from utils.map_view import build_location_dataframe, render_map_figure, PLOTLY_AVAILABLE
except ImportError:
    PLOTLY_AVAILABLE = False



BASE_DIR = Path(__file__).resolve().parent



MODEL_METRICS = {
    "precision": 0.473,
    "recall": 0.553,
    "map50": 0.515,
    "map50_95": 0.314,
    "train_images": 402,
    "val_images": 110,
}



CLASSES = [
    {"code": "AIR", "name": "Aircraft", "chip": "#FFB020", "css": "aircraft",
     "desc": "Downed or submerged airframes and wreckage returns."},
    {"code": "FSH", "name": "Fish", "chip": "#5EEAD4", "css": "fish",
     "desc": "Biological returns -- shoals and large individual specimens."},
    {"code": "REF", "name": "Reef", "chip": "#38BDF8", "css": "reef",
     "desc": "Artificial reef structures and hard bottom returns on the seabed."},
    {"code": "SHP", "name": "Ship", "chip": "#A78BFA", "css": "ship",
     "desc": "Submerged hull signatures with characteristic cylindrical returns."},
]



# ------------------------------------------------------------------ #
# Page config + CSS injection
# ------------------------------------------------------------------ #
st.set_page_config(
    page_title="AquaShield | AI Sonar Object Detection",
    page_icon="\U0001F30A",
    layout="wide",
    initial_sidebar_state="expanded",
)




def inject_css():
    css_path = BASE_DIR / "assets" / "style.css"
    st.markdown(
        """
        <link rel="preconnect" href="https://fonts.googleapis.com">
        <link href="https://fonts.googleapis.com/css2?family=Rajdhani:wght@500;600;700&family=Inter:wght@400;500;600&family=JetBrains+Mono:wght@400;500;600&display=swap" rel="stylesheet">
        """,
        unsafe_allow_html=True,
    )
    st.markdown(f"<style>{css_path.read_text()}</style>", unsafe_allow_html=True)




inject_css()



# ------------------------------------------------------------------ #
# Sidebar -- MODEL SETTINGS / ANALYSIS SETTINGS / LOCATION /
# REFERENCE IMAGE / SYSTEM STATUS
# All widgets below use the SAME keys/variable names the rest of the
# app already expects (conf_threshold, ds_latitude, ds_longitude,
# reference_file) -- this only relocates existing inputs into the
# sidebar, it does not change any downstream logic.
# ------------------------------------------------------------------ #
with st.sidebar:
    st.markdown('<div class="ds-sidebar-title">Model Settings</div>', unsafe_allow_html=True)
    st.caption("YOLO11n &middot; 4 trained classes &middot; backend inference", unsafe_allow_html=True)
    conf_threshold = st.slider(
        "Confidence threshold", min_value=0.05, max_value=0.95, value=0.50, step=0.05,
        key="sb_conf_threshold",
    )
    enhance_toggle = st.checkbox(
        "Enable prototype preprocessing (denoise + CLAHE)", value=False,
        help="Optional. Runs a second inference pass on an enhanced copy of "
             "the image so you can compare Original vs Enhanced detections.",
    )



    st.markdown('<div class="ds-sidebar-title">Analysis Settings</div>', unsafe_allow_html=True)
    show_shadow = st.checkbox("Run acoustic-shadow analysis (prototype heuristic)", value=True)
    show_priority = st.checkbox("Run explainable priority scoring", value=True)



    st.markdown('<div class="ds-sidebar-title">Location</div>', unsafe_allow_html=True)
    st.caption(
        "Optional, manually entered. Never auto-detected or looked up."
    )
    ds_latitude = st.text_input("Latitude (optional)", value="", placeholder="e.g. 13.0827")
    ds_longitude = st.text_input("Longitude (optional)", value="", placeholder="e.g. 80.2707")
    st.session_state["ds_latitude"] = ds_latitude
    st.session_state["ds_longitude"] = ds_longitude



    st.markdown('<div class="ds-sidebar-title">System Status</div>', unsafe_allow_html=True)
    st.markdown(
        '<span class="ds-status-badge good">&#9679; MODEL ONLINE</span>',
        unsafe_allow_html=True,
    )
    st.caption(f"mAP50 {MODEL_METRICS['map50']*100:.1f}% &middot; Precision {MODEL_METRICS['precision']*100:.1f}%", unsafe_allow_html=True)
    st.caption(f"Session history: {history_count()} analyses logged")



# ------------------------------------------------------------------ #
# Navbar
# ------------------------------------------------------------------ #
st.markdown(
    """
    <div class="ds-nav">
      <div class="brand">
        <div class="dot"></div>
        <div class="name">AQUA<b>SHIELD</b></div>
      </div>
      <div class="links">
        <a href="#home" onclick="dsScrollTo(event,'home')">Home</a>
        <a href="#performance" onclick="dsScrollTo(event,'performance')">Performance</a>
        <a href="#upload" onclick="dsScrollTo(event,'upload')">Analyze</a>
        <a href="#feedback-insights" onclick="dsScrollTo(event,'feedback-insights')">History &amp; Feedback</a>
      </div>
      <div style="display:flex; align-items:center; gap:10px;">
        <span class="ds-pill">PS <b>26057</b></span>
        <div class="status"><span>&#9679;</span> MODEL ONLINE -- YOLO11n</div>
      </div>
    </div>

    <script>
    function dsScrollTo(e, id) {
        e.preventDefault();
        var el = (window.parent && window.parent.document.getElementById(id)) || document.getElementById(id);
        if (el) {
            el.scrollIntoView({ behavior: "smooth", block: "start" });
            try { history.replaceState(null, "", "#" + id); } catch (err) {}
        }
    }
    </script>
    """,
    unsafe_allow_html=True,
)



# ------------------------------------------------------------------ #
# Compact header (replaces the old hero)
# ------------------------------------------------------------------ #
st.markdown(
    """
    <div class="ds-header" id="home">
      <div class="ds-header-copy">
        <div class="eyebrow">AI Sonar Intelligence &middot; Decision Support</div>
        <h1>AquaShield</h1>
        <p>
          YOLO11n object detection for sonar imagery &mdash; aircraft, fish, reef
          and ship signatures, scored in a single pass.
        </p>
      </div>
      <div class="ds-header-status">
        <span class="ds-status-badge good">&#9679; MODEL ONLINE</span>
        <span class="ds-header-meta">YOLO11n &middot; mAP50 51.5%</span>
        <span class="ds-header-meta">Backend &middot; FastAPI /detect</span>
      </div>
    </div>
    """,
    unsafe_allow_html=True,
)




# ------------------------------------------------------------------ #
# Compact detection class strip (replaces the 4 large cards)
# ------------------------------------------------------------------ #
st.markdown('<div class="ds-section-compact" id="classes">', unsafe_allow_html=True)
badges_html = ('<div class="ds-section-head" style="margin-bottom:14px;">'
               '<div class="eyebrow">Coverage</div>'
               '<h2 style="font-size:20px;"><span class="sec-icon"><svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><circle cx="12" cy="12" r="8"/><circle cx="12" cy="12" r="3"/></svg></span>Detection classes</h2></div>'
               '<div class="ds-class-strip">')
for c in CLASSES:
    badges_html += (
        f'<span class="ds-class-chip" style="--chip:{c["chip"]}">'
        f'<b>{c["code"]}</b> {c["name"]}'
        f'</span>'
    )
badges_html += "</div>"
st.markdown(badges_html, unsafe_allow_html=True)
st.markdown("</div>", unsafe_allow_html=True)



# ------------------------------------------------------------------ #
# Performance / model metrics
# ------------------------------------------------------------------ #
st.markdown('<div class="ds-section-compact" id="performance">', unsafe_allow_html=True)
st.markdown(
    """
    <div class="ds-section-head">
      <div class="eyebrow">Validated</div>
      <h2><span class="sec-icon"><svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M12 20a8 8 0 1 0 0-16 8 8 0 0 0 0 16z"/><path d="M12 12l4-4"/></svg></span>Model performance</h2>
      <p>Metrics from the current best.pt checkpoint, evaluated on the 110-image validation split (402 images used for training).</p>
    </div>
    """,
    unsafe_allow_html=True,
)
m1, m2, m3, m4 = st.columns(4)
stat_defs = [
    (m1, f"{MODEL_METRICS['precision']*100:.1f}%", "Precision"),
    (m2, f"{MODEL_METRICS['recall']*100:.1f}%", "Recall"),
    (m3, f"{MODEL_METRICS['map50']*100:.1f}%", "mAP@0.50"),
    (m4, f"{MODEL_METRICS['map50_95']*100:.1f}%", "mAP@0.50-0.95"),
]
for col, value, label in stat_defs:
    with col:
        with st.container(border=True):
            st.markdown(
                f'<div class="ds-stat"><div class="value">{value}</div><div class="label">{label}</div></div>',
                unsafe_allow_html=True,
            )
st.markdown(
    f"""
    <div class="ds-pill" style="margin-top:16px; display:inline-block;">
      Dataset &mdash; <b>{MODEL_METRICS['train_images']}</b> training images &nbsp;/&nbsp;
      <b>{MODEL_METRICS['val_images']}</b> validation images
    </div>
    """,
    unsafe_allow_html=True,
)
st.markdown("</div>", unsafe_allow_html=True)



# ------------------------------------------------------------------ #
# Upload + Analyze
# ------------------------------------------------------------------ #
st.markdown('<div class="ds-section upload-section" id="upload">', unsafe_allow_html=True)
st.markdown(
    """
    <div class="ds-section-head">
      <div class="eyebrow">Run a Detection</div>
      <h2><span class="sec-icon"><svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M12 16V4"/><path d="M6 10l6-6 6 6"/><path d="M4 20h16"/></svg></span>Upload a sonar image</h2>
      <p>Drop a scan below, tune the confidence threshold, then run the model.</p>
    </div>
    """,
    unsafe_allow_html=True,
)



left, right = st.columns([1, 1], gap="large")



with left:
    with st.container(border=True):
        st.markdown("**Sonar image input**")
        uploaded_file = st.file_uploader(
            "Drag & drop a sonar image, or click to browse",
            type=["png", "jpg", "jpeg"],
            label_visibility="visible",
        )
        st.caption(f"Confidence threshold: **{conf_threshold:.2f}** (set in sidebar &rarr; Model Settings)", unsafe_allow_html=True)
        analyze_clicked = st.button(
            "\u25B8  Analyze Image", width="stretch", disabled=uploaded_file is None,
        )



    with st.container(border=True):
        st.markdown("**Reference / Baseline Sonar Image**")
        st.caption(
            "Optional: upload a previous/baseline sonar scan to identify "
            "unclassified visual changes."
        )
        reference_file = st.file_uploader(
            "Reference / Baseline Sonar Image",
            type=["png", "jpg", "jpeg"],
            label_visibility="collapsed",
            key="reference_uploader",
        )



with right:
    with st.container(border=True):
        st.markdown("**Live status**")
        if uploaded_file is None:
            st.markdown(
                """
                <div class="status-panel waiting">
                    <div class="status-big">&#9673;</div>
                    <div class="status-title">AWAITING IMAGE</div>
                    <div class="status-text">Upload a sonar scan on the left, then press Analyze.</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                f"""
                <div class="status-panel ready">
                    <div class="status-big">&#9679;</div>
                    <div class="status-title">READY TO ANALYZE</div>
                    <div class="status-text">{uploaded_file.name}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )



# Run inference on click, persist results across reruns
if analyze_clicked and uploaded_file is not None:
    with st.spinner("Running YOLO11n inference on sonar frame..."):
        try:
            pil_image = Image.open(uploaded_file)
            pil_image.load()  # force full decode now so a corrupt/truncated
                               # file fails here, not mid-inference
        except UnidentifiedImageError:
            st.error(
                "This file couldn't be read as an image. "
                "Please upload a valid JPG, JPEG, or PNG sonar image."
            )
            st.stop()
        except Exception:
            st.error(
                "This file appears to be corrupted or unsupported. "
                "Please try a different JPG, JPEG, or PNG file."
            )
            st.stop()

        try:
            # Call backend instead of local inference
            result = detect_via_backend(pil_image, uploaded_file.name, conf_threshold=conf_threshold)
            st.session_state["ds_result"] = result
            st.session_state["ds_file_name"] = uploaded_file.name
            st.session_state["ds_original"] = pil_image
            st.session_state["ds_conf"] = conf_threshold

            # -------------------------------------------------- #
            # NEW: Image Quality Assessment (prototype heuristic layer)
            # -------------------------------------------------- #
            rgb_array = np.array(pil_image.convert("RGB"))
            quality_report = assess_image_quality(rgb_array)
            st.session_state["ds_quality"] = quality_report

            # -------------------------------------------------- #
            # NEW: Optional prototype preprocessing + comparison pass
            # -------------------------------------------------- #
            if enhance_toggle:
                from utils.preprocess import enhance_image
                enhanced_rgb = enhance_image(rgb_array, do_denoise=True, do_clahe=True)
                enhanced_pil = Image.fromarray(enhanced_rgb)
                # Enhanced pass also via backend
                enhanced_result = detect_via_backend(enhanced_pil, uploaded_file.name + "_enhanced", conf_threshold=conf_threshold)
                st.session_state["ds_enhanced_image"] = enhanced_pil
                st.session_state["ds_enhanced_result"] = enhanced_result
            else:
                st.session_state["ds_enhanced_image"] = None
                st.session_state["ds_enhanced_result"] = None

            # -------------------------------------------------- #
            # NEW: Acoustic-shadow analysis (prototype heuristic layer)
            # -------------------------------------------------- #
            if show_shadow and result["detections"]:
                shadow_results = analyze_all_shadows(rgb_array, result["detections"])
            else:
                shadow_results = []
            st.session_state["ds_shadows"] = shadow_results

            # -------------------------------------------------- #
            # NEW: Explainable priority scoring per detection
            # -------------------------------------------------- #
            priority_results = []
            if show_priority:
                for i, det in enumerate(result["detections"]):
                    shadow_label = shadow_results[i].reliability_label if i < len(shadow_results) else None
                    priority_results.append(
                        compute_priority(
                            confidence=det["confidence"],
                            bbox=det["bbox"],
                            image_size=result["image_size"],
                            quality_rating=quality_report.rating,
                            anomaly_present=False,  # updated below once anomaly result is known
                            shadow_reliability_label=shadow_label,
                        )
                    )
            st.session_state["ds_priorities"] = priority_results
        except RuntimeError as exc:
            # Backend unreachable or failed
            st.error(f"⚠️ {exc}")
            st.stop()
        except Exception:
            st.error(
                "Something went wrong while analyzing this image. "
                "Please try again or use a different file."
            )



        # ---------------------------------------------------------- #
        # OPTIONAL SECOND LAYER: classical CV visual anomaly check.
        # Only runs if a reference image was also uploaded. Never
        # touches, replaces, or affects the YOLO result above.
        # ---------------------------------------------------------- #
        if reference_file is not None and "ds_result" in st.session_state:
            try:
                ref_pil_image = Image.open(reference_file)
                ref_pil_image.load()
                anomaly_result = detect_visual_anomalies(ref_pil_image, pil_image)
                st.session_state["ds_anomaly_result"] = anomaly_result
            except UnidentifiedImageError:
                st.session_state["ds_anomaly_result"] = None
                st.warning(
                    "The reference image couldn't be read as an image, so the "
                    "optional anomaly comparison was skipped. The YOLO detection "
                    "result above is unaffected."
                )
            except Exception:
                st.session_state["ds_anomaly_result"] = None
                st.warning(
                    "Something went wrong running the optional anomaly comparison. "
                    "The YOLO detection result above is unaffected."
                )
        else:
            # No reference image this run -- clear any stale anomaly result
            # so it doesn't linger from a previous upload.
            st.session_state["ds_anomaly_result"] = None



        # ---------------------------------------------------------- #
        # NEW: fold anomaly presence into priority scores now that the
        # anomaly check (if any) has run, then log this run to session
        # history. Never touches feedback_log.csv or the model.
        # ---------------------------------------------------------- #
        anomaly_res = st.session_state.get("ds_anomaly_result")
        anomaly_present = bool(anomaly_res and anomaly_res.get("reliable") and anomaly_res.get("count", 0) > 0)
        anomaly_count_for_history = anomaly_res["count"] if (anomaly_res and anomaly_res.get("reliable")) else None



        quality_report = st.session_state.get("ds_quality")
        shadow_results = st.session_state.get("ds_shadows", [])
        recomputed_priorities = []
        if show_priority:
            for i, det in enumerate(result["detections"]):
                shadow_label = shadow_results[i].reliability_label if i < len(shadow_results) else None
                recomputed_priorities.append(
                    compute_priority(
                        confidence=det["confidence"],
                        bbox=det["bbox"],
                        image_size=result["image_size"],
                        quality_rating=quality_report.rating if quality_report else None,
                        anomaly_present=anomaly_present,
                        shadow_reliability_label=shadow_label,
                    )
                )
        st.session_state["ds_priorities"] = recomputed_priorities



        top_priority = "N/A"
        if recomputed_priorities:
            order = {"HIGH": 3, "REQUIRES HUMAN VERIFICATION": 3, "MEDIUM": 2, "LOW": 1}
            top_priority = max(recomputed_priorities, key=lambda p: order.get(p.priority, 0)).priority



        add_history_entry(
            image_name=uploaded_file.name,
            detections=result["detections"],
            priority=top_priority,
            latitude=st.session_state.get("ds_latitude", ""),
            longitude=st.session_state.get("ds_longitude", ""),
            anomaly_count=anomaly_count_for_history,
            feedback_status="Pending",
        )



# ------------------------------------------------------------------ #
# Results
# ------------------------------------------------------------------ #
if "ds_result" in st.session_state:
    result = st.session_state["ds_result"]
    detections = result["detections"]



    st.markdown("<div style='height:36px'></div>", unsafe_allow_html=True)
    st.markdown(
        f"""
        <div class="ds-section-head">
          <div class="eyebrow">Result</div>
          <h2>Detection output &mdash; {st.session_state['ds_file_name']}</h2>
          <p>{len(detections)} object(s) detected in {result['inference_time_ms']:.0f} ms
          at confidence &ge; {st.session_state['ds_conf']:.2f}.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    top_conf_display = f"{max((d['confidence'] for d in detections), default=0) * 100:.1f}%" if detections else "--"
    st.markdown(
        f"""
        <div class="result-summary">
          <div><span>Objects Found</span><strong>{len(detections)}</strong></div>
          <div><span>Top Confidence</span><strong>{top_conf_display}</strong></div>
          <div><span>Inference Time</span><strong>{result['inference_time_ms']:.0f} ms</strong></div>
          <div><span>Threshold</span><strong>{st.session_state['ds_conf']:.2f}</strong></div>
        </div>
        """,
        unsafe_allow_html=True,
    )



    # ---------------------------------------------------------------- #
    # NEW: Image Quality Assessment (prototype quality-awareness layer,
    # not an accuracy predictor)
    # ---------------------------------------------------------------- #
    quality_report = st.session_state.get("ds_quality")
    if quality_report is not None:
        st.markdown('<div class="ds-section-head" style="margin-top:8px;">'
                     '<div class="eyebrow">Pre-flight Check</div>'
                     '<h2>Image quality assessment '
                     '<span class="ds-proto-tag">Prototype heuristic</span></h2>'
                     '<p>Classical image statistics only &mdash; not a prediction of model accuracy.</p>'
                     '</div>', unsafe_allow_html=True)
        with st.container(border=True):
            rating_class = quality_report.rating.lower()
            st.markdown(
                f'<span class="ds-status-badge {rating_class}">&#9679; {quality_report.rating.upper()}</span>',
                unsafe_allow_html=True,
            )
            q = quality_report.as_dict()
            tiles_html = '<div class="ds-quality-grid">'
            for label in ["Brightness (0-255)", "Contrast (std dev)", "Noise Estimate", "Sharpness (Laplacian Var)"]:
                tiles_html += (
                    f'<div class="ds-quality-tile"><div class="value">{q[label]}</div>'
                    f'<div class="label">{label}</div></div>'
                )
            tiles_html += "</div>"
            st.markdown(tiles_html, unsafe_allow_html=True)
            for w in quality_report.warnings:
                st.markdown(f'<div class="ds-warning-line">&#9888; {w}</div>', unsafe_allow_html=True)



    st.markdown("<div style='height:20px'></div>", unsafe_allow_html=True)



    img_col1, img_col2 = st.columns(2, gap="large")
    with img_col1:
        with st.container(border=True):
            st.image(st.session_state["ds_original"], width="stretch", caption="Original scan")
    with img_col2:
        with st.container(border=True):
            st.image(result["annotated_image"], width="stretch", caption="AI-detected objects")



    # ---------------------------------------------------------------- #
    # NEW: Optional prototype preprocessing -- Original vs Enhanced
    # ---------------------------------------------------------------- #
    enhanced_image = st.session_state.get("ds_enhanced_image")
    enhanced_result = st.session_state.get("ds_enhanced_result")
    if enhanced_image is not None and enhanced_result is not None:
        st.markdown("<div style='height:20px'></div>", unsafe_allow_html=True)
        with st.container(border=True):
            st.markdown('**Prototype preprocessing &mdash; Original vs Enhanced** '
                        '<span class="ds-proto-tag">Prototype heuristic</span>', unsafe_allow_html=True)
            st.caption("Denoise + CLAHE contrast enhancement, then re-run through the same model.")
            ec1, ec2 = st.columns(2, gap="large")
            with ec1:
                st.image(result["annotated_image"], width="stretch",
                         caption=f"Original — {len(result['detections'])} detection(s)")
            with ec2:
                st.image(enhanced_result["annotated_image"], width="stretch",
                         caption=f"Enhanced — {len(enhanced_result['detections'])} detection(s)")



    # ---------------------------------------------------------------- #
    # OPTIONAL SECOND LAYER RESULTS -- classical CV visual anomaly check
    # Only shown when a reference/baseline image was uploaded this run.
    # This is entirely separate from the YOLO output above.
    # ---------------------------------------------------------------- #
    anomaly_result = st.session_state.get("ds_anomaly_result")
    if anomaly_result is not None:
        st.markdown("<div style='height:20px'></div>", unsafe_allow_html=True)
        with st.container(border=True):
            st.markdown("**Unclassified Visual Anomaly Check (optional, second layer)**")
            st.caption("Classical CV heuristic \u2014 not a trained AI detection.")



            if not anomaly_result["reliable"]:
                st.warning(anomaly_result["warning"])
            else:
                st.image(
                    anomaly_result["annotated_image"],
                    width="stretch",
                    caption=f"{anomaly_result['count']} unclassified visual anomaly region(s) highlighted",
                )
                if anomaly_result["count"] == 0:
                    st.caption("No significant visual changes detected versus the reference image.")



    # ---------------------------------------------------------------- #
    # NEW: Acoustic-shadow analysis (prototype heuristic, not real
    # sonar physics)
    # ---------------------------------------------------------------- #
    shadow_results = st.session_state.get("ds_shadows", [])
    if shadow_results:
        st.markdown("<div style='height:20px'></div>", unsafe_allow_html=True)
        st.markdown('<div class="ds-section-head">'
                     '<div class="eyebrow">Secondary Cue</div>'
                     '<h2>Acoustic-shadow analysis '
                     '<span class="ds-proto-tag">Prototype heuristic</span></h2>'
                     '<p>Classical-CV shadow footprint estimate per detection &mdash; not validated sonar physics.</p>'
                     '</div>', unsafe_allow_html=True)
        with st.container(border=True):
            for i, sr in enumerate(shadow_results):
                det_label = detections[i]["class"] if i < len(detections) else f"Object {i+1}"
                rel_class = sr.reliability_label.lower()
                st.markdown(
                    f'<div class="ds-priority-card">'
                    f'<b>{det_label}</b> &nbsp;'
                    f'<span class="ds-status-badge {rel_class}">{sr.reliability_label} reliability</span>'
                    f'<div class="factors">Shadow area: {sr.shadow_area_px}px &middot; '
                    f'Shadow length: {sr.shadow_length_px}px &middot; '
                    f'Target/shadow ratio: {sr.target_shadow_ratio}<br>{sr.note}</div>'
                    f'</div>',
                    unsafe_allow_html=True,
                )
                    # ---------------------------------------------------------------- #
    # NEW: Explainable priority scoring (decision-support heuristic)
    # ---------------------------------------------------------------- #
    priority_results = st.session_state.get("ds_priorities", [])
    if priority_results:
        st.markdown("<div style='height:20px'></div>", unsafe_allow_html=True)
        st.markdown('<div class="ds-section-head">'
                     '<div class="eyebrow">Triage</div>'
                     '<h2>Explainable priority scoring '
                     '<span class="ds-proto-tag">Decision-support heuristic</span></h2>'
                     '<p>Combines confidence, size, quality, anomaly and shadow signals into a plain-language recommendation. Final judgment always rests with a human reviewer.</p>'
                     '</div>', unsafe_allow_html=True)
        with st.container(border=True):
            for i, pr in enumerate(priority_results):
                det_label = detections[i]["class"] if i < len(detections) else f"Object {i+1}"
                badge_class = pr.priority.lower().replace(" ", "-")
                css_class = "verify" if pr.priority == "REQUIRES HUMAN VERIFICATION" else pr.priority.lower()
                st.markdown(
                    f'<div class="ds-priority-card">'
                    f'<b>{det_label}</b> &nbsp;'
                    f'<span class="ds-status-badge {css_class}">{pr.priority}</span>'
                    f'<span style="color:var(--text-faint); font-family:var(--font-mono); font-size:12px;"> &middot; score {pr.score}/100</span>'
                    f'<div class="factors">{" &rarr; ".join(pr.factors)}</div>'
                    f'</div>',
                    unsafe_allow_html=True,
                )



    st.markdown("<div style='height:20px'></div>", unsafe_allow_html=True)



    res_col1, res_col2 = st.columns([1.3, 1], gap="large")



    with res_col1:
        with st.container(border=True):
            st.markdown("**Detected objects**")
            if not detections:
                st.markdown(
                    """
                    <div style="
                        padding: 24px;
                        text-align: center;
                        border: 1px solid rgba(255,176,32,0.35);
                        background: rgba(255,176,32,0.05);
                        border-radius: 4px;
                        margin-top: 10px;
                    ">
                        <div style="
                            font-family: var(--font-mono);
                            font-size: 28px;
                            color: var(--amber);
                            margin-bottom: 10px;
                        ">&#9673;</div>
                        <div style="
                            font-family: var(--font-display);
                            font-size: 20px;
                            font-weight: 700;
                            color: var(--text);
                            margin-bottom: 8px;
                        ">NO OBJECT DETECTED</div>
                        <div style="
                            font-size: 13px;
                            color: var(--text-dim);
                            line-height: 1.6;
                        ">
                            No trained object was detected above the current
                            confidence threshold.
                            <br>
                            Try lowering the confidence threshold and analyze again.
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
            else:
                chip_map = {c["css"]: c["chip"] for c in CLASSES}
                current_file_name = st.session_state["ds_file_name"]



                for idx, det in enumerate(detections):
                    css_cls = det["class"].lower()
                    chip = chip_map.get(css_cls, "#5EEAD4")
                    pct = det["confidence"] * 100
                    row_html = (
                        f'<div class="ds-det-row">'
                        f'<span class="ds-badge {css_cls}">{det["class"]}</span>'
                        f'<div class="ds-det-conf">{pct:.1f}%'
                        f'<div class="ds-conf-track"><div class="ds-conf-fill" style="width:{pct:.0f}%; background:{chip};"></div></div>'
                        f'</div></div>'
                    )
                    st.markdown(row_html, unsafe_allow_html=True)



                    feedback_key = f"fb_{current_file_name}_{idx}"
                    feedback_choice = st.radio(
                        "Is this detection correct?",
                        ["Correct", "Incorrect", "Unsure"],
                        key=feedback_key,
                        horizontal=True,
                        index=None,
                        label_visibility="collapsed",
                    )



                    saved_flag_key = f"{feedback_key}_saved_as"
                    if feedback_choice is not None and st.session_state.get(saved_flag_key) != feedback_choice:
                        save_feedback(
                            image_filename=current_file_name,
                            class_id=det["class_id"],
                            displayed_class=det["class"],
                            confidence=det["confidence"],
                            bbox=det["bbox"],
                            feedback=feedback_choice,
                            latitude=st.session_state.get("ds_latitude", ""),
                            longitude=st.session_state.get("ds_longitude", ""),
                        )
                        st.session_state[saved_flag_key] = feedback_choice



                    if feedback_choice is not None:
                        st.caption(f"Feedback recorded: {feedback_choice}")



                st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)
                st.caption(
                    "Human feedback is stored for future model improvement and retraining."
                )



    with res_col2:
        with st.container(border=True):
            st.markdown("**Detection report**")
            st.markdown(
                "<p style='font-size:13px;'>Export a plain-text summary with class breakdown, "
                "bounding boxes and current model metrics.</p>",
                unsafe_allow_html=True,
            )
            lat_val = st.session_state.get("ds_latitude", "")
            lon_val = st.session_state.get("ds_longitude", "")
            if lat_val.strip() and lon_val.strip():
                coord_text = f"&#128205; {lat_val.strip()}, {lon_val.strip()}"
            else:
                coord_text = "&#128205; Location not provided"
            st.markdown(f'<div class="report-coord-line">{coord_text}</div>', unsafe_allow_html=True)



            anomaly_result_for_report = st.session_state.get("ds_anomaly_result")
            report_anomaly_count = (
                anomaly_result_for_report["count"]
                if anomaly_result_for_report is not None and anomaly_result_for_report["reliable"]
                else None
            )



            report_text = build_text_report(
                file_name=st.session_state["ds_file_name"],
                conf_threshold=st.session_state["ds_conf"],
                detections=detections,
                inference_time_ms=result["inference_time_ms"],
                image_size=result["image_size"],
                model_metrics=MODEL_METRICS,
                latitude=lat_val,
                longitude=lon_val,
                anomaly_count=report_anomaly_count,
            )
            st.download_button(
                "\u2913  Download Report (.txt)",
                data=report_text,
                file_name=f"aquashield_report_{Path(st.session_state['ds_file_name']).stem}.txt",
                mime="text/plain",
                width="stretch",
            )



st.markdown("</div>", unsafe_allow_html=True)  # close #upload section



# ------------------------------------------------------------------ #
# Human Feedback Insights (read-only view over feedback_log.csv)
# ------------------------------------------------------------------ #
st.markdown('<div class="ds-section" id="feedback-insights">', unsafe_allow_html=True)
st.markdown(
    """
    <div class="ds-section-head">
      <div class="eyebrow">Feedback Loop</div>
      <h2><span class="sec-icon"><svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M21 12a8 8 0 0 1-8 8H4l2-3a8 8 0 1 1 15-5z"/></svg></span>Human Feedback Insights</h2>
      <p>A live view of the reviewer feedback collected in this session and prior
      sessions, read directly from the local feedback log.</p>
    </div>
    """,
    unsafe_allow_html=True,
)



feedback_df = None
feedback_read_error = False



if FEEDBACK_CSV_PATH.exists():
    try:
        feedback_df = pd.read_csv(FEEDBACK_CSV_PATH)
        if feedback_df.empty:
            feedback_df = None
    except Exception:
        feedback_read_error = True



if feedback_read_error:
    st.warning(
        "The feedback log exists but couldn't be read right now. "
        "It may be mid-write -- try refreshing in a moment."
    )
elif feedback_df is None:
    st.info("No feedback collected yet.")
else:
    with st.container(border=True):
        total_entries = len(feedback_df)
        counts = feedback_df["feedback"].value_counts()
        correct_n = int(counts.get("Correct", 0))
        incorrect_n = int(counts.get("Incorrect", 0))
        unsure_n = int(counts.get("Unsure", 0))



        fi1, fi2, fi3, fi4 = st.columns(4)
        fi1.metric("Total feedback entries", total_entries)
        fi2.metric("Correct", correct_n)
        fi3.metric("Incorrect", incorrect_n)
        fi4.metric("Unsure", unsure_n)



        decided_n = correct_n + incorrect_n
        if decided_n > 0:
            positive_rate = correct_n / decided_n * 100
            st.metric("Positive-feedback rate", f"{positive_rate:.1f}%")
        else:
            st.metric("Positive-feedback rate", "N/A")
        st.caption(
            "Prototype feedback statistic based on human reviewer input -- "
            'not a measure of model accuracy. "Unsure" entries are excluded '
            "from this rate."
        )



        st.markdown("<div style='height:10px'></div>", unsafe_allow_html=True)
        st.markdown("**Breakdown by displayed class**")
        if "displayed_class" in feedback_df.columns:
            class_breakdown = (
                feedback_df.groupby("displayed_class")["feedback"]
                .value_counts()
                .unstack(fill_value=0)
            )
            st.dataframe(class_breakdown, width="stretch")
        else:
            st.caption("No class information found in the feedback log.")



        # ------------------------------------------------------ #
        # NEW: Dataset-ready export (does not retrain the model)
        # ------------------------------------------------------ #
        st.markdown("<div style='height:10px'></div>", unsafe_allow_html=True)
        st.markdown("**Export feedback as a dataset-ready CSV**")
        st.caption(
            "Reformats feedback_log.csv into a clean CSV a person can review "
            "manually for future labeling work. Does not retrain the model "
            "and does not modify the original log."
        )
        if st.button("Prepare dataset-ready export"):
            try:
                export_df = export_dataset_ready_csv()
                st.session_state["ds_export_df"] = export_df
            except FileNotFoundError as e:
                st.warning(str(e))
        export_df = st.session_state.get("ds_export_df")
        if export_df is not None:
            st.download_button(
                "\u2913  Download dataset-ready CSV",
                data=export_df.to_csv(index=False),
                file_name="feedback_export_dataset_ready.csv",
                mime="text/csv",
                width="stretch",
            )



st.markdown("</div>", unsafe_allow_html=True)  # close #feedback-insights section



# ------------------------------------------------------------------ #
# NEW: Session History (in-memory, no database)
# ------------------------------------------------------------------ #
st.markdown('<div class="ds-section" id="history">', unsafe_allow_html=True)
st.markdown(
    """
    <div class="ds-section-head">
      <div class="eyebrow">This Session</div>
      <h2><span class="sec-icon"><svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><circle cx="12" cy="12" r="8"/><path d="M12 8v4l3 2"/></svg></span>Session history</h2>
      <p>A local, in-memory log of analyses run in this browser session. Clears on refresh — nothing here is written to disk.</p>
    </div>
    """,
    unsafe_allow_html=True,
)
history_df = get_history_df()
with st.container(border=True):
    if history_df.empty:
        st.info("No analyses run yet this session.")
    else:
        st.markdown('<div class="ds-history-wrap">', unsafe_allow_html=True)
        st.dataframe(history_df, width="stretch", hide_index=True)
        st.markdown('</div>', unsafe_allow_html=True)



        if PLOTLY_AVAILABLE:
            loc_df = build_location_dataframe(history_df)
            if not loc_df.empty:
                st.markdown("<div style='height:14px'></div>", unsafe_allow_html=True)
                st.markdown("**Analyzed locations** (manually entered coordinates only)")
                fig = render_map_figure(loc_df)
                if fig is not None:
                    st.plotly_chart(fig, width="stretch")
st.markdown("</div>", unsafe_allow_html=True)  # close #history section



# ------------------------------------------------------------------ #
# Slim footer
# ------------------------------------------------------------------ #
st.markdown(
    """
    <div class="ds-footer-slim">
      <span>
        &copy; 2026 AquaShield &mdash; research prototype,
        not certified for navigational use.
      </span>
      <span class="stack">
        <span>Python</span>
        <span>YOLO11n</span>
        <span>Streamlit</span>
        <span>OpenCV</span>
      </span>
    </div>
    """,
    unsafe_allow_html=True,
)