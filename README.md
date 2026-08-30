# 🌊 AquaShield

## AI-Powered Underwater Object Detection & Monitoring System

AquaShield is an AI-powered computer vision prototype designed to analyze underwater/sonar-style images and automatically detect objects using a trained YOLO object detection model.

The system combines AI-based object detection with image-quality assessment, confidence filtering, priority scoring, acoustic-shadow cues, reporting, detection history, feedback logging, and map support to provide a complete prototype workflow for underwater image analysis.

---

## 🚀 Live Demo

**Try AquaShield:**

https://jesika-23-aquashield-app-bnkxiv.streamlit.app/

---

## 📌 Project Overview

Underwater environments can contain objects that are difficult and time-consuming to identify manually, particularly when large numbers of images need to be reviewed.

AquaShield provides a simple AI-assisted interface where users can upload an underwater/sonar-style image and obtain detection and analysis results.

The application allows users to:

- 📤 Upload an underwater image
- 🤖 Analyze the image using a trained YOLO model
- 🎚️ Adjust the confidence threshold
- 🖼️ View bounding boxes around detected objects
- 📊 View object classes and confidence scores
- 🔍 Assess image quality
- 🧭 Review detection priority
- 🌑 Examine acoustic-shadow cues
- 📝 Generate a structured report
- 🗺️ Add location information
- 📜 Review detection history
- 💬 Provide feedback on detections

AquaShield is developed as a **research and educational prototype** demonstrating how AI-based computer vision can support underwater monitoring and image-review workflows.

---

# 🎯 Key Features

## 1. 📤 Image Upload

Users can upload an underwater or sonar-style image through the application.

The uploaded image becomes the input for the AI detection and analysis pipeline.

---

## 2. 🤖 AI Object Detection

AquaShield uses a trained **YOLO object detection model** to analyze uploaded images.

For each detected object, the model provides:

- Object class
- Bounding box
- Confidence score

### Supported Detection Classes

| Class ID | Object |
|----------|--------|
| 0 | ✈️ Aircraft |
| 1 | 🐟 Fish |
| 2 | 🪸 Reef |
| 3 | 🚢 Ship |

---

## 3. 🎚️ Confidence Threshold

The application provides a confidence-threshold control that allows the user to decide how confident the AI prediction must be before it is displayed.

**Lower threshold:**  
A lower threshold can display more possible detections, including weaker predictions.

**Higher threshold:**  
A higher threshold displays fewer but generally more confident predictions.

The current default threshold is **0.50**.

Users can still adjust the threshold according to the image and analysis requirement.

---

## 4. 🖼️ Visual Detection Results

After analysis, AquaShield displays the processed image with bounding boxes around detected objects.

The visual results help users compare the AI predictions with the actual image.

Detection information can include:

- Object class
- Confidence score
- Bounding-box location

---

## 5. 📊 Detection Analysis

AquaShield provides additional information beyond the bounding boxes.

The analysis helps users understand what the model detected and provides supporting information such as:

- Detected object class
- Confidence
- Object characteristics
- Priority information
- Supporting analysis signals

This makes the system an **AI-assisted analysis tool** rather than simply an image-labeling interface.

---

## 6. 🔍 Image Quality Assessment

The application includes an image-quality assessment component.

Underwater imagery can be affected by factors such as visibility, contrast, noise, and other image conditions.

Poor-quality input can affect computer-vision predictions.

AquaShield therefore provides quality-related information alongside the detection results to give the reviewer additional context.

---

## 7. 🧭 Priority Scoring

AquaShield includes a prototype priority-scoring layer to help reviewers identify detections that may require additional attention.

Possible priority levels include:

- 🔴 **HIGH**
- 🟠 **REQUIRES HUMAN VERIFICATION**
- 🟡 **MEDIUM**
- 🟢 **LOW**

The priority explanation can consider multiple signals, including:

- Detection confidence
- Object size
- Image quality
- Shadow-related cues
- Anomaly-related signals

This feature is intended as **decision support** and does not replace human judgment.

---

## 8. 🌑 Acoustic-Shadow Analysis

The application contains a prototype acoustic-shadow analysis component.

It estimates shadow-related characteristics around detected objects and uses them as an additional supporting signal.

> **Important:** This is a heuristic computer-vision approach and is **not a validated sonar-physics model**.

The feature demonstrates how secondary visual cues could potentially support underwater object analysis.

---

## 9. 📝 Report Generation

AquaShield includes a reporting module that generates a structured text summary of the analysis.

The report can include information such as:

- Image analyzed
- Detected objects
- Confidence values
- Priority information
- Supporting analysis

This makes detection results easier to document and review.

---

## 10. 🗺️ Map & Location Support

AquaShield contains a map-view component for associating analysis with geographic information.

Users can enter coordinates manually and visualize the corresponding location.

This provides a foundation for future integration with:

- GPS data
- Survey missions
- Autonomous underwater vehicles
- Monitoring operations
- Geographic detection records

---

## 11. 📜 Detection History

The application includes history functionality for tracking previous analysis activity.

This provides a foundation for reviewing detection results across multiple analysis sessions.

---

## 12. 💬 Feedback Logging

AquaShield includes a feedback mechanism that allows users to review AI detections and provide human feedback.

Feedback is stored in:

```text
feedback_log.csv

The feedback log can be used for future analysis, labeling, and model improvement.

🧠 How AquaShield Works

The overall workflow is:

User
  ↓
Upload Underwater / Sonar-Style Image
  ↓
Image Preprocessing
  ↓
Image Quality Assessment
  ↓
YOLO AI Model
  ↓
Object Detection
  ↓
Confidence Filtering
  ↓
Visual Detection Results
  ↓
Detection Analysis
  ↓
Priority & Supporting Analysis
  ↓
Report Generation
  ↓
Feedback & Detection History
🏗️ System Architecture
                    ┌──────────────────────┐
                    │        User          │
                    └──────────┬───────────┘
                               │
                               ▼
                    ┌──────────────────────┐
                    │   Streamlit UI       │
                    │   Image Upload        │
                    └──────────┬───────────┘
                               │
                               ▼
                    ┌──────────────────────┐
                    │ Image Preprocessing  │
                    │ & Quality Assessment │
                    └──────────┬───────────┘
                               │
                               ▼
                    ┌──────────────────────┐
                    │    YOLO Detector     │
                    │      best.pt         │
                    └──────────┬───────────┘
                               │
                               ▼
                    ┌──────────────────────┐
                    │ Detection Results    │
                    │ Class + Confidence   │
                    │ + Bounding Boxes     │
                    └──────────┬───────────┘
                               │
                 ┌─────────────┼─────────────┐
                 ▼             ▼             ▼
          ┌────────────┐ ┌────────────┐ ┌────────────┐
          │  Priority  │ │   Shadow   │ │   Quality  │
          │  Scoring   │ │  Analysis  │ │   Analysis │
          └─────┬──────┘ └─────┬──────┘ └─────┬──────┘
                │              │              │
                └──────────────┼──────────────┘
                               ▼
                    ┌──────────────────────┐
                    │ Analysis & Reporting │
                    └──────────┬───────────┘
                               │
                    ┌──────────┴───────────┐
                    ▼                      ▼
             ┌─────────────┐       ┌─────────────┐
             │  Feedback   │       │   History   │
             └─────────────┘       └─────────────┘
🛠️ Technology Stack
Technology	Purpose
Python	Core application and AI processing
Streamlit	Web application interface
Ultralytics YOLO	Object detection
OpenCV	Image processing and computer vision
Pandas	Data handling and feedback logging
Pillow	Image loading and processing
Plotly	Data visualization
🤖 AI Model

AquaShield uses a trained YOLO11n object detection model.

The trained model is stored in:

model/best.pt

The model was trained to detect four classes:

0 → Aircraft
1 → Fish
2 → Reef
3 → Ship

The model provides bounding-box detections and confidence scores for objects identified in an input image.

📊 Model Performance
Metric	Value
Model	YOLO11n
Precision	47.3%
Recall	55.3%
mAP@0.50	51.5%
mAP@0.50-0.95	31.4%
Training Images	402
Validation Images	110

These metrics represent the performance observed during the model-development process and should not be interpreted as production-level accuracy.

📁 Project Structure
AquaShield/
│
├── app.py
├── requirements.txt
├── feedback_log.csv
├── .gitignore
│
├── model/
│   └── best.pt
│
├── assets/
│   └── style.css
│
├── utils/
│   ├── __init__.py
│   ├── inference.py
│   ├── preprocess.py
│   ├── quality.py
│   ├── priority.py
│   ├── shadow.py
│   ├── report.py
│   ├── history.py
│   ├── feedback.py
│   └── map_view.py
│
└── text/
    └── images/
🔄 Application Workflow
Step 1 — Upload

The user selects an underwater or sonar-style image through the Streamlit interface.

Step 2 — Preprocessing

The uploaded image is prepared for analysis.

Step 3 — Quality Assessment

The system evaluates image characteristics that may affect detection reliability.

Step 4 — AI Detection

The trained YOLO model analyzes the image and identifies objects.

Step 5 — Confidence Filtering

Only detections meeting the selected confidence threshold are presented.

The default threshold is 0.50.

Step 6 — Visual Results

Detected objects are displayed using bounding boxes and associated information.

Step 7 — Supporting Analysis

The application provides additional analysis such as image quality, priority scoring, and shadow-related cues.

Step 8 — Reporting

The user can generate a structured summary of the analysis.

Step 9 — Location

Coordinates can be entered when geographic information is available.

Step 10 — Feedback & History

Users can review previous analysis activity and provide feedback on detections.

🧑‍💻 Getting Started
Prerequisites

Make sure Python is installed on your system.

A virtual environment is recommended.

1. Clone the repository
git clone https://github.com/Jesika-23/AquaShield.git
cd AquaShield
2. Create a virtual environment
python -m venv venv
3. Activate the virtual environment

Windows:

venv\Scripts\activate

Linux / macOS:

source venv/bin/activate
4. Install dependencies
pip install -r requirements.txt
5. Run the application
streamlit run app.py

The application will open in your browser.

🌐 Deployment

AquaShield is deployed using Streamlit Community Cloud and connected to the GitHub repository.

Live Application

https://jesika-23-aquashield-app-bnkxiv.streamlit.app/

The deployment automatically uses the project's requirements.txt file to install the required Python dependencies.

🔮 Future Enhancements

AquaShield can be extended in several directions:

Larger and more diverse underwater datasets
Improved object-detection accuracy
Real sonar-data integration
Real-time video or sonar-stream detection
GPS and mission-data integration
Advanced anomaly detection
Improved acoustic-shadow modeling
More robust underwater image enhancement
Human-feedback-based model improvement
Integration with autonomous underwater vehicles
Multi-frame object tracking
Cloud-based detection history
More advanced monitoring dashboards
⚠️ Limitations & Disclaimer

AquaShield is a research and educational prototype.

The current detection model was trained using a relatively small dataset. Model predictions may therefore contain false positives or false negatives.

The image-quality assessment, priority scoring, anomaly signals, and acoustic-shadow analysis are prototype heuristic components. They should be considered supporting decision-making features rather than scientifically validated sonar measurements.

In particular, the acoustic-shadow component is not a validated sonar-physics model.

AquaShield is not certified for navigation, safety-critical operations, or autonomous decision-making.

All AI-generated detections should be reviewed and verified by a human operator before any real-world action is taken.

📌 Project Status

Status: Functional Research Prototype

The current prototype demonstrates an end-to-end workflow for:

Image Upload
      ↓
AI Detection
      ↓
Confidence Filtering
      ↓
Visual Results
      ↓
Supporting Analysis
      ↓
Priority Assessment
      ↓
Reporting
      ↓
Feedback & History
📄 License & Notes

© 2026 AquaShield — Research prototype for educational and demonstration purposes.

This project is intended to demonstrate the potential of AI-assisted underwater image analysis and monitoring workflows.
