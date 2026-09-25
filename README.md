# 🌊 AquaShield

## AI-Powered Sonar-Style Image Analysis Prototype

**Smart India Hackathon 2026 | Problem Statement SIH26057 | Team Hackcult_80 | R.M.D. Engineering College**

AquaShield is a research and educational prototype for AI-assisted analysis of underwater and sonar-style images.

The current system uses a YOLO object-detection model to identify objects in uploaded images and presents the results through a Streamlit web application. It combines object detection, confidence filtering, image-quality assessment, prototype priority scoring, shadow-related heuristic cues, reporting, location support, history, and feedback logging.

AquaShield demonstrates how AI can support future marine-debris monitoring and seabed-survey workflows. It is not a production-ready or safety-certified navigation system.

---

## 🚀 Live Demo

Try the current Streamlit prototype:

[Open AquaShield Live Demo](https://jesika-23-aquashield-app-bnkxiv.streamlit.app/)

Source code: [github.com/Jesika-23/Hackcult_AquaShield](https://github.com/Jesika-23/Hackcult_AquaShield)

## Prototype Dashboard

<img width="1917" height="872" alt="image" src="https://github.com/user-attachments/assets/b5eacf53-ce5b-4f98-9a58-9abdb0504da3" />


## 📌 Prototype Status

**Status: Functional Streamlit Prototype — Approximately 40% of Planned System**

The current public application demonstrates the core AI-assisted image-analysis workflow.

### Completed in Current Prototype

- Upload underwater or sonar-style images
- Image preprocessing using OpenCV
- YOLO11n object detection
- Bounding-box visualization
- Confidence-threshold filtering
- Image-quality assessment
- Prototype priority scoring
- Shadow-related heuristic analysis
- Structured report generation
- Manual map and location support
- Detection history
- Human feedback logging
- Deployment as an integrated Streamlit application on Streamlit Community Cloud

### Planned Future Extensions

- Training on real, labeled marine-debris sonar datasets
- Marine-debris-specific classes such as ghost nets, plastic debris, wreckage, and seabed hazards
- Separation into FastAPI backend and Streamlit frontend
- Dockerized deployment on cloud (for example AWS) and edge infrastructure
- Real geo-referencing using survey navigation data
- Debris clustering and prioritized debris maps
- Navigation-risk and ecological-sensitivity GIS layers
- ONNX Runtime optimization for lightweight CPU inference
- Field validation with hydrographic survey agencies, port authorities, and marine-cleanup organizations

The current Streamlit deployment is the integrated prototype being demonstrated. The additional architecture, dataset, and deployment modules are planned future extensions.

---

## 🎯 Project Overview

Marine debris, abandoned fishing gear, plastic waste, wreckage, and other seabed hazards can affect marine ecosystems, fisheries, and navigation safety.

Survey agencies may collect large volumes of side-scan sonar or underwater imagery. Reviewing these images manually is slow, subjective, and difficult to scale.

AquaShield provides an AI-assisted workflow in which users can upload an underwater or sonar-style image and receive object-detection results with supporting analysis.

The project was developed by Team Hackcult_80 from R.M.D. Engineering College for Smart India Hackathon 2026, Problem Statement **SIH26057**: *AI-Powered Automated Underwater Marine Debris and Anomaly Detection Using Side-Scan Sonar Imagery*.

AquaShield is designed as a decision-support prototype. It assists human reviewers and does not replace expert judgment.

---

## 🧠 Problem Statement

Underwater and sonar imagery can contain objects that are difficult and time-consuming to identify manually.

Current review processes can be:

- Slow when large numbers of images must be inspected
- Subjective across different analysts
- Difficult to scale for large seabed survey areas
- Limited in their ability to prioritize detections for further review

AquaShield explores how AI-based object detection and supporting analysis can help reviewers identify potential objects faster and organize their attention more effectively.

---

## ✨ Key Features

### 1. 📤 Image Upload

Users can upload an underwater or sonar-style image through the Streamlit application.

The uploaded image becomes the input for the AI detection and analysis pipeline.

### 2. 🤖 AI Object Detection

AquaShield uses a trained YOLO11n object-detection model to analyze uploaded images.

For each detected object, the model provides:

- Object class
- Bounding box
- Confidence score

#### Prototype Detection Classes

| Class ID | Prototype Class |
|---:|---|
| 0 | ✈️ Aircraft |
| 1 | 🐟 Fish |
| 2 | 🪸 Reef |
| 3 | 🚢 Ship |

> **Note:** These are prototype or proxy classes used to demonstrate the object-detection pipeline. They are not final marine-debris categories. Future versions will require training on labeled marine-debris and real sonar datasets for classes such as ghost nets, plastic debris, wreckage, and other seabed hazards.

### 3. 🎚️ Confidence Threshold

The application includes a confidence-threshold control.

This allows the user to select how confident the AI prediction must be before it is displayed.

- **Lower threshold:** Displays more possible detections, including weaker predictions.
- **Higher threshold:** Displays fewer detections but generally with greater confidence.

The default threshold is **0.50**.

Users can adjust the threshold based on image quality and analysis requirements.

### 4. 🖼️ Visual Detection Results

After analysis, AquaShield displays the processed image with bounding boxes around detected objects.

The visual result helps users compare AI predictions with the original image.

Detection information can include:

- Object class
- Confidence score
- Bounding-box position
- Supporting analysis information

### 5. 📊 Detection Analysis

AquaShield provides supporting information beyond bounding boxes.

The analysis may include:

- Detected object class
- Detection confidence
- Object size or bounding-box characteristics
- Prototype priority information
- Image-quality context
- Shadow-related heuristic cues

This makes the application an AI-assisted analysis prototype rather than only an image-labeling interface.

### 6. 🔍 Image Quality Assessment

The application includes an image-quality assessment component.

Image quality can influence object-detection reliability. Underwater and sonar-style images may be affected by factors such as:

- Low contrast
- Noise
- Blur
- Poor visibility
- Uneven intensity

AquaShield provides quality-related information alongside detections to give reviewers additional context.

### 7. 🧭 Prototype Priority Scoring

AquaShield includes a prototype priority-scoring layer to help reviewers identify detections that may require additional attention.

Possible priority levels include:

- 🔴 **HIGH**
- 🟠 **REQUIRES HUMAN VERIFICATION**
- 🟡 **MEDIUM**
- 🟢 **LOW**

The prototype priority score can consider signals such as:

- Detection confidence
- Object size
- Image quality
- Shadow-related cues
- Anomaly-related signals

This feature is intended for decision support only and does not replace human review.

In a future marine-debris system, priority scoring can be extended using debris size, navigation risk, ecological sensitivity, survey location, and environmental context.

### 8. 🌑 Shadow-Related Analysis

The application contains a prototype shadow-related analysis component.

It estimates visual characteristics around detected objects and uses them as an additional supporting signal.

> **Important:** This component uses heuristic computer-vision logic. It is not a validated sonar-physics or acoustic-shadow model.

Future work can improve this module using real side-scan sonar data and validated acoustic-shadow models.

### 9. 📝 Report Generation

AquaShield includes a reporting module that generates a structured analysis summary.

The report can include:

- Image analyzed
- Detected objects
- Confidence values
- Priority information
- Image-quality information
- Supporting analysis results

This makes results easier to document, review, and share.

### 10. 🗺️ Map and Location Support

AquaShield contains a map-view component for associating an analysis with geographic information.

Users can enter coordinates manually and visualize the corresponding location.

This provides a foundation for future integration with:

- GPS data
- Survey missions
- Autonomous underwater vehicles
- Vessel navigation data
- Geographic detection records
- Seabed debris maps

### 11. 📜 Detection History

The application includes history functionality for tracking previous analysis activity.

This creates a foundation for reviewing results across multiple analysis sessions.

### 12. 💬 Feedback Logging

AquaShield includes a feedback mechanism that allows users to review AI detections and provide human feedback.

Feedback is stored in:

```text
feedback_log.csv
```

In future versions, feedback can support data labeling, error analysis, and model improvement.

---

## 🧠 How AquaShield Works

The current prototype workflow is:

```text
User
  ↓
Upload Underwater / Sonar-Style Image
  ↓
Image Preprocessing
  ↓
Image Quality Assessment
  ↓
YOLO11n Object Detection
  ↓
Confidence Filtering (default threshold 0.50)
  ↓
Bounding Boxes and Detection Results
  ↓
Prototype Priority and Shadow-Related Analysis
  ↓
Report Generation (optional manual location)
  ↓
Feedback and Detection History
```

---

## 🏗️ Current System Architecture

```text
                    ┌──────────────────────┐
                    │        User          │
                    └──────────┬───────────┘
                               │
                               ▼
                    ┌──────────────────────┐
                    │     Streamlit UI     │
                    │    Image Upload      │
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
                    │  Detection Results   │
                    │ Class + Confidence   │
                    │ + Bounding Boxes     │
                    └──────────┬───────────┘
                               │
                 ┌─────────────┼─────────────┐
                 ▼             ▼             ▼
          ┌────────────┐ ┌────────────┐ ┌────────────┐
          │  Priority  │ │ Shadow-    │ │  Quality   │
          │  Scoring   │ │ Related    │ │  Analysis  │
          │            │ │ Analysis   │ │            │
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
```

---

## 🛠️ Technology Stack

| Technology | Purpose |
|---|---|
| Python 3.12 (recommended) | Core application and AI processing |
| Streamlit | Web application interface |
| Ultralytics YOLO11n | Object detection |
| OpenCV | Image preprocessing and computer vision |
| Pandas | Data handling and feedback logging |
| Pillow | Image loading and processing |
| Plotly | Data visualization |
| Streamlit Community Cloud | Current deployment platform |

---

## 🤖 AI Model

AquaShield uses a trained **YOLO11n object-detection model** in the current prototype.

The trained model is stored in:

```text
model/best.pt
```

The model detects the following current prototype classes:

```text
0 → Aircraft
1 → Fish
2 → Reef
3 → Ship
```

The model provides bounding boxes and confidence scores for objects identified in an uploaded image.

> The current model demonstrates the AI-detection workflow. It is not yet a validated marine-debris detector and should not be used for real navigation, cleanup, or safety-critical decisions.

---

## 📊 Current Model Performance

| Metric | Value |
|---|---:|
| Model | YOLO11n |
| Precision | 47.3% |
| Recall | 55.3% |
| mAP@0.50 | 51.5% |
| mAP@0.50–0.95 | 31.4% |
| Training Images | 402 |
| Validation Images | 110 |

These metrics represent performance observed during model development on the current prototype dataset.

They should not be interpreted as production-level performance or as final marine-debris detection accuracy.

---

## 📁 Project Structure

```text
Hackcult_AquaShield/
│
├── app.py
├── requirements.txt
├── feedback_log.csv
├── .gitignore
├── README.md
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
```

---

## 🧑‍💻 Getting Started

### Prerequisites

- Python 3.12 (recommended; Python 3.13 caused PyTorch errors during development)
- A virtual environment is recommended

### 1. Clone the Repository

```bash
git clone https://github.com/Jesika-23/Hackcult_AquaShield.git
cd Hackcult_AquaShield
```

### 2. Create a Virtual Environment

```bash
python -m venv venv
```

### 3. Activate the Virtual Environment

**Windows**

```bash
venv\Scripts\activate
```

**Linux / macOS**

```bash
source venv/bin/activate
```

### 4. Install Dependencies

```bash
pip install -r requirements.txt
```

### 5. Run the Application

```bash
streamlit run app.py
```

The application will open in your browser.

---

## 🌐 Deployment

AquaShield is currently deployed using **Streamlit Community Cloud** and connected to the GitHub repository.

The deployed application uses the project `requirements.txt` file to install the required dependencies.

> The current public deployment is the integrated Streamlit prototype. Future versions may separate the system into a FastAPI backend and Streamlit frontend, with Docker-based deployment on cloud platforms such as **AWS**, or on edge infrastructure.

---

## 🔮 Future Enhancements

AquaShield can be extended through:

- Larger and more diverse real sonar datasets
- Labeled marine-debris datasets
- Improved object-detection accuracy
- Real side-scan sonar integration
- Debris-specific model classes
- FastAPI backend and Streamlit frontend separation
- Dockerized cloud deployment (for example on AWS) and edge deployment
- ONNX Runtime optimization for lightweight CPU inference
- GPS, survey-mission, and vessel-data integration
- Geo-referencing and debris clustering
- Navigation-risk layers
- Ecological-sensitivity layers
- Advanced anomaly detection
- Validated acoustic-shadow modeling
- Human-feedback-based model improvement
- Multi-frame object tracking
- Cloud-based detection history
- Advanced mapping and monitoring dashboards
- Field trials with survey agencies and marine-cleanup organizations

---

## ⚠️ Limitations and Disclaimer

AquaShield is a **research and educational prototype**.

The current detection model was trained on a relatively small dataset and uses prototype or proxy object classes. Predictions may contain false positives or false negatives.

The image-quality assessment, priority scoring, anomaly-related signals, map/location support, and shadow-related analysis are prototype heuristic components.

They should be treated as supporting decision-making features, not scientifically validated sonar measurements.

In particular:

- The shadow-related component is **not a validated acoustic-sonar physics model**
- The current object classes are **not final marine-debris categories**
- The model is **not validated on operational marine-debris survey data**
- Location support is manual; there is no automatic geo-referencing
- Feedback is logged for future use, but the model does not learn from it yet
- The system is **not certified for navigation or safety-critical operations**
- All detections must be reviewed and verified by a qualified human operator before real-world action is taken

---

## 👥 Team

**Team Hackcult_80** — R.M.D. Engineering College

- Haripriya S [Team Lead]
- Jesika M V
- Kodavala Dhurgasree
- Rupendra M
- Pranav Mani S
- Sanjay Raaj D

---

## 📄 License and Notes

© 2026 AquaShield — Research prototype for educational and demonstration purposes.

This project uses Ultralytics YOLO, which is released under the AGPL-3.0 license. Please review the Ultralytics licensing terms before reusing or redistributing the model or code.

This project demonstrates the potential of AI-assisted underwater and sonar-style image analysis for future marine monitoring, survey review, and debris-prioritization workflows.
