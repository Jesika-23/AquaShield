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

### Lower threshold

A lower threshold can display more possible detections, including weaker predictions.

### Higher threshold

A higher threshold displays fewer but generally more confident predictions.

The current default threshold is:

**0.50**

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
