# Intelligent Traffic and Road Safety Detection System

AI-powered traffic monitoring and road safety analysis using YOLO, OpenCV, and machine learning.

## 1. Project Overview

The Intelligent Traffic and Road Safety Detection System is an AI-powered project designed to detect vehicles, monitor traffic conditions, and analyze road traffic using computer vision and deep learning techniques.

The system uses YOLO object detection to identify vehicles in traffic images and videos. It also includes vehicle tracking, traffic density estimation, helmet detection, number plate recognition, and traffic violation monitoring.

A Streamlit dashboard provides a user-friendly interface to view traffic statistics, reports, and detection results.

## 2. Problem Statement

Traditional traffic monitoring systems often require manual observation and monitoring. This can be time-consuming and may lead to inaccurate traffic analysis.

This project aims to develop an automated traffic monitoring system using artificial intelligence and computer vision to improve vehicle detection, traffic analysis, and road safety monitoring.

## 3. Objectives

- Detect vehicles in traffic images and videos.
- Classify vehicles into different categories.
- Track vehicles across video frames.
- Estimate traffic density.
- Detect motorcycle helmets.
- Recognize vehicle number plates.
- Monitor traffic signals and violations.
- Generate traffic analysis reports.
- Develop an interactive traffic monitoring dashboard.

## 4. Technologies Used

| Technology | Purpose |
|---|---|
| Python | Programming language |
| OpenCV | Image and video processing |
| YOLOv8 | Object detection |
| PyTorch | Deep learning framework |
| ByteTrack | Object tracking |
| EasyOCR | Number plate text recognition |
| Pandas | Data analysis |
| Streamlit | Interactive dashboard |
| SQLite | Database management |
| Google Colab | Initial model development |
| VS Code | Local development environment |

## 5. Project Modules

### Module 1: Vehicle Detection
Uses YOLO to identify vehicles such as cars, buses, trucks, and motorcycles.

### Module 2: Vehicle Tracking
Uses object tracking to assign tracking IDs to detected vehicles across video frames.

### Module 3: Traffic Density Estimation
Analyzes vehicle counts across video frames to estimate traffic density.

### Module 4: Helmet Detection
Uses a trained YOLO model to detect motorcycle riders with and without helmets.

### Module 5: Number Plate Recognition
Detects vehicle number plates and applies OCR to extract readable text.

### Module 6: Traffic Signal and Violation Detection
Analyzes traffic signals and identifies potential traffic rule violations.

### Module 7: Data Analysis and Reporting
Uses Pandas to organize detection results and generate CSV and PDF reports.

### Module 8: Dashboard
Uses Streamlit to display vehicle statistics, traffic density information, charts, reports, and detection results.

## 6. Project Structure

```text
Intelligent traffic code/
│
├── app.py
├── database.py
├── detect_video.py
├── helmet_detection.py
├── number_plate_detection.py
├── traffic_light_detection.py
├── traffic_density.py
├── traffic_violation.py
├── track_video.py
├── pdf_report.py
├── README.md
│
├── data/
├── models/
├── outputs/
├── reports/
├── runs/
└── venv/
```

## 7. Experimental Results

The project has been tested using traffic videos and trained detection models.

Vehicle detection results from the latest experiment:

- Cars: 494
- Motorcycles: 22
- Buses: 50
- Trucks: 42
- Total vehicle detections: 608
- Total video frames processed: 559

Other modules generate separate reports for vehicle tracking, traffic density, helmet detection, number plate recognition, and traffic violations.

Results may vary depending on the input video, model, and detection settings.

## 8. How to Run the Project

### Step 1: Activate the virtual environment

```bash
source venv/bin/activate
```

### Step 2: Install dependencies

```bash
python -m pip install streamlit pandas opencv-python ultralytics
```

Install any additional dependencies required by the individual modules.

### Step 3: Run the dashboard

```bash
python -m streamlit run app.py
```

### Step 4: Open in browser

```text
http://localhost:8501
```

## 9. Future Enhancements

- Improve real-time vehicle detection.
- Enhance number plate recognition accuracy.
- Improve helmet detection performance.
- Integrate real-time traffic monitoring.
- Improve traffic violation detection.
- Add vehicle speed estimation.
- Enhance database-based analytics.
- Deploy the dashboard online.

## 10. Conclusion

The Intelligent Traffic and Road Safety Detection System demonstrates how computer vision and deep learning can be applied to automated traffic monitoring.

The project combines vehicle detection, tracking, traffic density analysis, helmet detection, number plate recognition, traffic signal monitoring, reporting, and dashboard development.

It provides a foundation for developing intelligent traffic monitoring and road safety applications.

---

**Project:** Intelligent Traffic and Road Safety Detection System  
**Programming Language:** Python  
**Development Environment:** Google Colab and Visual Studio Code