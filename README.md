# Intelligent Traffic and Road Safety Detection System

## 1. Project Overview

The Intelligent Traffic and Road Safety Detection System is an AI-powered project designed to detect vehicles, monitor traffic conditions, and analyze road traffic using computer vision and deep learning techniques.

The system uses the YOLO object detection model to identify vehicles in traffic images and videos. It also explores vehicle tracking and traffic density estimation to support intelligent traffic monitoring.

A Streamlit dashboard provides a user-friendly interface to view traffic statistics and upload traffic images or videos.

## 2. Problem Statement

Traditional traffic monitoring systems often require manual observation and monitoring. This can be time-consuming and may lead to inaccurate traffic analysis.

This project aims to develop an automated traffic monitoring system using artificial intelligence and computer vision to improve vehicle detection and traffic analysis.

## 3. Objectives

- Detect vehicles in traffic images and videos.
- Classify vehicles into different categories.
- Track vehicles across video frames.
- Estimate traffic density.
- Generate traffic analysis reports.
- Develop an interactive traffic monitoring dashboard.
- Explore road safety monitoring using AI.

## 4. Technologies Used

| Technology | Purpose |
|---|---|
| Python | Programming language |
| OpenCV | Image and video processing |
| YOLOv8 | Vehicle detection |
| PyTorch | Deep learning framework |
| ByteTrack | Object tracking |
| Pandas | Data analysis |
| Streamlit | Interactive dashboard |
| Google Colab | Initial model development |
| VS Code | Local development environment |

## 5. Project Modules

### Module 1: Vehicle Detection
Uses YOLO to identify vehicles such as cars, buses, trucks, and motorcycles.

### Module 2: Vehicle Tracking
Uses ByteTrack to assign tracking IDs to detected vehicles across video frames.

### Module 3: Traffic Density Estimation
Analyzes vehicle counts across video frames to estimate traffic density.

### Module 4: Data Analysis and Reporting
Uses Pandas to organize detection results and generate CSV reports.

### Module 5: Dashboard
Uses Streamlit to display vehicle statistics, traffic density information, charts, and uploaded traffic media.

## 6. Project Structure

```text
Intelligent traffic code/
│
├── traffic_detection.ipynb
├── app.py
├── README.md
│
├── data/
│   ├── images/
│   └── videos/
│
├── models/
├── outputs/
├── reports/
└── venv/
```

## 7. Current Implementation

The initial development and experiments were performed using Google Colab.

The following results were obtained during the experiments:

- Total tracked vehicle IDs: 37
- Cars: 19
- Buses: 11
- Trucks: 7
- Motorcycles: 0
- Average vehicles per frame: 0.96
- Maximum vehicles in a frame: 7
- Total video frames analyzed: 830
- Busiest frame: 303

These are results from the initial traffic video experiment, not live dashboard measurements.

The Streamlit dashboard has also been created and launched locally in VS Code.

## 8. How to Run the Project

### Step 1: Activate the virtual environment

```bash
source venv/bin/activate
```

### Step 2: Install dependencies

```bash
python -m pip install streamlit pandas opencv-python ultralytics
```

### Step 3: Run the dashboard

```bash
python -m streamlit run app.py
```

### Step 4: Open in browser

```text
http://localhost:8501
```

## 9. Future Enhancements

- Integrate real-time YOLO detection into the dashboard.
- Implement helmet detection.
- Implement number plate recognition.
- Detect traffic rule violations.
- Add vehicle speed estimation.
- Integrate a database for storing violation records.
- Generate detailed traffic analysis reports.
- Develop a real-time traffic monitoring interface.

## 10. Conclusion

The Intelligent Traffic and Road Safety Detection System demonstrates how computer vision and deep learning can be applied to automated traffic monitoring.

The project combines vehicle detection, tracking, traffic density analysis, data reporting, and dashboard development. Further enhancements will focus on integrating road safety violation detection and real-time traffic monitoring.

---

**Project:** Intelligent Traffic and Road Safety Detection System  
**Development Environment:** Google Colab and Visual Studio Code  
**Programming Language:** Python
