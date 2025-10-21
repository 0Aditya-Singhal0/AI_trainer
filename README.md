# Genyx Fitness AI Trainer - Advanced Exercise Recognition and Form Analysis

A comprehensive AI-powered fitness system that combines real-time exercise recognition, repetition counting, and detailed form analysis. This system leverages computer vision, pose estimation, and machine learning to provide personalized fitness coaching through advanced form analysis.

## Table of Contents
- [Overview](#overview)
- [Architecture](#architecture)
- [API Documentation](#api-documentation)
- [Reference-Based Analysis](#reference-based-analysis)
- [Response Format](#response-format)
- [How to Start the Server](#how-to-start-the-server)
- [Original Repository Credit](#original-repository-credit)

## Overview

This project is built upon the foundation of "Fitness AI Trainer With Automatic Exercise Recognition and Counting" with enhanced form analysis capabilities. The system provides:

- **Real-time Exercise Recognition**: Automatically detects exercise types using LSTM models
- **Repetition Counting**: Accurate counting of exercise repetitions using angle-based logic
- **Advanced Form Analysis**: Detailed form feedback using reference-based comparison
- **Dual Analysis Modes**: Both automatic detection and user-specified exercise analysis
- **Actionable Feedback**: Specific recommendations for form improvement

The system supports four main exercise types: squats, push-ups, bicep curls, and shoulder presses, with the capability to expand to additional exercises.

## Architecture

### Core Components

1. **API Layer (`api.py`)**: FastAPI endpoints for video and image analysis
2. **Service Layer (`services.py`)**: Business logic orchestration
3. **Pose Analysis (`pose_analyzer.py`)**: MediaPipe-based pose estimation
4. **Reference Analysis (`reference_maker.py`)**: Statistical form comparison system
5. **Model Integration**: TensorFlow LSTM for classification and reference models for form assessment

### Data Flow

```
Input (Video/Image) → Pose Estimation(via mediapipe) → Skeleton Normalization → Feature Extraction → 
Phase Segmentation → Reference Comparison → Error Detection → Feedback Generation → Output
```

## API Documentation

### Available Endpoints

**Automatic Detection Endpoints:**
- `POST /analyze-form`: Analyze video file with automatic exercise detection
- `POST /analyze-frame`: Analyze image with automatic exercise detection

**User-Specified Endpoints:**
- `POST /analyze-form-user-specified`: Analyze video with user-specified exercise
- `POST /analyze-frame-user-specified`: Analyze image with user-specified exercise

**Informational Endpoints:**
- `GET /health`: System health check
- `GET /model-info`: Model loading status

### Request Format

For user-specified endpoints, include both the file and exercise type:
```
Content-Type: multipart/form-data
- file: [video/image file]
- exercise_type: [push-up|squat|bicep_curl|shoulder_press|plank]
```

### File Requirements

- **Video formats**: MP4, MOV, AVI, M4V
- **Image formats**: JPG, JPEG, PNG
- **Recommended**: Clear visibility of the exerciser from front/side view

## Reference-Based Analysis

### How References Work

The system implements a sophisticated reference-based analysis system that:

1. **Normalizes Skeletons**: Translates, scales, and rotates poses to account for camera position and body proportions
2. **Extracts Robust Features**: Calculates joint angles, segment ratios, and alignment metrics
3. **Segments by Phase**: Divides dynamic exercises into phases (e.g., squat descent/ascent)
4. **Compares to Expert Models**: Uses statistical models built from expert performance videos
5. **Detects Deviations**: Identifies significant deviations from reference standards

### Feature Extraction

The system analyzes multiple biomechanical features:

- **Joint Angles**: Knee, hip, elbow, and trunk angles
- **Segment Ratios**: Knee-over-toe ratios, elbow-to-shoulder alignment
- **Alignment Metrics**: Hip-shoulder-ankle line, frontal plane knee valgus angles
- **Phase-Specific Values**: Features analyzed relative to exercise phase

### Error Detection Logic

- Individual features are compared against reference means and standard deviations
- Z-scores calculated for each feature: `z = |user_value - reference_mean| / reference_std`
- Features with z-score > 2.0 are considered significant deviations
- Deviations are categorized by severity: low (z < 2.0), medium (z 2.0-3.0), high (z > 3.0)

## Response Format

### Understanding the Output

```json
{
  "status": "success",
  "form_feedback": {
    "timestamp": "2025-10-20T13:32:50.114345",
    "exercise_type": "squat",
    "detected_errors": [...],
    "overall_score": 92.21,
    "recommendations": [...]
  },
  "processing_time": 6.89
}
```

**Key Points:**

- **Overall Score**: Represents overall similarity to reference form (0-100 scale)
- **Detected Errors**: Specific biomechanical issues with confidence and severity
- **Confidence Scores**: Based on z-score deviation normalized to 0-1 range
- **Severity Levels**: 
  - High: Significant deviations (z > 3.0)
  - Medium: Noticeable deviations (z 2.0-3.0) 
  - Low: Minor deviations (z < 2.0)

**Important**: A high overall score with multiple medium-confidence errors indicates good overall form but with technical areas for improvement. This is beneficial as it provides specific feedback without discouraging users who are performing exercises correctly overall.

## How to Start the Server

### Prerequisites
- Python 3.10+
- Required dependencies (see pyproject.toml)

### Local Setup

1. **Clone the repository:**
```bash
git clone <repository_url>
cd <repository_name>
```

2. **Set up virtual environment:**
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

or with `uv`

```bash
uv venv #--seed if you want pip and setuptools
```


3. **Install dependencies:**
```bash
pip install -r pyproject.toml
```

or

```bash
uv sync
```

4. **Place required model files in root directory:**
- `final_forthesis_bidirectionallstm_and_encoders_exercise_classifier_model.h5`
- `thesis_bidirectionallstm_scaler.pkl`
- `thesis_bidirectionallstm_label_encoder.pkl`

5. **Build reference models (optional but recommended):**
```bash
python build_references.py
```

6. **Start the server:**
```bash
python -m uvicorn api:app #--reload for dev
```

or 

```bash
python main.py
```

The API will be available at `http://localhost:8000` with API documentation at `/docs`

### Docker Setup

The application includes containerization support for easy deployment and scaling.

**Docker Components:**
- **backend/Dockerfile**: Multi-stage build configuration for the FastAPI backend using uv for fast dependency installation
- **docker-compose.yml**: Multi-service orchestration (API, models, etc.) - builds from backend directory 
- **backend/.dockerignore**: Specifies files and directories to exclude from Docker builds

**Build and Run with Docker:**

1. **Build the Docker image (from backend directory):**
```bash
cd backend
docker build -t genyx-fitness-backend .
```

2. **Run the container:**
```bash
docker run -p 8000:8000 -v ../models:/app/models genyx-fitness-backend
```

**Using Docker Compose (from root directory):**
```bash
docker-compose up
```

**Docker Configuration:**
The `backend/.dockerignore` file ensures only necessary files are included in the Docker image:
- Includes only the backend folder and models directory
- Excludes ai_utils/, data/, video files (.mp4, .mov, etc.) and other unnecessary files
- Uses uv for fast dependency installation via `uv sync`
- Optimizes build context for faster builds

### Testing the API

Verify the API is running by accessing:
- Health check: `GET http://localhost:8000/health`
- Model info: `GET http://localhost:8000/model-info`
- API Documentation: `GET http://localhost:8000/docs`

## Original Repository Credit

This project builds upon the foundational work from "Fitness AI Trainer With Automatic Exercise Recognition and Counting" by Riccardo Riccio. The original implementation provided the core exercise recognition and repetition counting system that was enhanced with advanced form analysis capabilities.

**Original Project Links:**
- GitHub: https://github.com/RiccardoRiccio/Fitness-AI-Trainer-With-Automatic-Exercise-Recognition-and-Counting#
- Paper: "Real-Time Fitness Exercise Classification and Counting from Video Frames" (https://arxiv.org/abs/2411.11548)
- Dataset: https://www.kaggle.com/datasets/riccardoriccio/real-time-exercise-recognition-dataset
- Author: Riccardo Riccio

The original system provided:
- LSTM-based exercise classification
- Pose estimation for repetition counting
- Streamlit-based user interface
- Basic form analysis capabilities

This enhanced version adds sophisticated reference-based form analysis while maintaining all original functionality.

---