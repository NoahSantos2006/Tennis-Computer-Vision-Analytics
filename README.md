# 🎾 CourtVision — Tennis Computer Vision Analytics

A full-stack computer vision application for analyzing tennis match footage. A **FastAPI** backend runs a YOLO/Roboflow-based detection pipeline (player detection, court detection, ball tracking, bounce detection, homography) on an uploaded video as a background job, and a **React** frontend lets a user upload a clip, watch job progress, and view the resulting shot chart.

**Live demo:** [tennis-computer-vision-analytics.vercel.app](https://tennis-computer-vision-analytics.vercel.app/)

## Features

- 🎥 **Video upload & job queue** — upload a match clip via the web app; processing runs asynchronously as a background job
- 👤 **Player detection** via Ultralytics YOLO
- 🟡 **Ball detection & tracking**, including a dedicated ball-tracking class serialized per job
- 🏟️ **Court detection** — court reference points extracted per video
- 📐 **Homography transformation** — maps camera-perspective coordinates onto a normalized court
- 💥 **Bounce detection** using a trained XGBoost model (`backend/models/model.ubj`)
- 📊 **Shot chart visualization** in the frontend, built from the transformed ball-position data
- 🔄 **Job status polling** — the frontend polls the backend for live processing status
- ☁️ **Roboflow Inference** integration for hosted model inference

## Architecture

```text
┌──────────────────┐        upload video        ┌───────────────────────┐
│  React Frontend  │ ───────────────────────────▶│    FastAPI Backend    │
│  (Vite + React)  │                              │                       │
│                   │◀───── job_id, status ────────│  /analyze             │
│  VideoUpload      │                              │  /jobs/{id}/status    │
│  ProcessingVideo  │◀──── results.json ───────────│  /jobs/{id}/results   │
│  TennisCourt      │                              │  /video/{id}/{file}   │
│  ShotChart        │                              │                       │
└──────────────────┘                              └──────────┬────────────┘
                                                               │ background task
                                                               ▼
                                                   ┌───────────────────────┐
                                                   │   Detection Pipeline   │
                                                   │  (backend/predict.py) │
                                                   │                       │
                                                   │ YOLO / Roboflow       │
                                                   │ Court + Ball tracking │
                                                   │ Homography            │
                                                   │ XGBoost bounce model  │
                                                   └──────────┬────────────┘
                                                               ▼
                                              storage/jobs/{job_id}/output/
                                              ├── BallTracking/
                                              ├── court_points/
                                              ├── predictions/
                                              ├── status.json
                                              └── results.json
```

### Processing pipeline

1. **Upload & job creation** — the frontend posts a video to `/analyze`; the backend creates a UUID-named job folder under `storage/jobs/<job_id>/` with `input/` and `output/` subdirectories, then kicks off processing as a FastAPI background task.
2. **Video validation** — the input clip is checked before full processing begins.
3. **Object detection** — each frame is run through YOLO / Roboflow Inference to detect players, the ball, and court features.
4. **Court detection** — court reference points are extracted and saved to `output/court_points/`.
5. **Homography** — detected court points are used to compute a homography matrix mapping camera coordinates to normalized court coordinates.
6. **Ball tracking & bounce detection** — `backend/scripts/ball_tracker.py` reconstructs ball trajectory across frames; a trained XGBoost model (`backend/models/model.ubj`) classifies bounces.
7. **Output generation** — frame predictions, court points, ball-tracking data, and a final `results.json` (including a bounce-detection dictionary) are written to the job's `output/` folder.
8. **Frontend polling & visualization** — the frontend polls `/jobs/{job_id}/status` until processing completes, then fetches `/jobs/{job_id}/results` and renders the shot chart over a `TennisCourt` layout.

## Tech Stack

| Layer | Technology |
| --- | --- |
| Backend framework | FastAPI, Uvicorn |
| Computer vision | Ultralytics YOLO, OpenCV, Roboflow `inference_sdk` |
| Bounce classification | XGBoost |
| Data handling | NumPy, pandas, PyArrow, scikit-learn |
| Video downloading (tooling) | yt-dlp, ffmpeg |
| Config | python-dotenv |
| Frontend | React 19, Vite, react-router-dom |

## Project Structure

```text
Tennis Computer Vision Analytics/
├── backend/
│   ├── main.py                  # FastAPI app & routes
│   ├── predict.py                # Core video analysis pipeline
│   ├── models/
│   │   └── model.ubj              # Trained XGBoost bounce-detection model
│   └── scripts/
│       ├── ball_tracker.py        # Ball tracking across frames
│       ├── homography.py          # Court mapping / perspective transform
│       ├── side_functions.py      # Predictions, court points, angle calcs
│       ├── prepping_model.py      # Model prep helpers
│       ├── status.py              # Thread-safe job status read/write
│       └── print_memory.py        # Memory usage debugging helper
├── frontend/
│   ├── src/
│   │   ├── main.jsx
│   │   ├── header.jsx / header.css
│   │   ├── VideoUpload.jsx / .css   # Upload UI
│   │   ├── ProcessingVideo.jsx / .css # Job status / progress UI
│   │   ├── TennisCourt.jsx          # Court rendering
│   │   ├── ShotChart.jsx / .css     # Shot/bounce visualization
│   │   └── images/
│   ├── index.html
│   └── package.json
├── storage/
│   └── jobs/<job_id>/
│       ├── input/                 # Uploaded + validated video
│       └── output/                # Pipeline output (see below)
├── requirements.txt
└── .env                          # Not committed — see Environment Variables
```

## Getting Started

### Prerequisites

- Python 3.10+
- Node.js 18+
- A [Roboflow](https://roboflow.com) API key

### 1. Clone the repository

```bash
git clone git@github.com:NoahSantos2006/Tennis-Pickleball-Computer-Vision-Analytics.git
cd "Tennis Computer Vision Analytics"
```

### 2. Backend setup

```bash
python -m venv venv
source venv/bin/activate      # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

Create a `.env` file in the project root:

```env
ROBOFLOW_API_KEY=your_roboflow_api_key
MAX_WORKERS=4
VISION_MODEL_ID=your_model_id
```

> **Never commit `.env` or API keys.** It's already listed in `.gitignore`.

Run the API server from the project root (so the `backend` package resolves correctly):

```bash
uvicorn backend.main:app --reload
```

The API will be available at `http://localhost:8000`.

### 3. Frontend setup

```bash
cd frontend
npm install
npm run dev
```

The app will be available at `http://localhost:5173` (the default Vite port, already allow-listed in the backend's CORS config).

## API Reference

| Method | Endpoint | Description |
| --- | --- | --- |
| `GET` | `/` | Health check |
| `POST` | `/analyze` | Upload a video (`multipart/form-data`, field `video`); creates a job and starts processing in the background. Returns `job_id`. |
| `GET` | `/video/{job_id}/{filename}` | Streams the original uploaded video for a job |
| `GET` | `/jobs/{job_id}/status` | Returns the current status of a processing job |
| `GET` | `/jobs/{job_id}/results` | Returns the final analytics (`results.json`), including the bounce-detection dictionary |

## Output Data (per job)

```text
storage/jobs/<job_id>/output/
├── BallTracking/<video_name>/
│   ├── <video_name>_ball_tracking.json      # Ball position/trajectory data
│   └── <video_name>_ball_tracking_class.pkl # Serialized BallTracker object
├── court_points/
│   └── <video_name>_court_points.json       # Detected court reference points
├── predictions/
│   └── <video_name>_predictions.txt         # Frame-by-frame model predictions
└── status.json                              # Live job status
storage/jobs/<job_id>/results.json           # Final combined results
```

## Notes

- On backend startup, `storage/jobs/` is wiped and recreated, so job data does not persist across server restarts.
- Deployment note: the CORS config in `backend/main.py` allow-lists `http://localhost:5173` and a Vercel-hosted frontend URL — update this list for other deployment targets.

## License

Copyright © 2026 Noah Santos. All rights reserved.

This project and its source code are proprietary. No part of this repository may be copied, modified, distributed, or used, in whole or in part, without express written permission from the copyright holder.
