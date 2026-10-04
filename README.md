# 🎾 Tennis Computer Vision Analytics

A full-stack computer vision application for analyzing tennis match footage. A **FastAPI** backend runs a Roboflow-based detection pipeline (player detection, court detection, ball tracking, homography, bounce and hit detection) on an uploaded video as a background job, and a **React** frontend lets a user upload a clip, watch job progress, and view the resulting shot chart.

**Live demo:** [tennis-computer-vision-analytics.vercel.app](https://tennis-computer-vision-analytics.vercel.app/)

## Features

- 🎥 **Video upload & job queue** — upload a match clip via the web app; processing runs asynchronously as a background job
- 👤 **Player detection** via Roboflow RF-DETR
- 🟡 **Ball detection & tracking**, including a dedicated ball-tracking class serialized per job
- 🏟️ **Court detection** — court reference points extracted per video
- 📐 **Homography transformation** — maps camera-perspective coordinates onto a normalized court
- ⏭️ **Repeat-frame skipping** — duplicated frames (from frame-rate conversion) are detected and not sent for inference, while frame numbering stays aligned with the original video
- 💥 **Bounce & hit detection** using a trained XGBoost model (`backend/models/model.ubj`) with configurable probability thresholds
- 📊 **Shot chart visualization** in the frontend, built from the transformed ball-position data
- 🔄 **Job status polling** — the frontend polls the backend for live processing status
- ☁️ **Roboflow Inference** integration for hosted model inference

## Architecture

```text
┌──────────────────┐        upload video        ┌───────────────────────┐
│  React Frontend  │ ─────────────────────────▶  │   FastAPI Backend    │
│  (Vite + React)  │                             │                       │
│                  │◀───── job_id, status ───── │  /analyze             │
│  VideoUpload     │                             │  /jobs/{id}/status    │
│  ProcessingVideo │◀──── results.json ───────── │  /jobs/{id}/results   │
│  TennisCourt     │                              │  /video/{id}/{file}   │
│  ShotChart       │                              │                       │
└──────────────────┘                              └──────────┬────────────┘
                                                               │ background task
                                                               ▼
                                                   ┌───────────────────────┐
                                                   │   Detection Pipeline  │
                                                   │  (backend/predict.py) │
                                                   │                       │
                                                   │ Roboflow  RF-DETR     │
                                                   │ Court + Ball tracking │
                                                   │ Homography            │
                                                   │ XGBoost bounce/hit    │
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
2. **Repeat-frame detection** — each frame is downscaled to 480×270 grayscale and compared with the previous real frame; if fewer than 20 pixels differ by more than 20, it is marked `"Repeat Frame"` and skipped by inference, tracking, and classification.
3. **Object detection** — frames are sent in parallel (`MAX_WORKERS` threads, `MAX_ATTEMPTS` retries with backoff) to Roboflow workflows. Frame 1 and every 50th frame use the workflow that also returns court keypoints (`WORKFLOW_ID_WITH_COURT_POINTS`); the rest use `WORKFLOW_ID`. Progress is written to `output/status.json` as frames complete.
4. **Court detection** — court keypoints are saved to `output/court_points/`.
5. **Homography** — court keypoints are fit to a top-down court template (RANSAC) so ball and player positions can be mapped onto the court.
6. **Ball tracking** — `backend/scripts/ball_tracker.py` (`BallTracker`) picks the real ball each frame, estimates positions through missed detections, and corrects false positives. Velocity, speed and direction-change angle are computed per frame.
7. **Bounce & hit classification** — each real frame gets a feature window of ±5 real frames (speed, velocity, angle, court position, distance to nearest player, estimation flag). The XGBoost model (`backend/models/model.ubj`) outputs bounce/hit probabilities; `XGB_BOUNCE_THRESHOLD` and `XGB_HIT_THRESHOLD` decide which peaks become events.
8. **Output generation** — frame predictions, court points, ball-tracking data, and `results.json` (a per-frame label — 0 none, 1 bounce, 2 hit — with the ball's court location) are written to the job folder.
9. **Frontend polling & visualization** — the frontend polls `/jobs/{job_id}/status` until processing completes, then fetches `/jobs/{job_id}/results` and renders the shot chart over a `TennisCourt` layout.

## Tech Stack

| Layer | Technology |
| --- | --- |
| Backend framework | FastAPI, Uvicorn |
| Computer vision | OpenCV, Roboflow `inference_sdk` |
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
│       ├── side_functions.py      # Tracking driver, homography, angles, bounce/hit picking
│       ├── prepping_model.py      # Windowed feature rows for the XGBoost model
│       ├── status.py              # Thread-safe job status read/write
│       └── sweep.py               # Background cleanup of finished jobs
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
git clone https://github.com/NoahSantos2006/Tennis-Computer-Vision-Analytics.git
cd Tennis-Computer-Vision-Analytics
```

### 2. Backend setup

```bash
python -m venv venv
source venv/bin/activate      # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

Create a `.env` file in the project root. Every variable below is read at startup, so the backend will fail to import if one is missing.

| Variable | Purpose |
| --- | --- |
| `ROBOFLOW_API_KEY` | Roboflow API key |
| `WORKSPACE_NAME` | Roboflow workspace that hosts the workflows |
| `WORKFLOW_ID` | Workflow for object detection (players + ball) |
| `WORKFLOW_ID_WITH_COURT_POINTS` | Workflow that also returns court keypoints (frame 1 and every 50th frame) |
| `MAX_WORKERS` | Parallel inference requests |
| `MAX_ATTEMPTS` | Retries per frame before it is recorded as failed |
| `XGB_BOUNCE_THRESHOLD` | Minimum bounce probability for an event |
| `XGB_HIT_THRESHOLD` | Minimum hit probability for an event |
| `TENNIS_COURT_LENGTH` | Court length in meters |
| `TENNIS_COURT_WIDTH` | Court width in meters |
| `TENNIS_COURT_SCALE` | Pixels per meter on the top-down court template |
| `TENNIS_COURT_PADDING` | Padding in pixels around the top-down court |
| `MAX_AGE` | Seconds a finished job is kept before the sweeper deletes it |
| `CHECK_EVERY` | Seconds between sweeper runs |

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

Set `VITE_API_URL` (for example in `frontend/.env`) to the backend's URL, such as `http://localhost:8000`.

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
storage/jobs/<job_id>/results.json           # Per-frame labels (0 none, 1 bounce, 2 hit) + court location, fps, video name, job id
```

## Notes

- A background sweeper (`backend/scripts/sweep.py`) runs every `CHECK_EVERY` seconds and deletes job folders whose `results.json` is older than `MAX_AGE` seconds.
- Deployment note: the CORS config in `backend/main.py` allow-lists `http://localhost:5173` and a Vercel-hosted frontend URL — update this list for other deployment targets.

## License

Copyright © 2026 Noah Santos. All rights reserved.

This project and its source code are proprietary. No part of this repository may be copied, modified, distributed, or used, in whole or in part, without express written permission from the copyright holder.
