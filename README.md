# 🎾 Tennis & Pickleball Computer Vision Analytics

A computer vision system for analyzing **tennis and pickleball match footage** using object detection, player tracking, court detection, homography transformations, and ball tracking.

The project processes match videos frame-by-frame to detect players, locate the ball, identify court geometry, track movement, and generate structured analytics that can be used for sports analysis and visualization.

## 🚀 Features

* 🎾 **Tennis and Pickleball Support**
* 👤 **Player Detection & Tracking**
* 🟡 **Ball Detection & Tracking**
* 🏟️ **Court Detection**
* 📐 **Homography / Perspective Transformation**
* 💥 **Ball Bounce Analysis**
* 📍 **Player and Ball Position Tracking**
* 🎥 **Annotated Video Generation**
* 📊 **Frame-by-Frame Prediction Data**
* 💾 **JSON Analytics Output**
* ⚡ **Parallel Frame Processing**
* 🧠 **Roboflow Inference Integration**
* 🔎 **YOLO-Based Computer Vision**

---

## 🧠 How It Works

The system takes a tennis or pickleball video and processes it through several computer vision stages.

```text
                    Match Video
                         │
                         ▼
                ┌─────────────────┐
                │ Video Validation│
                └────────┬────────┘
                         │
                         ▼
                ┌─────────────────┐
                │ Player Detection│
                │  + Ball Detection│
                └────────┬────────┘
                         │
                         ▼
                ┌─────────────────┐
                │ Court Detection │
                └────────┬────────┘
                         │
                         ▼
                ┌─────────────────┐
                │   Homography    │
                │ Transformation  │
                └────────┬────────┘
                         │
                         ▼
                ┌─────────────────┐
                │  Ball Tracking  │
                │ + Bounce Logic  │
                └────────┬────────┘
                         │
                         ▼
             ┌───────────────────────┐
             │ Analytics + Annotated │
             │         Video         │
             └───────────────────────┘
```

### 1. Video Processing

OpenCV loads the input match footage and extracts information such as:

* frame rate
* resolution
* frame count
* individual video frames

The project also includes video validation logic to detect excessive repeated frames before analysis.

### 2. Object Detection

Computer vision models detect important objects within each frame, including:

* players
* tennis/pickleball ball
* court features

The project uses **Ultralytics YOLO** alongside **Roboflow Inference** for model inference.

### 3. Player Tracking

Detected players can be tracked across frames to maintain consistent player identities throughout the match.

This makes it possible to analyze player movement instead of treating each frame independently.

### 4. Court Detection

Court reference points are detected and stored for each video.

These points establish the geometry of the playing surface and allow image coordinates to be converted into meaningful court coordinates.

### 5. Homography Transformation

A homography matrix maps points from the camera perspective to a normalized court perspective.

```text
Camera Coordinates
        │
        ▼
   Homography
        │
        ▼
Court Coordinates
```

This allows player and ball positions from the original video to be analyzed relative to the actual court.

### 6. Ball Tracking

The ball tracker processes detections over time to reconstruct the movement of the ball.

Tracking data can then be used for analytics such as:

* ball trajectory
* ball position
* bounce detection
* shot visualization

### 7. Output Generation

The pipeline generates multiple output artifacts, including:

* annotated match videos
* frame-by-frame predictions
* detected court points
* ball tracking data
* serialized tracker data
* debugging information

---

## 🛠️ Tech Stack

| Technology             | Purpose                         |
| ---------------------- | ------------------------------- |
| **Python**             | Core application                |
| **OpenCV**             | Video and image processing      |
| **Ultralytics YOLO**   | Object detection                |
| **Roboflow Inference** | Hosted model inference          |
| **NumPy**              | Numerical operations            |
| **Supervision**        | Computer vision utilities       |
| **Homography**         | Perspective transformation      |
| **TQDM**               | Processing progress             |
| **python-dotenv**      | Environment variable management |
| **Pickle / JSON**      | Analytics and tracker storage   |

---

## 📁 Project Structure

```text
Tennis-Pickleball-Computer-Vision-Analytics/
│
├── main.py
├── requirements.txt
├── README.md
│
├── scripts/
│   ├── ball_tracker.py
│   ├── homography.py
│   └── side_functions.py
│
├── input/
│   └── match_video.mp4
│
└── output/
    ├── annotated_videos/
    ├── court_points/
    ├── predictions/
    └── BallTracking/
```

### Important Components

**`main.py`**

Main entry point for the computer vision pipeline. Handles video processing, inference, validation, predictions, and output generation.

**`scripts/ball_tracker.py`**

Handles ball tracking and stores ball-related analytics across video frames.

**`scripts/homography.py`**

Contains homography-related visualization and court mapping functionality.

**`scripts/side_functions.py`**

Contains supporting functionality including prediction processing, court point detection, angle calculations, and homography computation.

---

## ⚙️ Installation

### 1. Clone the repository

```bash
git clone https://github.com/NoahSantos2006/Tennis-Pickleball-Computer-Vision-Analytics.git

cd Tennis-Pickleball-Computer-Vision-Analytics
```

### 2. Create a virtual environment

#### macOS / Linux

```bash
python3 -m venv venv
source venv/bin/activate
```

#### Windows

```bash
python -m venv venv
venv\Scripts\activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

---

## 🔑 Environment Variables

The project uses Roboflow inference and expects an API key through an environment variable.

Create a `.env` file in the root directory:

```env
API_KEY=your_roboflow_api_key
```

The application loads the key using:

```python
API_KEY = os.getenv("API_KEY")
```

> **Important:** Never commit your `.env` file or API keys to GitHub.

Add the following to `.gitignore`:

```gitignore
.env
venv/
__pycache__/
output/
```

---

## 🎥 Adding a Video

Place the match video inside:

```text
input/
```

For example:

```text
input/
└── match1.mp4
```

Then update the configuration near the bottom of `main.py`:

```python
video_filename = "match1"
sport = "tennis"
```

For pickleball:

```python
video_filename = "match1"
sport = "pickleball"
```

The filename should be provided **without `.mp4`**.

---

## ▶️ Running the Project

Run:

```bash
python main.py
```

The program will locate the configured video inside the `input/` directory and begin processing it.

A progress bar displays processing progress while the video is analyzed.

---

## 📤 Output

Generated data is stored inside:

```text
output/
```

### Annotated Videos

```text
output/annotated_videos/
```

Contains processed videos with computer vision annotations.

### Predictions

```text
output/predictions/
```

Stores frame-by-frame model predictions.

Example:

```text
match1_predictions.txt
```

Prediction information is stored as structured JSON data.

### Court Points

```text
output/court_points/
```

Stores detected court coordinates used for perspective transformation.

Example:

```text
match1_court_points_from_model.json
```

### Ball Tracking

```text
output/BallTracking/<video_name>/
```

Stores ball tracking information and related analytics.

Example:

```text
output/
└── BallTracking/
    └── match1/
        ├── match1_ball_tracking.json
        ├── match1_ball_tracking_class.pkl
        ├── match1_actual_bounces.txt
        └── match1_bouncing_debugging.txt
```

---

## 📐 Court Coordinate Mapping

A major component of the project is transforming camera coordinates into court coordinates.

Given detected court points, the system calculates a homography matrix:

```text
Video Frame
    │
    │ detected court coordinates
    ▼
Homography Matrix
    │
    ▼
Normalized Court
```

This transformation makes it possible to compare player and ball movement independent of the camera's perspective.

---

## 📊 Potential Analytics

The computer vision pipeline provides a foundation for generating advanced racket-sport analytics such as:

* player movement distance
* player positioning
* court coverage
* ball trajectories
* shot placement
* bounce locations
* rally analysis
* shot charts
* player heatmaps
* movement patterns
* offensive vs. defensive positioning

These analytics can be useful for players, coaches, researchers, and sports analytics applications.
