import os

TEST_ENV = {
    "ROBOFLOW_API_KEY": "test-key",
    "WORKSPACE_NAME": "test-workspace",
    "WORKFLOW_ID": "detect",
    "WORKFLOW_ID_WITH_COURT_POINTS": "detect-with-court",
    "MAX_WORKERS": "2",
    "MAX_ATTEMPTS": "2",
    "XGB_BOUNCE_THRESHOLD": "0.7",
    "XGB_HIT_THRESHOLD": "0.5",
    "TENNIS_COURT_LENGTH": "23.77",
    "TENNIS_COURT_WIDTH": "10.97",
    "TENNIS_COURT_SCALE": "20",
    "TENNIS_COURT_PADDING": "50",
    "MAX_AGE": "3600",
    "CHECK_EVERY": "3600",
}
os.environ.update(TEST_ENV)

from pathlib import Path

import cv2
import numpy as np
import pytest

SCALE = float(TEST_ENV["TENNIS_COURT_SCALE"])
PADDING = float(TEST_ENV["TENNIS_COURT_PADDING"])

# The 14 court keypoints in court meters, in the same order as
# compute_homography's top_down_points.
COURT_METERS = np.array([
    [0, 23.77], [1.37, 23.77], [9.60, 23.77], [10.97, 23.77],
    [10.97, 0], [9.60, 0], [1.37, 0], [0, 0],
    [1.37, 18.285], [1.37, 5.485], [9.60, 18.285], [9.60, 5.485],
    [5.485, 18.285], [5.485, 5.485],
], dtype=np.float32)

# Where those keypoints land on the top-down template.
TOP_DOWN = PADDING + COURT_METERS * SCALE

# A known camera: a mild perspective warp from top-down template to video pixels.
# Tests check that the code recovers this mapping.
TOP_DOWN_TO_VIDEO = np.array([
    [0.55, 0.08, 150.0],
    [0.00, 0.45, 40.0],
    [0.00, 0.0002, 1.0],
], dtype=np.float64)

def to_video(points):
    pts = np.asarray(points, dtype=np.float32).reshape(-1, 1, 2)
    return cv2.perspectiveTransform(pts, TOP_DOWN_TO_VIDEO).reshape(-1, 2)

def to_top_down(points):
    pts = np.asarray(points, dtype=np.float32).reshape(-1, 1, 2)
    return cv2.perspectiveTransform(pts, np.linalg.inv(TOP_DOWN_TO_VIDEO)).reshape(-1, 2)

VIDEO_KEYPOINTS = [{"x": float(x), "y": float(y)} for x, y in to_video(TOP_DOWN)]


@pytest.fixture
def court_points_file(tmp_path: Path) -> Path:
    """A court_points.json with keypoints on frame 1, like predict() writes."""
    import json

    path = tmp_path / "court_points.json"
    path.write_text(json.dumps({"1": VIDEO_KEYPOINTS}))
    return path