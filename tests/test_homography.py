"""Mapping video pixels to top-down court coordinates."""
import json

import numpy as np
import pytest

from backend.scripts.ball_tracker import BallTracker, compute_homography
from backend.scripts.side_functions import detection_of_court_points
from tests.conftest import TOP_DOWN, VIDEO_KEYPOINTS, to_top_down, to_video

# Takes court points and checks if they land back into the top_down points I set
def test_homography_maps_keypoints_back_onto_the_court(court_points_file):
    H = compute_homography(COURT_POINTS_PATH=court_points_file, frame_id=1)

    video = np.array([[p["x"], p["y"]] for p in VIDEO_KEYPOINTS], dtype=np.float32)
    mapped = (H @ np.c_[video, np.ones(len(video))].T).T
    mapped = mapped[:, :2] / mapped[:, 2:]

    np.testing.assert_allclose(mapped, TOP_DOWN, atol=1.0)


def test_homography_is_empty_for_a_frame_without_keypoints(court_points_file):
    assert len(compute_homography(COURT_POINTS_PATH=court_points_file, frame_id=2)) == 0


def test_homography_tolerates_one_bad_keypoint(tmp_path):
    # RANSAC should ignore a single badly detected keypoint.
    points = [dict(p) for p in VIDEO_KEYPOINTS]
    points[4]["x"] += 80
    path = tmp_path / "court_points.json"
    path.write_text(json.dumps({"1": points}))

    H = compute_homography(COURT_POINTS_PATH=path, frame_id=1)
    tracker_like = BallTracker.__new__(BallTracker)
    tracker_like.HOMOGRAPHY_MATRIX = H

    centre_service_t = tuple(to_video([TOP_DOWN[12]])[0])
    x, y = tracker_like.compute_homographical_location(centre_service_t)
    assert (x, y) == pytest.approx(tuple(TOP_DOWN[12]), abs=2)


def test_ball_location_maps_to_court(court_points_file):
    tracker = BallTracker(COURT_POINTS_FILE=court_points_file)
    court_point = (200.0, 300.0)

    x, y = tracker.compute_homographical_location(tuple(to_video([court_point])[0]))

    assert (x, y) == pytest.approx(court_point, abs=2)


def test_player_location_uses_bottom_centre_of_box(court_points_file):
    """A player stands on the court at their feet, not at the box centre."""
    H = compute_homography(COURT_POINTS_PATH=court_points_file, frame_id=1)
    feet = tuple(to_video([(160.0, 500.0)])[0])
    box = [feet[0] - 20, feet[1] - 80, feet[0] + 20, feet[1]]

    x, y = detection_of_court_points(box=np.array(box, dtype=np.float32), H=H)

    assert (x, y) == pytest.approx(tuple(to_top_down([feet])[0]), abs=2)
