"""Ball tracker features (angle, speed, window columns) and event picking."""
import json
import math
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import xgboost as xgb

from backend.scripts.prepping_model import get_dataframe, get_windows, pick_events
from backend.scripts.side_functions import find_angles

MODEL_PATH = Path(__file__).resolve().parent.parent / "backend" / "models" / "model.ubj"


def frame(x, y, homography=(100, 200), estimation=False, repeat=False):
    if repeat:
        return {"repeat_frame": True}
    return {
        "vision model location": (x, y),
        "homography location": homography,
        "estimation": estimation,
        "repeat_frame": False,
    }


def tracker_from(locations):
    return SimpleNamespace(tracker={i: frame(x, y) for i, (x, y) in enumerate(locations, start=1)})


# ---------- find_angles ----------

def test_straight_line_has_zero_angle_and_constant_speed():
    t = tracker_from([(10 * i, 5 * i) for i in range(8)]).tracker
    find_angles(SimpleNamespace(tracker=t))

    for f in range(2, 7):
        assert t[f]["angle"] == pytest.approx(0, abs=1e-3)
        assert t[f]["velocity"] == pytest.approx((10, 5))
        assert t[f]["speed"] == pytest.approx(math.hypot(10, 5))


def test_bounce_shows_up_as_a_sharp_angle():
    # Ball falls (y grows in image space) then rises: a V at frame 4.
    path = [(0, 0), (10, 10), (20, 20), (30, 30), (40, 20), (50, 10), (60, 0)]
    t = tracker_from(path).tracker
    find_angles(SimpleNamespace(tracker=t))

    assert t[4]["angle"] == pytest.approx(90, abs=1e-3)
    assert t[3]["angle"] == pytest.approx(0, abs=1e-3)


def test_angle_skips_repeat_frames():
    # Frame 3 is a duplicate video frame; frame 4 should use 2 and 5 as neighbours.
    t = {
        1: frame(0, 0), 2: frame(10, 0), 3: frame(0, 0, repeat=True),
        4: frame(20, 0), 5: frame(30, 10), 6: frame(40, 20),
    }
    find_angles(SimpleNamespace(tracker=t))

    assert "angle" not in t[3]
    assert t[4]["angle"] == pytest.approx(45, abs=1e-3)


def test_no_angle_when_ball_is_missing():
    t = tracker_from([(0, 0), (10, 10), (-1, -1), (30, 30), (40, 40), (50, 50)]).tracker
    find_angles(SimpleNamespace(tracker=t))

    assert "angle" not in t[2] and "angle" not in t[3] and "angle" not in t[4]
    assert "angle" in t[5]


# ---------- window features ----------

def json_round_trip(d):
    """The pipeline saves the tracker to JSON and reads it back, which turns tuples into lists."""
    return {int(k): v for k, v in json.loads(json.dumps(d)).items()}


def tracked(n=20):
    t = tracker_from([(10 * i, 4 * i) for i in range(n)]).tracker
    for f in t:
        t[f]["homography location"] = (100 + f, 200 + 2 * f)
    find_angles(SimpleNamespace(tracker=t))
    return json_round_trip(t)


def player(hx, hy, box=True):
    p = {"class": "player", "homography location": [hx, hy]}
    if box:
        p["box"] = [0, 0, 10, 10]
    return p


def test_window_has_every_feature_for_every_offset():
    t = tracked()
    real_frames = sorted(t)
    row = get_windows(frame_id=10, ball_tracker=t, predictions_dict={}, bounce_window=5, real_frames=real_frames)

    assert len(row) == 11 * 8
    assert row["HOMOGRAPHY X(frame - 0)"] == 110
    assert row["HOMOGRAPHY Y(frame + 2)"] == 224
    assert row["ESTIMATION(frame - 3)"] == 0


def test_window_pads_with_nan_past_the_start_of_the_video():
    t = tracked()
    row = get_windows(frame_id=3, ball_tracker=t, predictions_dict={}, bounce_window=5, real_frames=sorted(t))

    # frames 1 and earlier don't exist (frame 1 never has an angle)
    assert np.isnan(row["SPEED(frame - 2)"])
    assert np.isnan(row["HOMOGRAPHY X(frame - 5)"])
    assert row["HOMOGRAPHY X(frame - 1)"] == 102


def test_distance_to_nearest_player():
    t = tracked()
    # ball at frame 10 is at (110, 220)
    preds = {10: [player(110, 260), player(113, 224), {"class": "ball"}]}
    row = get_windows(frame_id=10, ball_tracker=t, predictions_dict=preds, bounce_window=5, real_frames=sorted(t))

    assert row["DISTANCE FROM NEAREST PLAYER(frame - 0)"] == pytest.approx(5.0)


def test_players_without_a_box_are_ignored():
    """Regression: the demo once dropped the line that set pred['box'].

    Without it every player is skipped here, the distance feature is all NaN,
    and the model stops finding hits. See also test_api.py, which checks the
    real pipeline sets 'box'.
    """
    t = tracked()
    preds = {10: [player(113, 224, box=False)]}
    row = get_windows(frame_id=10, ball_tracker=t, predictions_dict=preds, bounce_window=5, real_frames=sorted(t))

    assert np.isnan(row["DISTANCE FROM NEAREST PLAYER(frame - 0)"])


def test_dataframe_columns_match_the_trained_model(tmp_path):
    """If the feature code and model.ubj drift apart, predict_proba breaks or silently misreads columns."""
    t = tracked()
    preds_file = tmp_path / "preds.json"
    preds_file.write_text(json.dumps({str(f): [] for f in t}))

    df = get_dataframe(ball_tracker={str(k): v for k, v in t.items()}, PREDICTIONS_FILE=preds_file)

    model = xgb.XGBClassifier()
    model.load_model(MODEL_PATH)
    assert list(df.drop(columns=["FRAME"]).columns) == model.get_booster().feature_names
    assert model.predict_proba(df.drop(columns=["FRAME"])).shape == (len(df), 3)


# ---------- pick_events ----------

def test_pick_events_keeps_the_peak_of_each_cluster():
    frames = np.array([10, 11, 12, 13, 30, 31, 50])
    p = np.array([0.6, 0.9, 0.8, 0.2, 0.75, 0.95, 0.4])

    assert list(pick_events(frames, p, threshold=0.5, gap=3)) == [11, 31]


def test_pick_events_splits_clusters_further_apart_than_gap():
    frames = np.array([10, 14, 18])
    p = np.array([0.9, 0.8, 0.7])

    assert list(pick_events(frames, p, threshold=0.5, gap=3)) == [10, 14, 18]
    assert list(pick_events(frames, p, threshold=0.5, gap=4)) == [10]


def test_pick_events_with_nothing_above_threshold():
    assert len(pick_events(np.array([1, 2, 3]), np.array([0.1, 0.2, 0.3]), threshold=0.5)) == 0
