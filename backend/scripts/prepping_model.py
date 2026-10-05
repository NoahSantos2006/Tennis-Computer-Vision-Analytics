import xgboost as XGB
from xgboost import XGBClassifier
import pandas as pd
import numpy as np
import os
from pathlib import Path
import json

from backend.scripts.ball_tracker import BallTracker

def pick_events(
        frames: pd.DataFrame, 
        p: np.array, 
        threshold: float = 0.5, 
        gap: int = 3
    ):
    # frames above thr that are within `gap` of each other form one cluster; keep its peak
    idx = np.where(p >= threshold)[0]
    events, cluster = [], []
    for k in idx:
        if cluster and frames[k] - frames[cluster[-1]] > gap:
            events.append(max(cluster, key=lambda c: p[c])); cluster = []
        cluster.append(k)
    if cluster:
        events.append(max(cluster, key=lambda c: p[c]))
    return frames[events]

def acquire_training_dataframes(
        VIDEO_FILENAME: str,
        OUTPUT_PATH: Path,
    ) -> pd.DataFrame:

        BALL_TRACKER_JSON_FILE = os.path.join(OUTPUT_PATH, "BallTracking", f"{VIDEO_FILENAME}", f"{VIDEO_FILENAME}_ball_tracking.json")
        PREDICTIONS_FILE = os.path.join(OUTPUT_PATH, f"predictions", f"{VIDEO_FILENAME}_predictions.txt")

        with open(BALL_TRACKER_JSON_FILE, "r") as f:

            ball_tracker = json.load(f)

        df = get_dataframe(
            ball_tracker=ball_tracker,
            VIDEO_FILENAME=VIDEO_FILENAME,
            PREDICTIONS_FILE=PREDICTIONS_FILE
        )

        return df

def get_dataframe(
    ball_tracker: dict,
    VIDEO_FILENAME: str,
    PREDICTIONS_FILE: Path,
    bounce_window: int = 5,
):

    temp_ball_tracker = {}
    real_frames = []
    for k, v in ball_tracker.items():

        temp_ball_tracker[int(k)] = v

        if not v['repeat_frame']:
            real_frames.append(int(k))

    ball_tracker = temp_ball_tracker

    with open(PREDICTIONS_FILE, "r") as file:

        predictions_dict = json.load(file)
        predictions_dict = {int(k): v for k, v in predictions_dict.items()}

    def get_windows(
        frame_id: int,
        ball_tracker: dict,
        predictions_dict: dict,
        bounce_window: int,
        real_frames: list,
        FEATURES: list = [
            "SPEED", "VX", "VY", "ANGLE", "HOMOGRAPHY X", "HOMOGRAPHY Y",
            "DISTANCE FROM NEAREST PLAYER", "ESTIMATION"
        ]
    ) -> dict:

        res = {}

        i = real_frames.index(frame_id)

        for offset in range(-bounce_window, bounce_window + 1):

            title = f"(frame - {-offset})" if offset <= 0 else f"(frame + {offset})"
            j = offset + i
            f = real_frames[j] if 0 <= j < len(real_frames) and real_frames[j] >= 2 else None
            
            if f is None:
                res.update({f"{k}{title}": np.nan for k in FEATURES})
            else:

                s = ball_tracker[f]

                vx, vy = s["velocity"] if isinstance(s.get("velocity"), list) else (np.nan, np.nan)

                hx, hy = s["homography location"] if isinstance(s.get("homography location"), list) else (np.nan, np.nan)

                dists = [np.hypot(p["homography location"][0] - hx, p["homography location"][1] - hy)
                        for p in predictions_dict.get(f, []) if p["class"] != "ball" and p.get("box")]
                est = s.get("estimation")

                res.update({
                    f"SPEED{title}": s.get("speed", np.nan),
                    f"VX{title}": vx, f"VY{title}": vy,
                    f"ANGLE{title}": s.get("angle", np.nan),
                    f"HOMOGRAPHY X{title}": hx, f"HOMOGRAPHY Y{title}": hy,
                    f"DISTANCE FROM NEAREST PLAYER{title}": min(dists) if dists else np.nan,
                    f"ESTIMATION{title}": np.nan if est is None else int(est),
                })

        return res
           
    frame_id = 1

    rows = []
    
    while frame_id < len(ball_tracker):

        if ball_tracker.get(frame_id).get("repeat_frame"):

            frame_id += 1
            continue

        rolling_windows = get_windows(
            frame_id=frame_id, 
            ball_tracker=ball_tracker, 
            predictions_dict=predictions_dict, 
            bounce_window=bounce_window,
            real_frames=real_frames
        )

        rolling_windows['FRAME'] = frame_id

        rows.append(rolling_windows)
        frame_id += 1

    training_data = pd.DataFrame(rows)

    return training_data
