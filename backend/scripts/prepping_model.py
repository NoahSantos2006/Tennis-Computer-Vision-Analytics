import xgboost as XGB
from xgboost import XGBClassifier
import pandas as pd
import numpy as np
import os
from pathlib import Path
import json

from scripts.ball_tracker import BallTracker

def get_training_dataframe(
    ball_tracker: dict,
    BOUNCE_LABELS_PATH: Path,
    HIT_LABELS_PATH: Path,
    VIDEO_FILENAME: str,
    PREDICTIONS_FILE: Path,
    bounce_window: int = 5,
):

    ball_tracker = {int(k): v for k, v in ball_tracker.items()}

    with open(BOUNCE_LABELS_PATH, "r") as file:

        bounces = file.read().split("\n")
        bounces = [int(val) for val in bounces]

    with open(HIT_LABELS_PATH, "r") as file:

        hits = file.read().split("\n")
        hits = [int(val) for val in hits]

    with open(PREDICTIONS_FILE, "r") as file:

        predictions_dict = json.load(file)
        predictions_dict = {int(k): v for k, v in predictions_dict.items()}

    def get_windows(
        frame_id: int,
        ball_tracker: dict,
        predictions_dict: dict,
        bounce_window: int
    ) -> dict:

        res = {}

        start_frame = frame_id - bounce_window
        end_frame = frame_id + bounce_window
        current_disparity = bounce_window
        
        while start_frame < 2:

            if current_disparity < 0:
                disparity_title = f"(frame + {abs(current_disparity)})"
            else:
                disparity_title = f"(frame - {abs(current_disparity)})"

            res[f"SPEED{disparity_title}"] = np.nan

            res[f"VX{disparity_title}"] = np.nan
            res[f"VY{disparity_title}"] = np.nan

            res[f"ANGLE{disparity_title}"] = np.nan

            res[f"HOMOGRAPHY X{disparity_title}"] = np.nan
            res[f"HOMOGRAPHY Y{disparity_title}"] = np.nan

            res[f"DISTANCE FROM NEAREST PLAYER{disparity_title}"] = np.nan

            res[f"ESTIMATION{disparity_title}"] = np.nan

            current_disparity -= 1
            start_frame += 1

        while start_frame <= len(ball_tracker) and start_frame <= end_frame:

            if current_disparity < 0:
                disparity_title = f"(frame + {abs(current_disparity)})"
            else:
                disparity_title = f"(frame - {abs(current_disparity)})"

            current_frame_stats = ball_tracker.get(start_frame, {})

            current_frame_speed = current_frame_stats.get("speed", np.nan)
            res[f"SPEED{disparity_title}"] = current_frame_speed

            current_frame_velocity = current_frame_stats.get("velocity", np.nan)
            if not isinstance(current_frame_velocity, tuple):
                current_frame_vx, current_frame_vy = np.nan, np.nan
            else:
                current_frame_vx, current_frame_vy = current_frame_velocity

            res[f"VX{disparity_title}"] = current_frame_vx
            res[f"VY{disparity_title}"] = current_frame_vy

            current_frame_angle = current_frame_stats.get("angle", np.nan)
            res[f"ANGLE{disparity_title}"] = current_frame_angle

            current_frame_homography_location = current_frame_stats.get("homography location", np.nan)
            if not isinstance(current_frame_homography_location, tuple):
                current_frame_homography_x, current_frame_homography_y = np.nan, np.nan
            else:
                current_frame_homography_x, current_frame_homography_y = current_frame_homography_location

            res[f"HOMOGRAPHY X{disparity_title}"] = current_frame_homography_x
            res[f"HOMOGRAPHY Y{disparity_title}"] = current_frame_homography_y

            current_predictions = predictions_dict.get(frame_id, [])
            closest_player_to_ball = None
            for pred in current_predictions:

                if pred['class'] == "ball" or not pred.get('box'): continue

                playerx, playery = pred['homography location']

                current_pixels_away = np.hypot(playerx - current_frame_homography_x, playery - current_frame_homography_y)
                        
                if closest_player_to_ball is None:
        
                    closest_player_to_ball = [pred['homography location'], current_pixels_away]
        
                else:
        
                    if current_pixels_away < closest_player_to_ball[1]:
                        closest_player_to_ball = [pred['homography location'], current_pixels_away]

            if closest_player_to_ball is not None:
                distance_to_closest_player = closest_player_to_ball[1]
            else:
                distance_to_closest_player = np.nan
            res[f"DISTANCE FROM NEAREST PLAYER{disparity_title}"] = distance_to_closest_player

            current_frame_is_estimation = current_frame_stats.get("estimation", np.nan)
            if np.isnan(current_frame_is_estimation):
                res[f"ESTIMATION{disparity_title}"] = np.nan
            else:
                res[f"ESTIMATION{disparity_title}"] = int(current_frame_is_estimation)

            current_disparity -= 1
            start_frame += 1

        while start_frame <= end_frame:

            if current_disparity < 0:
                disparity_title = f"(frame + {abs(current_disparity)})"
            else:
                disparity_title = f"(frame - {abs(current_disparity)})"

            res[f"SPEED{disparity_title}"] = np.nan

            res[f"VX{disparity_title}"] = np.nan
            res[f"VY{disparity_title}"] = np.nan

            res[f"ANGLE{disparity_title}"] = np.nan

            res[f"HOMOGRAPHY X{disparity_title}"] = np.nan
            res[f"HOMOGRAPHY Y{disparity_title}"] = np.nan

            res[f"DISTANCE FROM NEAREST PLAYER{disparity_title}"] = np.nan

            res[f"ESTIMATION{disparity_title}"] = np.nan

            current_disparity -= 1
            start_frame += 1

        return res
           
    frame_id = 1

    """
    
    Features:

        - Speed
        - Velocity
        - Homography Location
        - Angle

    Rolling window of +- 5
    
    """

    rows = []

    while frame_id < len(ball_tracker):

        rolling_windows = get_windows(frame_id=frame_id, ball_tracker=ball_tracker, predictions_dict=predictions_dict, bounce_window=bounce_window)

        if frame_id in bounces:

            rolling_windows['LABEL'] = 1

        elif frame_id in hits:

            rolling_windows['LABEL'] = 2

        else:

            rolling_windows['LABEL'] = 0

        rolling_windows['VIDEO_ID'] = VIDEO_FILENAME

        rows.append(rolling_windows)
        frame_id += 1

    training_data = pd.DataFrame(rows)

    return training_data

if __name__ == "__main__":

    pass