import json
from pathlib import Path
import numpy as np
import cv2
import time
import sys
import os
import matplotlib.pyplot as plt
import pandas as pd
from xgboost import XGBClassifier
from tqdm import tqdm


from backend.scripts.ball_tracker import BallTracker
from backend.scripts.status import update_status

BASE_DIR = Path(__file__).parent.parent

TENNIS_COURT_LENGTH = 23.77
TENNIS_COURT_WIDTH = 10.97
TENNIS_COURT_SCALE = 25
TENNIS_COURT_PADDING = 30

def get_coordinates_and_center(prediction: dict) -> tuple:

    x = prediction['x']
    y = prediction['y']
    w = prediction['width']
    h = prediction['height']

    x1 = x - w / 2
    y1 = y - h / 2
    x2 = x + w / 2
    y2 = y + h / 2

    return [x1, y1, x2, y2], (x, y)

def detection_of_court_points(box: tuple, H: np.array) -> tuple:

    x1, y1, x2, y2 = box

    # Find the center of the x coordinate
    ground_x = (x1 + x2) / 2
    ground_y = y2

    video_point = np.array(
        [[[ground_x, ground_y]]],
        dtype=np.float32
    )

    court_point = cv2.perspectiveTransform(video_point, H)

    court_x, court_y = court_point[0, 0]

    return int(court_x), int(court_y)

def ball_near_player(
    frame_id: int,
    predictions_by_frame: dict,
    ball_center: tuple,
    HOMOGRAPHY_MATRIX: np.array,
    PLAYER_WIDTH_PADDING_RATIO: float = 0.20,
    PLAYER_HEIGHT_PADDING_RATIO: float = 0.10
) -> tuple:

    found_player_near_ball = False
    closest_player_to_ball = None
    player_predictions = predictions_by_frame.get(frame_id)

    if not player_predictions: return False, (-1, -1)

    for pred in player_predictions:
    
        if pred['class'] == "ball" or not pred.get('box'): continue

        player_height = pred.get('height')
        player_width = pred.get("width")
        horizontal_padding = int(player_width * PLAYER_WIDTH_PADDING_RATIO)
        vertical_padding = int(player_height * PLAYER_HEIGHT_PADDING_RATIO)

        ball_x, ball_y = ball_center
        player_x1, player_y1, player_x2, player_y2 = pred.get('box')
        
        location = detection_of_court_points(
            box=np.array(pred['box'], dtype=np.float32),
            H=HOMOGRAPHY_MATRIX
        )

        ballx, bally = ball_center
        playerx, playery = location

        current_pixels_away = np.hypot(playerx - ballx, playery - bally)
        
        if closest_player_to_ball is None:

            closest_player_to_ball = [location, current_pixels_away]

        else:

            if current_pixels_away < closest_player_to_ball[1]:
                closest_player_to_ball = [location, current_pixels_away]

        if not found_player_near_ball:

            padded_x1 = player_x1 - horizontal_padding
            padded_y1 = player_y1 - vertical_padding
            padded_x2 = player_x2 + horizontal_padding
            padded_y2 = player_y2 + vertical_padding

            if (
                padded_x1 <= ball_x <= padded_x2 and
                padded_y1 <= ball_y <= padded_y2
            ):

                found_player_near_ball = True

    if found_player_near_ball:
        return True, closest_player_to_ball[0] # player location

    return False, (-1, -1)

def classify_bounce(
    frame_id: int,
    ball_tracker: dict,
    predictions_dict: dict,
    model: XGBClassifier,
    bounce_window: int = 5,
):

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
         
    windows_dictionary = get_windows(
        frame_id=frame_id,
        ball_tracker=ball_tracker,
        predictions_dict=predictions_dict,
        bounce_window=bounce_window
    )

    df = pd.DataFrame([windows_dictionary])
    
    label = model.predict(df)[0]

    return label

def get_bounces(
    ball_tracker_predictions: dict,
    vision_model_predictions: dict,
    XGBoost_model: XGBClassifier,
    STATUS_PATH: Path,
    BOUNCE_FRAME_TOLERANCE: int = 5
) -> dict:

    frame_id = 1
    last_bounced = None
    results = {}
    total_frames = len(ball_tracker_predictions)

    while frame_id < len(ball_tracker_predictions):

        label = int(classify_bounce(
            frame_id=frame_id,
            ball_tracker=ball_tracker_predictions,
            predictions_dict=vision_model_predictions,
            model=XGBoost_model
        ))

        location = ball_tracker_predictions[frame_id].get("homography location", None)

        if label == 0 or not location:

            frame_id += 1
            continue

        if not last_bounced:

            results[frame_id] = {
                "label": label,
                "homography location": location
            }
            last_bounced = frame_id

        else:

            if not last_bounced >= frame_id - BOUNCE_FRAME_TOLERANCE:

                results[frame_id] = {
                    "label": label,
                    "homography location": location
                }
                last_bounced = frame_id

        update_status(
            status_file=STATUS_PATH,
            status="in progress",
            stage="Detecting Bounces and Hits",
            current_frame=frame_id,
            total_frames=total_frames
        )

        frame_id += 1

    return results

def find_angles(
        ball_tracker_class: BallTracker,
    ) -> BallTracker:
        
    ball_tracker = ball_tracker_class.tracker

    # cleaning dict since the keys turn into strings after json serialize
    for k, v in ball_tracker.items(): 
        if isinstance(k, str): 
            ball_tracker = {int(keys): vals for keys, vals in ball_tracker.items()}
        break

    frame_id = 1

    # bounce detection
    while frame_id < len(ball_tracker):
        
        if (
            frame_id < 2 or
            ball_tracker[frame_id - 1]['vision model location'] == (-1, -1) or
            ball_tracker[frame_id]['vision model location'] == (-1, -1) or
            ball_tracker[frame_id + 1]['vision model location'] == (-1, -1)

        ):

            frame_id += 1
            continue

        # find locations
        x1, y1 = ball_tracker[frame_id - 1]['vision model location']
        x2, y2 = ball_tracker[frame_id]['vision model location']
        x3, y3 = ball_tracker[frame_id + 1]['vision model location']

        vx = (x3 - x1) / 2
        vy = (y3 - y1) / 2

        # calculates magnitude of a 2D velocity vector (pythagorean theorem)
        speed = np.hypot(vx, vy)

        # movement vectors
        v1 = np.array([x2 - x1, y2 - y1])
        v2 = np.array([x3 - x2, y3 - y2])

        # normalize vectors in case they have different lengths (only care about direction)

        v1norm = np.linalg.norm(v1)
        if v1norm > 0:
            v1 = v1 / v1norm
        else:
            v1 = np.zeros_like(v1)

        v2norm = np.linalg.norm(v2)
        if v2norm > 0:
            v2 = v2 / v2norm
        else:
            v2 = np.zeros_like(v2)

        # compute dot product to measure similarity between directions
        dot = np.clip(np.dot(v1, v2), -1, 1)

        # convert dot product into angle
        angle = np.arccos(dot)      # radians
        
        # convert to degrees
        angle_deg = np.degrees(angle)

        ball_tracker[frame_id]['angle'] = angle_deg
        ball_tracker[frame_id]['velocity'] = (vx, vy)
        ball_tracker[frame_id]['speed'] = speed

        frame_id += 1

    return ball_tracker_class

def run_predictions(
        COURT_POINTS_INPUT_FILE: Path, 
        PREDICTIONS_INPUT_FILE: Path, 
        ball_tracker: BallTracker,
    ) -> BallTracker:
        
    with open(PREDICTIONS_INPUT_FILE, "r") as f:

        print(f"Opening {PREDICTIONS_INPUT_FILE}")
        predictions_by_frame = json.load(f)
        total_frames = len(predictions_by_frame)

    with open(COURT_POINTS_INPUT_FILE, "r") as f:

        print(f"Opening {COURT_POINTS_INPUT_FILE}")
        court_points = json.load(f)
        court_points = {int(k): v for k, v in court_points.items()}


    for frame_id in range(1, total_frames + 1):

        predictions_array = predictions_by_frame.get(str(frame_id))

        current_ball_locations = []
    
        for pred in predictions_array:

            if pred['class'] == 'ball' and pred['confidence'] >= 0.5:

                coordinates, center = get_coordinates_and_center(prediction=pred)
                current_ball_locations.append(center)

            if pred['class'] != 'ball' and 'box' in pred:

                if frame_id in court_points:

                    HOMOGRAPHY_MATRIX = compute_homography(COURT_POINTS_PATH=COURT_POINTS_INPUT_FILE, frame_id=frame_id)

                location = detection_of_court_points(
                    box=np.array(pred['box'], dtype=np.float32),
                    H=HOMOGRAPHY_MATRIX
                )

                pred['homography location'] = location

        ball_tracker.update(
            frame_id=frame_id, 
            ball_locations=current_ball_locations
        )

    with open(PREDICTIONS_INPUT_FILE, "w") as f:
    
        print(f"Saving a new predictions file to {PREDICTIONS_INPUT_FILE}")
        json.dump(predictions_by_frame, f, indent=4)

    ball_tracker_class = find_angles(
            ball_tracker_class=ball_tracker
    )

    return predictions_by_frame, ball_tracker_class

def compute_homography(COURT_POINTS_PATH: Path, frame_id: int) -> np.array:

    court_points = []

    with open(COURT_POINTS_PATH, "r") as f:
        
        video_points = json.load(f)
        video_points = {int(k): v for k, v in video_points.items()}

    top_down_points = np.array([
        [TENNIS_COURT_PADDING, TENNIS_COURT_PADDING + 23.77 * TENNIS_COURT_SCALE],                              # 1 - near left doubles baseline
        [TENNIS_COURT_PADDING + 1.37 * TENNIS_COURT_SCALE, TENNIS_COURT_PADDING + 23.77 * TENNIS_COURT_SCALE],  # 2 - near left singles baseline
        [TENNIS_COURT_PADDING + 9.60 * TENNIS_COURT_SCALE, TENNIS_COURT_PADDING + 23.77 * TENNIS_COURT_SCALE],  # 3 - near right singles baseline
        [TENNIS_COURT_PADDING + 10.97 * TENNIS_COURT_SCALE, TENNIS_COURT_PADDING + 23.77 * TENNIS_COURT_SCALE], # 4 - near right doubles baseline
        [TENNIS_COURT_PADDING + 10.97 * TENNIS_COURT_SCALE, TENNIS_COURT_PADDING],                              # 5 - far right doubles baseline
        [TENNIS_COURT_PADDING + 9.60 * TENNIS_COURT_SCALE, TENNIS_COURT_PADDING],                               # 6 - far right singles baseline
        [TENNIS_COURT_PADDING + 1.37 * TENNIS_COURT_SCALE, TENNIS_COURT_PADDING],                               # 7 - far left singles baseline
        [TENNIS_COURT_PADDING, TENNIS_COURT_PADDING],                                                           # 8 - far left doubles baseline
        [TENNIS_COURT_PADDING + 1.37 * TENNIS_COURT_SCALE, TENNIS_COURT_PADDING + 18.285 * TENNIS_COURT_SCALE], # 9 - near left service line
        [TENNIS_COURT_PADDING + 1.37 * TENNIS_COURT_SCALE, TENNIS_COURT_PADDING + 5.485 * TENNIS_COURT_SCALE],  # 10 - far left service line
        [TENNIS_COURT_PADDING + 9.60 * TENNIS_COURT_SCALE, TENNIS_COURT_PADDING + 18.285 * TENNIS_COURT_SCALE], # 11 - near right service line
        [TENNIS_COURT_PADDING + 9.60 * TENNIS_COURT_SCALE, TENNIS_COURT_PADDING + 5.485 * TENNIS_COURT_SCALE],  # 12 - far right service line
        [TENNIS_COURT_PADDING + 5.485 * TENNIS_COURT_SCALE, TENNIS_COURT_PADDING + 18.285 * TENNIS_COURT_SCALE],# 13 - near centre service line
        [TENNIS_COURT_PADDING + 5.485 * TENNIS_COURT_SCALE, TENNIS_COURT_PADDING + 5.485 * TENNIS_COURT_SCALE]],# 14 - far centre service line
        dtype=np.float32
    )

    current_court_points = video_points.get(frame_id, [])
    if not current_court_points:

        return []

    for keypoint_detection_dict in current_court_points:
    
        curr_x = keypoint_detection_dict['x']
        curr_y = keypoint_detection_dict['y']

        court_points.append((curr_x, curr_y))
                
    court_points = np.array(court_points, dtype=np.float32)
    
    H, mask = cv2.findHomography(
        court_points,
        top_down_points,
        method=cv2.RANSAC,
        ransacReprojThreshold=3.0
    )

    if H is None: raise RuntimeError("Homography could not be computed")

    # Homography Matrix
    return H
