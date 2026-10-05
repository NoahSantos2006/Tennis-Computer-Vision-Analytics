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
from dotenv import load_dotenv


from backend.scripts.ball_tracker import BallTracker
from backend.scripts.prepping_model import acquire_training_dataframes, pick_events

BASE_DIR = Path(__file__).parent.parent

load_dotenv()

TENNIS_COURT_LENGTH = float(os.getenv("TENNIS_COURT_LENGTH"))
TENNIS_COURT_WIDTH = float(os.getenv("TENNIS_COURT_WIDTH"))
TENNIS_COURT_SCALE = int(os.getenv("TENNIS_COURT_SCALE"))
TENNIS_COURT_PADDING = int(os.getenv("TENNIS_COURT_PADDING"))

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

def get_bounces(
    VIDEO_FILENAME: str,
    OUTPUT_PATH: Path,
    MODEL: XGBClassifier,
    BALL_TRACKER_PREDICTIONS: dict,
    XGB_BOUNCE_THRESHOLD: float = float(os.getenv("XGB_BOUNCE_THRESHOLD")),
    XGB_HIT_THRESHOLD: float = float(os.getenv("XGB_HIT_THRESHOLD"))
) -> tuple:

    df = acquire_training_dataframes(
        VIDEO_FILENAME=VIDEO_FILENAME,
        OUTPUT_PATH=OUTPUT_PATH
    )

    X = df.drop(columns=["FRAME"])

    proba = MODEL.predict_proba(X)

    frames = df["FRAME"].to_numpy()
    order = np.argsort(frames)
    frames, proba = frames[order], proba[order]

    bounces = pick_events(frames, proba[:, 1], threshold=XGB_BOUNCE_THRESHOLD)
    hits    = pick_events(frames, proba[:, 2], threshold=XGB_HIT_THRESHOLD)

    frame_id = 1
    results = {}
    while frame_id < len(BALL_TRACKER_PREDICTIONS):

        location = BALL_TRACKER_PREDICTIONS[frame_id].get("homography location", None)

        if frame_id in bounces: label = 1
        elif frame_id in hits: label = 2
        else:
            frame_id += 1
            continue

        results[frame_id] = {
            "label": label,
            "homography location": location
        }

        frame_id += 1

    return results

def find_angles(ball_tracker_class: BallTracker) -> BallTracker:
        
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
            ball_tracker[frame_id]['repeat_frame'] or
            frame_id < 2
        ):

            frame_id += 1
            continue
        
        prev_frame = frame_id - 1
        next_frame = frame_id + 1

        while prev_frame > 1 and ball_tracker[prev_frame]['repeat_frame']:

            prev_frame -= 1

        while next_frame < len(ball_tracker) and ball_tracker[next_frame]['repeat_frame']:

            next_frame += 1

        if (
            ball_tracker[next_frame]['repeat_frame'] or
            ball_tracker[prev_frame]['vision model location'] == (-1, -1) or
            ball_tracker[frame_id]['vision model location'] == (-1, -1) or
            ball_tracker[next_frame]['vision model location'] == (-1, -1)

        ):

            frame_id += 1  
            continue

        # find locations
        x1, y1 = ball_tracker[prev_frame]['vision model location']
        x2, y2 = ball_tracker[frame_id]['vision model location']
        x3, y3 = ball_tracker[next_frame]['vision model location']

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

        predictions_by_frame = json.load(f)
        total_frames = len(predictions_by_frame)

    with open(COURT_POINTS_INPUT_FILE, "r") as f:

        court_points = json.load(f)
        court_points = {int(k): v for k, v in court_points.items()}


    for frame_id in range(1, total_frames + 1):

        predictions_array = predictions_by_frame.get(str(frame_id))

        if predictions_array == "Repeat Frame":

            ball_tracker.update(
                frame_id=frame_id,
                ball_locations=[],
                repeat_frame=True
            )

        else:

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
    
        json.dump(predictions_by_frame, f, indent=4)

    ball_tracker_class = find_angles(ball_tracker_class=ball_tracker)

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
