import os
import cv2 as cv2
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import json
import sys
import pickle

BASE_DIR = Path(__file__).parent.parent

from scripts.ball_tracker import BallTracker
from scripts.side_functions import get_coordinates_and_center, validate_bounce, ball_near_player, plot_ball_trajectory

"""
PICKLEBALL COURT MEASUREMENTS

                    20 ft
    <------------------------------->
   (0, 44)                           (20, 44)
    *-------------------------------*        -         -
    |               |               |        |         |
    |               |               |        |         |
    |               |               |        | 15 ft   |
    |               |               |        |         |
    |               |               |        |         |
    |(0, 29)        |               |(20, 29)|         |
    |-------------------------------*        -         |
    |                       -       |                  |
    |                       | 7 ft  |                  |
    |                       -       |                  |
    |-------------------------------* Net              | 44 ft
    |                               |                  |
    |                               |                  |
    |(0, 15)                        |(20, 15)          |
    |-------------------------------*                  |
    |               |               |                  |
    |               |               |                  |
    |               |               |                  |
    |               |               |                  |
    |               |               |                  |
    |               |               |                  |
    *-------------------------------*                  -
  (0, 0)                          (20, 0)
"""

"""

BALL TRACKER

Dictionary with:

    1. FRAME NUMBER
    2. BALL PREDICTIONS
    3. CHOSEN BALL LOCATION
    4. WHETHER IT'S AN ESTIMATION OR NOT
    5. TIES
        - if the last ball prediction had a tie between the amount of pixels between the previous detection than compare to the next one

    EXAMPLE:

    ball_tracker = {
        1: {
            ball_predictions: [
                {
                    homography_location: (x, y),
                    image_location: (x1, y1, x2, y2)
                }
            ]
            homography_location: (x, y)
            estimation: True
            ties: [(x1, y1), (x2, y2)]
        }
    }

Also add incorrect predictions like a light spot on the ground. The coordinates would be the same so we can rule out that detection

 - location predictions, if the x, y coordinate is the same then increase counter and a threshold will be met to rule out that detection for future predictions
 - might use x1, y1, x2, y2 for that so if the camera is still, then it's easier compared to homography

"""

"""

pickleball bouncing coordinates window

ex 1
    1.  [532.5, 701.5]
    2.  [504.0, 757.5]
    3.  [479.0, 767.0] --> bounced
    4.  [453.5, 767.0]
    5.  [430.0, 769.5]

ex 2
    1. [425.0, 307.0]
    2. [396.0, 326.5]
    3. [366.0, 347.5] --> bounced
    4. [347.0, 345.0]
    5. [328.5, 337.5]
    6. [310.5, 330.5]

ex 3
    1. [1345.0, 737.5]
    2. [1343.5, 781.5]
    3. [1343.0, 820.5] --> bounced
    4. [1346.0, 816.5]
    5. [1348.5, 806.5]

    

THINGS THAT I COULD IMPLEMENT

    - might implement avoiding detections that were in the same homographical spot for multiple frames meaning a false positive
"""

PICKLEBALL_COURT_HEIGHT = 44
PICKLEBALL_COURT_HEIGHT = 20
PICKLEBALL_COURT_SCALE = 20
PICKLEBALL_COURT_PADDING = 50

TENNIS_COURT_LENGTH = 23.77
TENNIS_COURT_WIDTH = 10.97
TENNIS_COURT_SCALE = 25
TENNIS_COURT_PADDING = 30




def compute_homography(video_points: np.array, sport: str) -> np.array:

    if sport == "pickleball":

        top_down_points = np.array([
            [PICKLEBALL_COURT_PADDING, PICKLEBALL_COURT_PADDING + 44 * PICKLEBALL_COURT_SCALE],                # near left baseline    ( 50 , 930 ) 
            [PICKLEBALL_COURT_PADDING + 20 * PICKLEBALL_COURT_SCALE, PICKLEBALL_COURT_PADDING + 44 * PICKLEBALL_COURT_SCALE],   # near right baseline   ( 450, 930 )
            [PICKLEBALL_COURT_PADDING + 20 * PICKLEBALL_COURT_SCALE, PICKLEBALL_COURT_PADDING],                # far right baseline    ( 450, 50  )
            [PICKLEBALL_COURT_PADDING, PICKLEBALL_COURT_PADDING],                             # far left baseline     ( 50 , 50  )
            [PICKLEBALL_COURT_PADDING, PICKLEBALL_COURT_PADDING + 29 * PICKLEBALL_COURT_SCALE],                # near left kitchen     ( 50 , 630 )
            [PICKLEBALL_COURT_PADDING + 20 * PICKLEBALL_COURT_SCALE, PICKLEBALL_COURT_PADDING + 29 * PICKLEBALL_COURT_SCALE],   # near right kitchen    ( 450, 630 )
            [PICKLEBALL_COURT_PADDING, PICKLEBALL_COURT_PADDING + 15 * PICKLEBALL_COURT_SCALE],                # far right kitchen     ( 50 , 350 ) 
            [PICKLEBALL_COURT_PADDING + 20 * PICKLEBALL_COURT_SCALE, PICKLEBALL_COURT_PADDING + 15 * PICKLEBALL_COURT_SCALE]],  # far left kitchen      ( 450, 350 )
            dtype=np.float32
        )
    elif sport == "tennis":

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
    
    H, mask = cv2.findHomography(
        video_points,
        top_down_points,
        method=cv2.RANSAC,
        ransacReprojThreshold=3.0
    )

    if H is None: raise RuntimeError("Homography could not be computed")

    # Homography Matrix
    return H

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

def draw_pickleball_court() -> np.array:

    OUTPUT_WIDTH = int(PICKLEBALL_COURT_HEIGHT * PICKLEBALL_COURT_SCALE)
    OUTPUT_HEIGHT = int(PICKLEBALL_COURT_HEIGHT * PICKLEBALL_COURT_SCALE)

    # 1. Create a black canvas (500x500 pixels, 3 color channels)
    image = np.full((OUTPUT_HEIGHT + int(PICKLEBALL_COURT_PADDING*2), OUTPUT_WIDTH + int(PICKLEBALL_COURT_PADDING*2), 3), (60, 110, 60), dtype="uint8")
    
    """
    Top size
    """

    # Far Side
    cv2.rectangle(
        image,
        (PICKLEBALL_COURT_PADDING, PICKLEBALL_COURT_PADDING + 15 * PICKLEBALL_COURT_SCALE),                     
        (PICKLEBALL_COURT_PADDING + 20 * PICKLEBALL_COURT_SCALE, PICKLEBALL_COURT_PADDING),                        
        (235, 160, 85),
        -1                                  # Thickness (positive number --> outline only; -1 --> fill entire rectangle)
    )


    # Kitchen
    cv2.rectangle(
        image,
        (PICKLEBALL_COURT_PADDING, PICKLEBALL_COURT_PADDING + 29 * PICKLEBALL_COURT_SCALE),
        (PICKLEBALL_COURT_PADDING + 20 * PICKLEBALL_COURT_SCALE, PICKLEBALL_COURT_PADDING + 15 * PICKLEBALL_COURT_SCALE),
        (150, 60, 38),
        -1
    )

    # Near Side
    cv2.rectangle(
        image,
        (PICKLEBALL_COURT_PADDING, PICKLEBALL_COURT_PADDING + 44 * PICKLEBALL_COURT_SCALE),
        (PICKLEBALL_COURT_PADDING + 20 * PICKLEBALL_COURT_SCALE, PICKLEBALL_COURT_PADDING + 29 * PICKLEBALL_COURT_SCALE),
        (235, 160, 85),
        -1
    )

    # Net
    cv2.line(
        image,
        (PICKLEBALL_COURT_PADDING, PICKLEBALL_COURT_PADDING + 22 * PICKLEBALL_COURT_SCALE),
        (PICKLEBALL_COURT_PADDING + 20 * PICKLEBALL_COURT_SCALE, PICKLEBALL_COURT_PADDING + 22 * PICKLEBALL_COURT_SCALE),
        (0, 0, 0),
        3
    )

    # Far Kitchen Line
    cv2.line(
        image,
        (PICKLEBALL_COURT_PADDING, PICKLEBALL_COURT_PADDING + 15 * PICKLEBALL_COURT_SCALE),
        (PICKLEBALL_COURT_PADDING + 20 * PICKLEBALL_COURT_SCALE, PICKLEBALL_COURT_PADDING + 15 * PICKLEBALL_COURT_SCALE),
        (255, 255, 255),
        3
    )

    # Near Kitchen Line
    cv2.line(
        image,
        (PICKLEBALL_COURT_PADDING, PICKLEBALL_COURT_PADDING + 29 * PICKLEBALL_COURT_SCALE),
        (PICKLEBALL_COURT_PADDING + 20 * PICKLEBALL_COURT_SCALE, PICKLEBALL_COURT_PADDING + 29 * PICKLEBALL_COURT_SCALE),
        (255, 255, 255),
        3
    )

    # Top Baseline
    cv2.line(
        image,
        (PICKLEBALL_COURT_PADDING, PICKLEBALL_COURT_PADDING),
        (PICKLEBALL_COURT_PADDING + 20 * PICKLEBALL_COURT_SCALE, PICKLEBALL_COURT_PADDING),
        (255, 255, 255),
        3 
    )

    # Left SideLine
    cv2.line(
        image,
        (PICKLEBALL_COURT_PADDING, PICKLEBALL_COURT_PADDING),
        (PICKLEBALL_COURT_PADDING, PICKLEBALL_COURT_PADDING + 44 * PICKLEBALL_COURT_SCALE),
        (255, 255, 255),
        3
    )

    # Top Side Line Seperator
    cv2.line(
        image,
        (PICKLEBALL_COURT_PADDING + 10 * PICKLEBALL_COURT_SCALE, PICKLEBALL_COURT_PADDING + 15 * PICKLEBALL_COURT_SCALE),
        (PICKLEBALL_COURT_PADDING + 10 * PICKLEBALL_COURT_SCALE, PICKLEBALL_COURT_PADDING),
        (255, 255, 255),
        3
    )

    # Bottom Baseline
    cv2.line(
        image,
        (PICKLEBALL_COURT_PADDING, PICKLEBALL_COURT_PADDING + 44 * PICKLEBALL_COURT_SCALE),
        (PICKLEBALL_COURT_PADDING + 20 * PICKLEBALL_COURT_SCALE, PICKLEBALL_COURT_PADDING + 44 * PICKLEBALL_COURT_SCALE),
        (255, 255, 255),
        3
    )

    # Right Sideline
    cv2.line(
        image,
        (PICKLEBALL_COURT_PADDING + 20 * PICKLEBALL_COURT_SCALE, PICKLEBALL_COURT_PADDING),
        (PICKLEBALL_COURT_PADDING + 20 * PICKLEBALL_COURT_SCALE, PICKLEBALL_COURT_PADDING + 44 * PICKLEBALL_COURT_SCALE),
        (255, 255, 255),
        3
    )

    # Near Side Line Seperator
    cv2.line(
        image,
        (PICKLEBALL_COURT_PADDING + 10 * PICKLEBALL_COURT_SCALE, PICKLEBALL_COURT_PADDING + 44 * PICKLEBALL_COURT_SCALE),
        (PICKLEBALL_COURT_PADDING + 10 * PICKLEBALL_COURT_SCALE, PICKLEBALL_COURT_PADDING + 29 * PICKLEBALL_COURT_SCALE),
        (255, 255, 255),
        3
    )

    return image

def draw_tennis_court() -> np.array:

    # Tennis court dimensions in metres
    COURT_LENGTH = 23.77
    COURT_WIDTH = 10.97

    DOUBLES_ALLEY = 1.37
    SERVICE_LINE_FROM_NET = 6.40

    HALF_LENGTH = COURT_LENGTH / 2
    HALF_WIDTH = COURT_WIDTH / 2

    # Drawing settings
    SCALE = 25
    PADDING = 30

    # Vertical orientation:
    # width = court width
    # height = court length
    OUTPUT_WIDTH = int(COURT_WIDTH * SCALE)
    OUTPUT_HEIGHT = int(COURT_LENGTH * SCALE)

    image = np.full(
        (
            OUTPUT_HEIGHT + PADDING * 2,
            OUTPUT_WIDTH + PADDING * 2,
            3
        ),
        (60, 110, 60),
        dtype="uint8"
    )

    # Convert court coordinates in metres to pixels
    # x = court width
    # y = court length
    def point(x, y):
        return (
            int(PADDING + x * SCALE),
            int(PADDING + y * SCALE)
        )

    WHITE = (255, 255, 255)
    COURT_COLOR = (80, 150, 80)
    SERVICE_COLOR = (70, 140, 70)

    # -----------------------------------
    # Court background
    # -----------------------------------
    cv2.rectangle(
        image,
        point(0, 0),
        point(COURT_WIDTH, COURT_LENGTH),
        COURT_COLOR,
        -1
    )

    # -----------------------------------
    # Important Y positions
    # -----------------------------------
    top_service_y = HALF_LENGTH - SERVICE_LINE_FROM_NET
    bottom_service_y = HALF_LENGTH + SERVICE_LINE_FROM_NET

    # -----------------------------------
    # Service box area
    # -----------------------------------
    cv2.rectangle(
        image,
        point(DOUBLES_ALLEY, top_service_y),
        point(COURT_WIDTH - DOUBLES_ALLEY, bottom_service_y),
        SERVICE_COLOR,
        -1
    )

    # -----------------------------------
    # Outer doubles lines
    # -----------------------------------

    # Top baseline
    cv2.line(
        image,
        point(0, 0),
        point(COURT_WIDTH, 0),
        WHITE,
        3
    )

    # Bottom baseline
    cv2.line(
        image,
        point(0, COURT_LENGTH),
        point(COURT_WIDTH, COURT_LENGTH),
        WHITE,
        3
    )

    # Left doubles sideline
    cv2.line(
        image,
        point(0, 0),
        point(0, COURT_LENGTH),
        WHITE,
        3
    )

    # Right doubles sideline
    cv2.line(
        image,
        point(COURT_WIDTH, 0),
        point(COURT_WIDTH, COURT_LENGTH),
        WHITE,
        3
    )

    # -----------------------------------
    # Singles sidelines
    # -----------------------------------

    # Left singles sideline
    cv2.line(
        image,
        point(DOUBLES_ALLEY, 0),
        point(DOUBLES_ALLEY, COURT_LENGTH),
        WHITE,
        3
    )

    # Right singles sideline
    cv2.line(
        image,
        point(COURT_WIDTH - DOUBLES_ALLEY, 0),
        point(COURT_WIDTH - DOUBLES_ALLEY, COURT_LENGTH),
        WHITE,
        3
    )

    # -----------------------------------
    # Service lines
    # -----------------------------------

    # Top service line
    cv2.line(
        image,
        point(DOUBLES_ALLEY, top_service_y),
        point(COURT_WIDTH - DOUBLES_ALLEY, top_service_y),
        WHITE,
        3
    )

    # Bottom service line
    cv2.line(
        image,
        point(DOUBLES_ALLEY, bottom_service_y),
        point(COURT_WIDTH - DOUBLES_ALLEY, bottom_service_y),
        WHITE,
        3
    )

    # -----------------------------------
    # Centre service line
    # -----------------------------------
    cv2.line(
        image,
        point(HALF_WIDTH, top_service_y),
        point(HALF_WIDTH, bottom_service_y),
        WHITE,
        3
    )

    # -----------------------------------
    # Net
    # -----------------------------------
    cv2.line(
        image,
        point(0, HALF_LENGTH),
        point(COURT_WIDTH, HALF_LENGTH),
        (0, 0, 0),
        4
    )

    # -----------------------------------
    # Centre marks on baselines
    # -----------------------------------
    CENTER_MARK_LENGTH = 0.10

    # Top baseline centre mark
    cv2.line(
        image,
        point(HALF_WIDTH, 0),
        point(HALF_WIDTH, CENTER_MARK_LENGTH),
        WHITE,
        3
    )

    # Bottom baseline centre mark
    cv2.line(
        image,
        point(HALF_WIDTH, COURT_LENGTH),
        point(HALF_WIDTH, COURT_LENGTH - CENTER_MARK_LENGTH),
        WHITE,
        3
    )

    return image

def live_homography_graph(
    COURT_POINTS_INPUT_FILE: Path,
    PREDICTIONS_INPUT_FILE: Path,
    BALL_TRACKING_CLASS_FILE: Path
):

    PICKLEBALL_COURT_IMAGE = draw_pickleball_court()

    with open(COURT_POINTS_INPUT_FILE, "r") as f:
        print(f"Opening {COURT_POINTS_INPUT_FILE}")
        court_points = json.load(f)

    with open(PREDICTIONS_INPUT_FILE, "r") as f:
        print(f"Opening {PREDICTIONS_INPUT_FILE}")
        predictions_by_frame = json.load(f)
        predictions_by_frame = {int(keys): vals for keys, vals in predictions_by_frame.items()}

    with open(BALL_TRACKING_CLASS_FILE, "rb") as f:
        print(f"Opening {BALL_TRACKING_CLASS_FILE}")
        ball_tracker = pickle.load(f)

    video_points = [
        coordinate
        for point_location, coordinate
        in court_points["image_points"].items()
    ]

    video_points = np.array(video_points, dtype=np.float32)

    HOMOGRAPHY_MATRIX = compute_homography(
        video_points=video_points
    )

    window_name = "Pickleball Court"

    ball_tracking_stats = ball_tracker.tracker

    if len(ball_tracking_stats) == 0:
        print("No frames were found.")
        return

    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

    def on_trackbar_change(value):
        # The frame is drawn in the main loop.
        pass

    cv2.createTrackbar(
        "Frame",
        window_name,
        0,
        len(ball_tracking_stats) - 1,
        on_trackbar_change
    )

    current_frame_index = -1

    while True:

        frame_index = cv2.getTrackbarPos(
            "Frame",
            window_name
        )

        frame_id = 1
        # Only redraw when the slider changes.
        if frame_index != current_frame_index:

            current_frame_index = frame_index

            predictions_array = predictions_by_frame[frame_id]

            court_frame = PICKLEBALL_COURT_IMAGE.copy()
            ball_found = False

            cv2.putText(
                court_frame,
                f"Frame Number: {frame_id}",
                (1, 50),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 0, 0),
                2
            )

            frame_ball_tracking = ball_tracking_stats.get(
                frame_id
            )

            if frame_ball_tracking is None:
                continue

            location = frame_ball_tracking.get(
                "homography location"
            )

            if location == (-1, -1):
                continue

            ball_found = True

            location = (
                int(round(location[0])),
                int(round(location[1]))
            )

            cv2.circle(
                court_frame,
                location,
                5,
                (0, 0, 255),
                -1
            )

            cv2.putText(
                court_frame,
                f"Ball",
                location,
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 255, 255),
                2
            )

            cv2.putText(
                court_frame,
                f"Ball Coordinates: {location}",
                (1, 25),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 0, 0),
                2
            )

            for pred in predictions_array:

                # Skip only additional ball detections.
                # Your original code skipped every remaining prediction
                # after finding the ball.
            
                location = None

                if pred["class"] == "ball": continue

                coordinates, center = get_coordinates_and_center(
                    prediction=pred
                )

                location = detection_of_court_points(
                    box=coordinates,
                    H=HOMOGRAPHY_MATRIX
                )

                if location is None:
                    continue

                location = (
                    int(round(location[0])),
                    int(round(location[1]))
                )

                cv2.circle(
                    court_frame,
                    location,
                    5,
                    (0, 0, 255),
                    -1
                )

                label_location = (
                    location[0],
                    max(location[1] - 10, 15)
                )

                cv2.putText(
                    court_frame,
                    pred["class"],
                    label_location,
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (255, 255, 255),
                    2
                )

            cv2.imshow(
                window_name,
                court_frame
            )

        key = cv2.waitKey(33) & 0xFF

        if key == ord("q"):
            break

        # Previous frame
        elif key == ord("a"):
            previous_index = max(
                0,
                current_frame_index - 1
            )

            cv2.setTrackbarPos(
                "Frame",
                window_name,
                previous_index
            )

        # Next frame
        elif key == ord("d"):
            next_index = min(
                len(ball_tracking_stats) - 1,
                current_frame_index + 1
            )

            cv2.setTrackbarPos(
                "Frame",
                window_name,
                next_index
            )

    cv2.destroyAllWindows()

def shot_chart(
    COURT_POINTS_INPUT_FILE: Path,
    PREDICTIONS_INPUT_FILE: Path,
    BALL_TRACKING_CLASS_FILE: Path,
    VIDEO_FILE: Path,
    SPORT: str,
    bounce_angle_threshold: np.float64 = np.float64(40),
    player_angle_difference_threshold: np.float64 = np.float64(90),
    bounce_frame_cooldown: int = 5,
    debug: bool = False,
    graph_trajectory: bool = False,
    BOUNCE_DEBUG_PATH: Path = None
):

    cap = cv2.VideoCapture(VIDEO_FILE)

    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {VIDEO_FILE}")

    with open(COURT_POINTS_INPUT_FILE, "r") as f:
        print(f"Opening {COURT_POINTS_INPUT_FILE}")
        court_points = json.load(f)

    if SPORT == "pickleball":
        window_name = "Video and Pickleball Court"
        CURRENT_COURT_IMAGE = draw_pickleball_court()

        video_points = [
                coordinate
                for point_location, coordinate
                in court_points["image_points"].items()
            ]

    elif SPORT == "tennis":

        window_name = "Video and Tennis Court"
        CURRENT_COURT_IMAGE = draw_tennis_court()
        video_points = []

        first_frame_court_points = court_points.get("1", [])
        if not first_frame_court_points:

            print(f"Could not find court points for the first frame.")
            os._exit(1)

        for keypoint_detection_dict in first_frame_court_points:

            curr_x = keypoint_detection_dict['x']
            curr_y = keypoint_detection_dict['y']

            video_points.append((curr_x, curr_y))

    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, 960, 540)
    fps = cap.get(cv2.CAP_PROP_FPS)

    with open(PREDICTIONS_INPUT_FILE, "r") as f:
        print(f"Opening {PREDICTIONS_INPUT_FILE}")
        predictions_by_frame = json.load(f)
        predictions_by_frame = {int(keys): vals for keys, vals in predictions_by_frame.items()}

    with open(BALL_TRACKING_CLASS_FILE, "rb") as f:
        print(f"Opening {BALL_TRACKING_CLASS_FILE}")
        ball_tracker = pickle.load(f)
        for k, v in ball_tracker.tracker.items(): 
            if isinstance(k, str): 
                ball_tracker.tracker = {int(keys): vals for keys, vals in ball_tracker.items()}
            break

    video_points = np.array(video_points, dtype=np.float32)

    HOMOGRAPHY_MATRIX = compute_homography(
        video_points=video_points,
        sport=SPORT
    )

    frame_id = 1
    bounce_cooldown = 0
    bounces = []

    while True:

        success, video_frame = cap.read()

        if not success:
            break

        court_image = CURRENT_COURT_IMAGE.copy()

        current_ball_frame_stats = ball_tracker.tracker.get(frame_id)
        if not current_ball_frame_stats: 
            frame_id += 1
            continue

        ball_homography_location = current_ball_frame_stats.get("homography location")
        ball_vision_model_location = current_ball_frame_stats.get("vision model location")
        ball_angle = current_ball_frame_stats.get("angle")
        bounced = False
        player_locations = []

        if ball_angle:

            bounced, hit_by_player = validate_bounce(
                frame_id=frame_id,
                ball_tracker=ball_tracker,
                bounce_angle_threshold=bounce_angle_threshold,
                debug=debug,
                bounce_debugging_path=BOUNCE_DEBUG_PATH
            )

            if bounced:

                if bounce_cooldown == 0:

                    bounced = True

                    if ball_angle < player_angle_difference_threshold:
                        cv2.circle(
                            img=CURRENT_COURT_IMAGE,
                            center=ball_homography_location,
                            radius=5,
                            color=(0, 0, 0),
                            thickness=3
                        )

                    bounce_cooldown = bounce_frame_cooldown

            if bounce_cooldown > 0 and not bounced: 

                bounce_cooldown -= 1

            # plot live ball
            cv2.circle(
                img=court_image,
                center=ball_homography_location,
                radius=5,
                color=(255, 255, 0),
                thickness=3
            )

            for pred in predictions_by_frame[frame_id]:

                if pred['class'] == 'ball' or 'box' not in pred: continue

                coordinates, center = get_coordinates_and_center(prediction=pred)

                location = detection_of_court_points(
                    box=np.array(pred['box'], dtype=np.float32),
                    H=HOMOGRAPHY_MATRIX
                )

                player_locations.append((location, center))

                cv2.circle(
                    img=court_image,
                    center=location,
                    radius=1,
                    color=(0, 0, 0),
                    thickness=1
                )

                cv2.putText(
                    court_image,
                    pred['class'],
                    (
                        location[0],
                        max(location[1] - 10, 15)
                    ),
                    cv2.FONT_HERSHEY_COMPLEX,
                    0.6,
                    (0, 0, 0),
                    2
                )
            
            if hit_by_player:

                near_player, nearest_player_location = ball_near_player(
                    frame_id=frame_id, 
                    predictions_by_frame=predictions_by_frame, 
                    ball_center=ball_vision_model_location,
                    HOMOGRAPHY_MATRIX=HOMOGRAPHY_MATRIX
                )

                all_zeros = True

                if bounces:

                    start_frame, end_frame = bounces[-1], frame_id

                    for frame in range(start_frame + 1, end_frame):

                        if ball_tracker.tracker[frame]['angle'] >= 0.1:
                            all_zeros = False
                            break

                else:

                    all_zeros = False
                    
                if not all_zeros:    

                    cv2.circle(
                        img=CURRENT_COURT_IMAGE,
                        center=nearest_player_location,
                        radius=5,
                        color=(255, 255, 255),
                        thickness=3
                    )

                else:

                    bounced = False

        cv2.putText(
            court_image,
            f"Frame: {frame_id}",
            (10, 25),
            cv2.FONT_HERSHEY_COMPLEX,
            0.6,
            (0, 0, 0),
            2
        )

        # Resize court image to match the video frame's height.
        video_height, video_width = video_frame.shape[:2]
        PICKLEBALL_COURT_HEIGHT, PICKLEBALL_COURT_HEIGHT = court_image.shape[:2]
        PICKLEBALL_COURT_SCALE = video_height / PICKLEBALL_COURT_HEIGHT
        resized_PICKLEBALL_COURT_HEIGHT = int(PICKLEBALL_COURT_HEIGHT * PICKLEBALL_COURT_SCALE)

        court_frame = cv2.resize(
            court_image,
            (resized_PICKLEBALL_COURT_HEIGHT, video_height)
        )

        # Combine horizontally.
        side_by_side = cv2.hconcat([
            video_frame,
            court_frame
        ])

        cv2.imshow(
            window_name,
            side_by_side
        )

        if bounced:
            if graph_trajectory:
                plot_ball_trajectory(ball_dict=ball_tracker.tracker, frame_num=frame_id)

        delay = max(1, int(1000 / fps))
        key = cv2.waitKey(delay) & 0xFF

        if key == ord("q"):
            break

        frame_id += 1

    cv2.destroyAllWindows()


if __name__ == "__main__":

    from side_functions import run_predictions, get_court_points, get_coordinates_and_center

    video_filename = "8"

    COURT_POINTS_INPUT_FILE = os.path.join(BASE_DIR, "output", "court_points", f"{video_filename}_court_points.json")
    if not os.path.isfile(COURT_POINTS_INPUT_FILE):

        VIDEO_PATH = os.path.join(BASE_DIR, "input", f"{video_filename}.mp4")
        OUTPUT_PATH = COURT_POINTS_INPUT_FILE

        get_court_points(VIDEO_PATH=VIDEO_PATH, OUTPUT_PATH=OUTPUT_PATH)

    with open(COURT_POINTS_INPUT_FILE, "r") as f:
    
        video_points = json.load(f)

    court_points = []
    for point_location, coordinates in video_points['image_points'].items():

        court_points.append(coordinates)

    PREDICTIONS_INPUT_FILE = os.path.join(BASE_DIR, "output", "predictions", f"{video_filename}_predictions.txt")
    BALL_TRACKING_FILE = os.path.join(BASE_DIR, "output", "BallTracking", f"{video_filename}_ball_tracking.json")

    H = compute_homography(video_points=np.array(court_points, dtype=np.float32))
    ball_tracker = BallTracker(homography_matrix=H)

    run_predictions(COURT_POINTS_INPUT_FILE=COURT_POINTS_INPUT_FILE, PREDICTIONS_INPUT_FILE=PREDICTIONS_INPUT_FILE, BALL_TRACKING_OUTPUT_FILE=BALL_TRACKING_FILE, ball_tracker=ball_tracker)



    