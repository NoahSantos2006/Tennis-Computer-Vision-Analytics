import json
from pathlib import Path
import numpy as np
import cv2
import time
import sys
import os
import matplotlib.pyplot as plt
from scripts.ball_tracker import BallTracker

BASE_DIR = Path(__file__).parent.parent

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

def plot_ball_trajectory(
    ball_dict: dict, 
    frame_num: int, 
    window: int = 10):
    """
    Plot the ball's vision-model trajectory around a specific frame.

    Parameters
    ----------
    ball_dict : dict
        Dictionary keyed by frame number as strings.

    frame_num : int
        Frame you want to inspect.

    window : int
        Number of frames before and after frame_num to plot.
    """

    print(f"Plotting ball's trajectory...")

    start_frame = frame_num - window
    end_frame = frame_num + window

    xs = []
    ys = []
    frames = []
    estimations = []

    for frame in range(start_frame, end_frame + 1):

        # Your dictionary uses strings for frame keys
        frame_data = ball_dict.get(frame)

        if frame_data is None:
            continue

        location = frame_data.get("vision model location")

        # Skip frames without a location
        if location is None or len(location) < 2:
            continue

        x, y = location

        xs.append(x)
        ys.append(y)
        frames.append(frame)
        estimations.append(frame_data.get("estimation", False))

    if not frames:
        print(f"No ball locations found around frame {frame_num}")
        return

    # Plot trajectory
    plt.figure(figsize=(10, 8))

    plt.plot(xs, ys, marker="o")

    # Label every point with its frame number
    for x, y, frame, estimation in zip(
        xs, ys, frames, estimations
    ):
        label = str(frame)

        if estimation:
            label += " (est)"

        plt.annotate(
            label,
            (x, y),
            xytext=(5, 5),
            textcoords="offset points"
        )

    # Highlight the requested frame
    if frame_num in frames:
        index = frames.index(frame_num)

        plt.scatter(
            xs[index],
            ys[index],
            s=150,
            facecolors="none",
            edgecolors="black",
            linewidths=2
        )

    plt.xlabel("X position (pixels)")
    plt.ylabel("Y position (pixels)")
    plt.title(
        f"Ball trajectory: frames {start_frame}–{end_frame}"
    )

    # Image coordinates have Y increasing downward,
    # so this makes the graph visually match the video.
    plt.gca().invert_yaxis()

    plt.grid()
    plt.show()

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

def validate_bounce(
        frame_id: int, 
        ball_tracker: BallTracker, 
        bounce_angle_threshold: np.float64, 
        bounce_debugging_path: Path = None,
        bounce_window: int = 4,
        slowdown_velocity_threshold: int = 12,
        consecutive_non_degrees_threshold: int = 3,
        debug: bool = True,
        angle_threshold_with_sign_change: float = 20.0,
        speed_threshold: float = 2.0,
        bounce_angle_with_low_speed_threshold: float = 100.0,
        max_bounce_angle: float = 100.0,                        # if the bounce angle is higher than 100 degrees it's most likely hit by a player,
        minimum_velocity_change_threshold: float = 2.0,
        homographic_y_change_threshold: float = 20.0,
        homographic_window_disparity: int = 7
    ):

    def local_speed_minima(
        frame_id: int,
        ball_tracker: dict
    ):

        minimum = True
        start_frame, end_frame = max(2, frame_id - bounce_window), min(len(ball_tracker) - 1, frame_id + bounce_window)
        speed_window = []
        # if frame = 5 then we go through [2, 3, 4, 5, 6, 7, 8]
        for frame in range(start_frame + 1, end_frame):

            if (
                'speed' not in ball_tracker[frame] or
                'speed' not in ball_tracker[frame - 1]
            ): return False

            speed_window.append((float(np.round(ball_tracker[frame]['speed'], 2)), frame))

            if frame == start_frame: continue
            if speed_window:

                if frame <= frame_id and np.round(ball_tracker[frame]['speed'], 2) >= np.round(ball_tracker[frame - 1]['speed'], 2):

                    minimum = False
                    break

                # ex frame = 143, frame_id = 142 -> if 143's speed = 23 and 142's speed = 26.46 then minimum = False
                if frame > frame_id and np.round(ball_tracker[frame]['speed'], 2) <= np.round(ball_tracker[frame - 1]['speed'], 2):

                    minimum = False
                    break

        if minimum:
            return True

        return False

    previous_ball_frame_stats = ball_tracker.tracker.get(frame_id - 1, None)
    current_ball_frame_stats = ball_tracker.tracker.get(frame_id, None)
    next_ball_frame_stats = ball_tracker.tracker.get(frame_id + 1, None)
    next_next_ball_frame_stats = ball_tracker.tracker.get(frame_id + 2, None)

    if (
        (frame_id == len(ball_tracker.tracker)) or
        ('velocity' not in previous_ball_frame_stats) or
        ('velocity' not in current_ball_frame_stats) or
        ('velocity' not in next_ball_frame_stats) or 
        ('velocity' not in next_next_ball_frame_stats)
    ): 

        if debug:
            with open(bounce_debugging_path, "a") as f:
                f.write("\n\n")

        return False, False

    # get all data
    prev_location = previous_ball_frame_stats.get("vision model location")
    prev_bounce_velo = previous_ball_frame_stats.get("velocity")
    prev_homographic_location = previous_ball_frame_stats.get("homography location")

    cur_bounce_angle = current_ball_frame_stats.get("angle")
    cur_bounce_velo = current_ball_frame_stats.get('velocity')
    cur_location = current_ball_frame_stats.get("vision model location")
    cur_homographic_location = current_ball_frame_stats.get("homography location")
    cur_homographic_x, cur_homographic_y = cur_homographic_location

    next_bounce_angle = next_ball_frame_stats.get("angle")
    next_bounce_velo = next_ball_frame_stats.get('velocity')
    next_location = next_ball_frame_stats.get("vision model location")
    next_homographic_x, next_homographic_y = next_ball_frame_stats.get("homography location")

    next_next_bounce_angle = next_next_ball_frame_stats.get("angle")
    next_next_bounce_velo = next_next_ball_frame_stats.get("velocity")

    curr_frame, end_frame = max(frame_id - bounce_window, 1), min(frame_id + bounce_window, len(ball_tracker.tracker))

    start_homography_frame, end_homography_frame = max(frame_id - homographic_window_disparity, 1), min(frame_id + homographic_window_disparity, len(ball_tracker.tracker))
    start_frame_homography_x, start_frame_homography_y = ball_tracker.tracker.get(start_homography_frame, {}).get("homography location", None)
    end_frame_homography_x, end_frame_homography_y = ball_tracker.tracker.get(end_homography_frame, {}).get("homography location", None)

    next_to_end_y_change = abs(next_homographic_y - end_frame_homography_y)
    # if the homography position is bigger than it goes down the y-axis
    if start_frame_homography_y > cur_homographic_y:
        start_homographic_sign = True
    else:
        start_homographic_sign = False

    if next_homographic_y > end_frame_homography_y: # means the ball goes up the y-axis
        end_homographic_sign = True
    else:
        end_homographic_sign = False                # the ball is decreasing on the y-axis

    if (
        start_homographic_sign != end_homographic_sign and          # if the ball is increasing in y and then after the current frame it decreases in y then that means a hit
        next_to_end_y_change > homographic_y_change_threshold
    ):
        homographic_sign_change = True
    else:
        homographic_sign_change = False

    consecutive_non_degrees = 0

    # if the ball wasn't found and it's suddenly found, the tracker gets a spike in location which causes a faulty detection
    """
    example:
        Frame 96 ball is 0.00 degrees at velocity (5.5, -7.5)
        Frame 97 ball is 0.00 degrees at velocity (5.5, -7.5)
        Frame 98 ball is 0.00 degrees at velocity (5.5, -7.5)
        Frame 99 ball is 148.32 degrees at velocity (5.5, -7.5)
        The ball bounced on frame 99.
        Frame 100 ball is 0.00 degrees at velocity (-2.0, 25.0)
        Frame 101 ball is 0.00 degrees at velocity (-2.0, 25.0)
        Frame 102 ball is 0.00 degrees at velocity (-2.0, 25.0)
    """
    positive_velocities = []
    bounce_angles = []
    while curr_frame <= end_frame:

        temp_frame_stats = ball_tracker.tracker.get(curr_frame)
        temp_cur_angle = temp_frame_stats.get('angle', None)
        velocity = temp_frame_stats.get('velocity', None)

        # if velocity == None
        if not velocity:
            curr_frame += 1
            continue

        temp_vx, temp_vy = velocity

        if temp_cur_angle <= 0.1:
            consecutive_non_degrees += 1
        else: 
            consecutive_non_degrees = 0

        # if velo[0:3] < 0 and velo[3] > 0 and velo[4:7] < 0
        if temp_vy < 0: positive_velocities.append(False)
        else: positive_velocities.append(True)

        if curr_frame == frame_id:
            if sum(bounce_angles) == 0: zeros_before_frame = True
            else: zeros_before_frame = False
        else:
            bounce_angles.append(temp_cur_angle)

        curr_frame += 1

    next_consecutive_estimations = 0
    for frame in range(frame_id + 1, min(frame_id + 6, len(ball_tracker.tracker))):

        estimation = ball_tracker.tracker[frame].get('estimation')

        if estimation:
            next_consecutive_estimations += 1
        else:
            break

    angle_change = False
    sign_change = False
    slowdown = False
    
    x1, y1 = prev_location
    x3, y3 = next_location

    vx = (x3 - x1) / 2
    vy = (y3 - y1) / 2
    
    speed = np.hypot(vx, vy)

    cur_vx, cur_vy = cur_bounce_velo
    next_vx, next_vy = next_bounce_velo
    next_next_vx, next_next_vy = next_next_bounce_velo
    
    if debug:
        with open(bounce_debugging_path, "a") as f:
            f.write(f"curr_vy = {cur_vy:.3f} | next_vy = {next_vy:.3f} | homography location: ({cur_homographic_x}, {cur_homographic_y})\n\n")

    if cur_vy * next_vy < 0: sign_change = True
    if abs(next_next_vy - cur_vy) > slowdown_velocity_threshold: slowdown = True
    if cur_bounce_angle > bounce_angle_threshold: angle_change = True

    local_minima = local_speed_minima(frame_id=frame_id, ball_tracker=ball_tracker.tracker)

    if (
        sum(positive_velocities) == 1 and (next_vy > 0 or cur_vy > 0) or                                # if there's a noise with vy sign change
        sum(positive_velocities) == len(positive_velocities) - 1 and (next_vy < 0 or cur_vy < 0) or     # ^
        zeros_before_frame or                                                                           # if the angles leading up to frame_id is all 0    
        next_consecutive_estimations > 3 or                                                             # if the next 3 ball locations are estimations
        abs(next_vy - cur_vy) < minimum_velocity_change_threshold or                                    # minimum y velocity change threshold (for noise) 
        cur_bounce_angle == 90.0 or                                                                     # if something wrong with the frame 
        cur_location == next_location                                                                   # repeat frame
    ):

        bounced = False
        hit_by_player = False

        return bounced, hit_by_player

    # player hit detection
    if homographic_sign_change == True: 
        bounced = True
        hit_by_player = True

        return bounced, hit_by_player

    if (
        (speed > speed_threshold or cur_bounce_angle > bounce_angle_with_low_speed_threshold) and
        (angle_change or
        ((sign_change and angle_change > angle_threshold_with_sign_change) or slowdown) and 
        consecutive_non_degrees < consecutive_non_degrees_threshold)
    ): 

        bounced = True
        hit_by_player = False

        return bounced, hit_by_player
    
    bounced = False
    hit_by_player = False

    return bounced, hit_by_player

def find_angles(
        ball_tracker_class: BallTracker,
        predictions_dict: dict,
        DEBUG_PATH: Path,
        BOUNCE_DEBUG_PATH: Path = None,
        bounce_angle_threshold: np.float64 = np.float64(40), 
        angle_padding: int = 5,
        actual_bounces_path: Path = None,
        debug: bool = False,
        correct_frame_disparity: int = 5,
    ) -> tuple:

    actual_bounces = []

    if debug:

        if not os.path.isfile(actual_bounces_path):

            print(f"Could not find the training data for bounces.")
            os._exit(1)

        with open(actual_bounces_path, "r") as f:

            actual_bounces = set([int(line.strip()) for line in f])
            og_length = len(actual_bounces)

        with open(BOUNCE_DEBUG_PATH, "w") as f:
            pass
        
    ball_tracker = ball_tracker_class.tracker

    # cleaning dict since the keys turn into strings after json serialize
    for k, v in ball_tracker.items(): 
        if isinstance(k, str): 
            ball_tracker = {int(keys): vals for keys, vals in ball_tracker.items()}
        break

    frame_id = 1
    cur_pad = 0
    total_false_positives = 0
    false_positives = []
    true_positives = []

    # bounce detection
    while frame_id < len(ball_tracker):
        
        if (
            frame_id < 2 or
            ball_tracker[frame_id - 1]['vision model location'] == (-1, -1) or
            ball_tracker[frame_id]['vision model location'] == (-1, -1) or
            ball_tracker[frame_id + 1]['vision model location'] == (-1, -1)

        ):

            with open(DEBUG_PATH, "a") as file:

                file.write(f"Skipped frame {frame_id}\n")

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

        if debug:
            with open(BOUNCE_DEBUG_PATH, "a") as f:
                f.write(f"Frame {frame_id} ball is {angle_deg:.2f} degrees at velocity ({vx:.3f}, {vy:.3f}) with speed {speed:.3f} ")

        ball_tracker[frame_id]['angle'] = angle_deg
        ball_tracker[frame_id]['velocity'] = (vx, vy)
        ball_tracker[frame_id]['speed'] = speed

        bounced, hit_by_player = validate_bounce(
            frame_id=frame_id,
            ball_tracker=ball_tracker_class,
            bounce_angle_threshold=bounce_angle_threshold,
            debug=debug,
            bounce_debugging_path=BOUNCE_DEBUG_PATH
        )
        
        if bounced:

            false_positive = True
            if cur_pad == 0:

                # if the program finds a bounce +- correct_frame_disparity from an actual bounce then we still count it
                start_frame = max(1, frame_id - correct_frame_disparity)
                end_frame = min(len(ball_tracker), frame_id + correct_frame_disparity)
                same_bounce_frames = set()

                for frame in range(start_frame + 1, end_frame):

                    same_bounce_frames.add(frame)

                for frame in same_bounce_frames:

                    if frame in actual_bounces:
                        actual_bounces.discard(frame)
                        true_positives.append(frame_id)
                        false_positive = False
                        break

                # if all the angles from the last time it bounced to the current frame id is <= 0.1 then we count it as the same as the last bounce
                # i'll probably change this later so that if the ball > estimation threshold then we don't count it as a bounce and go next
                if false_positive:

                    all_zeros = False
                    if true_positives:

                        all_zeros = True
                        for frame in range(last_bounced + correct_frame_disparity, frame_id):
                            if ball_tracker[frame]['angle'] >= 0.1:
                                all_zeros = False
                                break

                    if not all_zeros:

                        # if not hit_by_player:
                        false_positives.append(frame_id)
                        total_false_positives += 1

                last_bounced = frame_id
                cur_pad = angle_padding

        if cur_pad > 0: cur_pad -= 1

        frame_id += 1

    ball_tracker_class.tracker = ball_tracker

    if actual_bounces_path:

        score = og_length - len(actual_bounces)
        actual_bounces = sorted(list(actual_bounces))
        print(f"\nWe detected {score} / {og_length} bounces\n"
              f"{len(true_positives)} true positives -> {true_positives}\n"
              f"{total_false_positives} false positives -> {false_positives}\n"
              f"{len(actual_bounces)} false negatives -> {actual_bounces}")

    if debug:
        return true_positives, false_positives, actual_bounces, ball_tracker_class

    return ball_tracker_class

def run_predictions(
        COURT_POINTS_INPUT_FILE: Path, 
        PREDICTIONS_INPUT_FILE: Path, 
        BALL_TRACKING_OUTPUT_FILE: Path, 
        ball_tracker: BallTracker,
        DEBUG_PATH: Path,
        BOUNCE_DEBUG_PATH: Path,
        debug: bool = False
    ) -> BallTracker:
        
    with open(PREDICTIONS_INPUT_FILE, "r") as f:

        print(f"Opening {PREDICTIONS_INPUT_FILE}")
        predictions_by_frame = json.load(f)
        total_frames = len(predictions_by_frame)
        

    number_of_repeat_frames = 0

    for frame_id in range(1, total_frames + 1):

        predictions_array = predictions_by_frame.get(str(frame_id))

        current_ball_locations = []
    
        for pred in predictions_array:

            if pred['class'] == 'ball' and pred['confidence'] >= 0.5:

                coordinates, center = get_coordinates_and_center(prediction=pred)
                current_ball_locations.append(center)

        ball_tracker.update(
            frame_id=frame_id, 
            ball_locations=current_ball_locations
        )

        if debug:

            if frame_id > 1:

                previous_location = ball_tracker.tracker[frame_id - 1].get("vision model location", None)
                current_location = ball_tracker.tracker[frame_id].get("vision model location", None)

                if previous_location and current_location:

                    if previous_location == current_location and current_location not in ball_tracker.false_positives:

                        number_of_repeat_frames += 1
    if debug:
        print(f"We found {number_of_repeat_frames} repeat frames")

    return find_angles(
            ball_tracker_class=ball_tracker, 
            predictions_dict=predictions_by_frame, 
            BOUNCE_DEBUG_PATH=BOUNCE_DEBUG_PATH,
            DEBUG_PATH=DEBUG_PATH, 
            debug=debug
        )

def find_velocities(ball_track: dict, fps: int = 30) -> dict:

    frame_id = 1
    velocity = {}

    # velocities calcaulted in pixels/second

    while frame_id < len(ball_track):

        x, y = ball_track[frame_id]['vision model location']

        if frame_id == 1:

            vel_x, vel_y = 0, 0

            velocity[frame_id] = (vel_x, vel_y)
        
        else:

            prev_x, prev_y = ball_track[frame_id - 1]['vision model location']
 
            vel_x = (x - prev_x) / (1 / fps)
            vel_y = (y - prev_y) / (1 / fps) 
            
            velocity[frame_id] = (vel_x, vel_y)

        frame_id += 1

    return velocity

COURT_HEIGHT = 44
COURT_WIDTH = 20
PICKLEBALL_COURT_SCALE = 20
PICKLEBALL_COURT_PADDING = 50

TENNIS_COURT_LENGTH = 23.77
TENNIS_COURT_WIDTH = 10.97
TENNIS_COURT_SCALE = 25
TENNIS_COURT_PADDING = 30


def compute_homography(COURT_POINTS_PATH: Path, SPORT: str) -> np.array:

    court_points = []

    with open(COURT_POINTS_PATH, "r") as f:
        
        video_points = json.load(f)

    if SPORT == "pickleball":

        top_down_points = np.array([
            [PICKLEBALL_COURT_PADDING, PICKLEBALL_COURT_PADDING + 44 * PICKLEBALL_COURT_SCALE],                # near left baseline          
            [PICKLEBALL_COURT_PADDING + 20 * PICKLEBALL_COURT_SCALE, PICKLEBALL_COURT_PADDING + 44 * PICKLEBALL_COURT_SCALE],   # near right baseline
            [PICKLEBALL_COURT_PADDING + 20 * PICKLEBALL_COURT_SCALE, PICKLEBALL_COURT_PADDING],                # far right baseline
            [PICKLEBALL_COURT_PADDING, PICKLEBALL_COURT_PADDING],                             # far left baseline
            [PICKLEBALL_COURT_PADDING, PICKLEBALL_COURT_PADDING + 29 * PICKLEBALL_COURT_SCALE],                # near left kitchen
            [PICKLEBALL_COURT_PADDING + 20 * PICKLEBALL_COURT_SCALE, PICKLEBALL_COURT_PADDING + 29 * PICKLEBALL_COURT_SCALE],   # near right kitchen
            [PICKLEBALL_COURT_PADDING, PICKLEBALL_COURT_PADDING + 15 * PICKLEBALL_COURT_SCALE],                # far right kitchen
            [PICKLEBALL_COURT_PADDING + 20 * PICKLEBALL_COURT_SCALE, PICKLEBALL_COURT_PADDING + 15 * PICKLEBALL_COURT_SCALE]],  # far left kitchen
            dtype=np.float32
        )

        for point_location, coordinates in video_points['image_points'].items():
        
            court_points.append(coordinates)

    elif SPORT == "tennis":

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

        first_frame_court_points = video_points.get("1", [])
        if not first_frame_court_points:

            print(f"Could not find court points for the first frame.")
            os._exit(1)

        for keypoint_detection_dict in first_frame_court_points:
        
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

def get_court_points(VIDEO_PATH: Path, OUTPUT_PATH: Path):

    # Click these landmarks in this exact order.
    POINT_NAMES = [
        "near_left_baseline",
        "near_right_baseline",
        "far_right_baseline",
        "far_left_baseline",
        "near_left_kitchen",
        "near_right_kitchen",
        "far_right_kitchen",
        "far_left_kitchen",
    ]

    clicked_points: list[list[int]] = []
    display_frame = None
    original_frame = None


    def mouse_callback(event, x, y, flags, param):
        
        global display_frame

        if event != cv2.EVENT_LBUTTONDOWN:
            return

        if len(clicked_points) >= len(POINT_NAMES):
            return

        clicked_points.append([x, y])

        point_number = len(clicked_points)
        point_name = POINT_NAMES[point_number - 1]

        print(f"{point_number}. {point_name}: ({x}, {y})")

        display_frame = original_frame.copy()

        for index, point in enumerate(clicked_points):
            px, py = point

            cv2.circle(
                display_frame,
                (px, py),
                6,
                (0, 255, 0),
                -1,
            )

            cv2.putText(
                display_frame,
                str(index + 1),
                (px + 8, py - 8),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 0),
                2,
            )

    def get_frame(video_path: str, frame_id: int = 0):

        capture = cv2.VideoCapture(video_path)

        if not capture.isOpened():
            raise RuntimeError(f"Could not open video: {video_path}")

        capture.set(cv2.CAP_PROP_POS_FRAMES, frame_id)
        success, frame = capture.read()
        capture.release()

        if not success:
            raise RuntimeError(f"Could not read frame {frame_id}")

        return frame

    def save_points():
        data = {
            "video": VIDEO_PATH,
            "point_order": POINT_NAMES,
            "image_points": {
                name: point
                for name, point in zip(POINT_NAMES, clicked_points)
            },
        }

        Path(OUTPUT_PATH).write_text(
            json.dumps(data, indent=2),
            encoding="utf-8",
        )

        print(f"Saved court points to {OUTPUT_PATH}")

    original_frame = get_frame(VIDEO_PATH, frame_id=0)
    display_frame = original_frame.copy()

    window_name = "Click court points"

    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.setMouseCallback(window_name, mouse_callback)

    print("Click the court points in this order:")

    for index, name in enumerate(POINT_NAMES, start=1):
        print(f"{index}. {name}")

    print("\nControls:")
    print("S = save")
    print("U = undo last point")
    print("R = reset all points")
    print("Q = quit")

    while True:
        frame_to_show = display_frame.copy()

        next_index = len(clicked_points)

        if next_index < len(POINT_NAMES):
            instruction = f"Click: {POINT_NAMES[next_index]}"
        else:
            instruction = "All points selected. Press S to save."

        cv2.putText(
            frame_to_show,
            instruction,
            (20, 35),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 255),
            2,
        )

        cv2.imshow(window_name, frame_to_show)

        key = cv2.waitKey(20) & 0xFF

        if key == ord("q"):
            break

        elif key == ord("u"):
            if clicked_points:
                removed = clicked_points.pop()
                print(f"Removed point: {removed}")

                display_frame = original_frame.copy()

                for index, point in enumerate(clicked_points):
                    px, py = point
                    cv2.circle(display_frame, (px, py), 6, (0, 255, 0), -1)
                    cv2.putText(
                        display_frame,
                        str(index + 1),
                        (px + 8, py - 8),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6,
                        (0, 255, 0),
                        2,
                    )

        elif key == ord("r"):
            clicked_points.clear()
            display_frame = original_frame.copy()
            print("All points reset.")

        elif key == ord("s"):
            if len(clicked_points) != len(POINT_NAMES):
                print(
                    f"Select all {len(POINT_NAMES)} points before saving. "
                    f"Currently selected: {len(clicked_points)}"
                )
            else:
                save_points()
                break

    cv2.destroyAllWindows()

if __name__ == "__main__":

    from ball_tracker import BallTracker
