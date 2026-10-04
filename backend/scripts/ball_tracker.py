import numpy as np
import cv2
import sys
import os
import json
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

TENNIS_COURT_LENGTH = float(os.getenv("TENNIS_COURT_LENGTH"))
TENNIS_COURT_WIDTH = float(os.getenv("TENNIS_COURT_WIDTH"))
TENNIS_COURT_SCALE = int(os.getenv("TENNIS_COURT_SCALE"))
TENNIS_COURT_PADDING = int(os.getenv("TENNIS_COURT_PADDING"))


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

class BallTracker:
 
    def __init__(
            self, 
            COURT_POINTS_FILE,
            max_consecutive_predictions=30, 
            max_displacement_px=300, 
            max_false_positive_count=15, 
            false_positives=None                                        
        ):
 
        with open(COURT_POINTS_FILE, "r") as file:
 
            court_points = json.load(file)
 
        self.court_points = {int(k): v for k, v in court_points.items()}
 
        self.COURT_POINTS_FILE = COURT_POINTS_FILE
 
        self.HOMOGRAPHY_MATRIX = compute_homography(COURT_POINTS_PATH=COURT_POINTS_FILE, frame_id=1)
        if len(self.HOMOGRAPHY_MATRIX) == 0:
 
            raise Exception(f"Could not find court points for first frame: CHECK {COURT_POINTS_FILE}")
 
        self.MAX_CONSECUTIVE_ESTIMATIONS = max_consecutive_predictions  # max number of estimations the tracker has before going back and checking the locations again
        self.MAX_CONSECUTIVE_ESTIMATIONS_PADDING = 30                   # when looking through the frames for an estimation error it goes back this any frames
        self.MAX_ESTIMATIONS_IN_REDO = 20                                # when creating the new ball tracker there can be this amount of estimations before we realize we just lost the ball
        self.MAX_DISPLACEMENT_PX = max_displacement_px                  # number of pixels between detection a ball can travel through frames
 
        self.tracker = {}
 
        self.no_location_found = None                                   # for the beginning when we have no detections
        self.consecutive_no_detections = 0                              
        self.consecutive_estimations = 0
        self.is_estimation = True
        self.detections_track = {}
        self.false_positives = set(false_positives) if false_positives else set()   
        self.MAX_FALSE_POSITIVE_COUNT = max_false_positive_count
        self.fixing_false_positives = False
 
        self.first_frame_where_ball_is_found = 1
        self.false_positives_count = 0
 
        self.real_frames = []
 
    def prev_real_frame(self, frame_id: int, n: int = 1):
 
        prev_frame_id = frame_id - 1
        while n > 0:
 
            while prev_frame_id in self.tracker and self.tracker[prev_frame_id].get('repeat_frame', False):
                prev_frame_id -= 1
 
            if prev_frame_id not in self.tracker: return None
 
            n -= 1
            if n > 0: prev_frame_id -= 1
 
        return prev_frame_id
 
    def estimate_ball_location(self, frame_id: int, predictions: list):
 
        prev_1 = self.prev_real_frame(frame_id, 1)                      
        prev_2 = self.prev_real_frame(frame_id, 2)                      
 
        # if it's the first frame then the ball isn't found yet
        if prev_1 is None:
 
            self.no_location_found = True
            chosen_ball_location = (-1, -1)
 
        # if it's the second frame we just use the last location
        elif prev_2 is None:
 
            chosen_ball_location = self.tracker[prev_1]['vision model location']
 
        # if frame number > 2 then we check if the ball's been found yet. if not then (-1, -1), else we use estimations from previous locations
        else:
 
            if self.tracker[prev_1]['vision model location'] == (-1, -1) or self.tracker[prev_2]['vision model location'] == (-1, -1):
 
                chosen_ball_location = (-1, -1)
 
            else:
 
                x1, y1 = self.tracker[prev_2]['vision model location']
                x2, y2 = self.tracker[prev_1]['vision model location']
 
                x_estimation = x2 + (x2 - x1)
                y_estimation = y2 + (y2 - y1)
 
                chosen_ball_location = (x_estimation, y_estimation)
 
        # if the ball hasn't been found yet we don't increment consecutive_estimations
        if chosen_ball_location != (-1, -1): self.consecutive_estimations += 1
        self.is_estimation = True
 
        if self.consecutive_estimations >= self.MAX_CONSECUTIVE_ESTIMATIONS:
 
            
            # The repeats in between are replayed as repeats so the temp tracker sees the same numbering.
            redo_real_frames = [f for f in self.real_frames if f < frame_id][-self.MAX_CONSECUTIVE_ESTIMATIONS:]
            start_frame = redo_real_frames[0]
 
            temp_ball_tracker = BallTracker(
                COURT_POINTS_FILE=self.COURT_POINTS_FILE,
                false_positives=self.false_positives,                   
            )
 
            for f in range(start_frame, frame_id):
 
                if self.tracker[f].get('repeat_frame', False):
                    temp_ball_tracker.update(frame_id=f, ball_locations=[], repeat_frame=True)
                else:
                    temp_ball_tracker.update(
                        frame_id=f, 
                        ball_locations=self.tracker[f]['ball locations'],
                        allow_estimation=False
                    )
 
            estimation_counter = sum(1 for f in redo_real_frames if temp_ball_tracker.tracker[f]['estimation'])
 
            if estimation_counter > self.MAX_ESTIMATIONS_IN_REDO:
 
                for f in redo_real_frames:
 
                    self.tracker[f] = {
                        'ball locations': temp_ball_tracker.tracker[f]['ball locations'],
                        'homography location': (-1, -1),
                        'vision model location': (-1, -1),
                        'estimation': True,
                        'ties': [],
                        'ball lost': True,
                        'repeat_frame': False                           
                    }
 
                self.consecutive_estimations = 0
 
            else:
 
                for f in redo_real_frames:
 
                    self.tracker[f] = temp_ball_tracker.tracker[f]
 
                
                # Keep only the estimations still trailing at the end of the redone window.
                self.consecutive_estimations = 0
                for f in reversed(redo_real_frames):
                    if not self.tracker[f]['estimation']: break
                    self.consecutive_estimations += 1
 
            if predictions:
 
               was_estimation = self.update(
                    frame_id=frame_id,
                    ball_locations=predictions,
                    allow_estimation=False
                )
 
               if not was_estimation:
                self.is_estimation = False
                chosen_ball_location = self.tracker[frame_id]['vision model location']
 
        return chosen_ball_location
 
    def compute_homographical_location(self, ball_location: tuple) -> tuple:
 
        ground_x, ground_y = ball_location
 
        video_point = np.array(
            [[[ground_x, ground_y]]],
            dtype=np.float32
        )
 
        court_point = cv2.perspectiveTransform(video_point, self.HOMOGRAPHY_MATRIX)
 
        court_x, court_y = court_point[0, 0]
 
        return int(court_x), int(court_y)
 
    def fix_no_detections(self, last_frame: int):
 
        
        i = self.real_frames.index(last_frame)
 
        if  (
                (i < 2) or 
                (self.tracker[self.real_frames[i - 1]]['vision model location'] == (-1, -1)) or 
                (self.tracker[last_frame]['vision model location'] == (-1, -1)) or
                (self.tracker[last_frame].get('ball lost', False))
            ): return
 
        self.first_frame_where_ball_is_found = last_frame
 
        cur_x, cur_y = self.tracker[last_frame]['vision model location']
        prev_x, prev_y = self.tracker[self.real_frames[i - 1]]['vision model location']
 
        x_change = cur_x - prev_x
        y_change = cur_y - prev_y
 
        k = i - 2
        while k >= 0 and self.tracker[self.real_frames[k]]['vision model location'] == (-1, -1):
 
            changing_index = self.real_frames[k]
            next_x, next_y = self.tracker[self.real_frames[k + 1]]['vision model location'] 
            self.tracker[changing_index]['vision model location'] = (next_x - x_change, next_y - y_change)
            self.tracker[changing_index]['homography location'] = self.compute_homographical_location(self.tracker[changing_index]['vision model location'])
 
            k -= 1
 
        self.no_location_found = False
 
    def no_ball_found(self, frame_id: int, ball_locations: list = []) -> tuple:
 
        self.consecutive_no_detections += 1
        return self.estimate_ball_location(frame_id=frame_id, predictions=ball_locations)
 
    def fix_false_positives(self):
 
        if self.fixing_false_positives: return
 
        self.fixing_false_positives = True
 
        try:
 
            frames = sorted(self.tracker.items())
 
            # and the old counts carry into the replay
            self.real_frames = []
            self.detections_track = {}
            self.consecutive_estimations = 0
            self.consecutive_no_detections = 0
            self.no_location_found = None
 
            for frame_id, frame_data in frames:
 
                if frame_data.get('repeat_frame', False):               
                    self.update(frame_id=frame_id, ball_locations=[], repeat_frame=True)
                    continue
                
                self.update(
                    frame_id=frame_id,
                    ball_locations=frame_data['ball locations']
                )
 
            self.false_positives_count += 1
 
        finally:
 
            self.fixing_false_positives = False
 
    def interpolate_estimations(self, frame_id):
 
        i = self.real_frames.index(frame_id)
        j = i - 1
        while j >= 0 and self.tracker[self.real_frames[j]]['estimation']:
            
            if self.tracker[self.real_frames[j]].get('ball lost', False): return
            j -= 1
 
        # no detection before the gap, or no gap at all
        if j < 0 or j == i - 1: return
 
        start_pos = self.tracker[self.real_frames[j]]['vision model location']
        end_pos = self.tracker[frame_id]['vision model location']
        steps = i - j
 
        # only real frames between the last detection and frame_id
        for k in range(j + 1, i):
 
            frame = self.real_frames[k]
            ratio = (k - j) / steps
 
            x = start_pos[0] + ratio * (end_pos[0] - start_pos[0])
            y = start_pos[1] + ratio * (end_pos[1] - start_pos[1])
 
            self.tracker[frame]['vision model location'] = (x, y)      
            self.tracker[frame]['homography location'] = self.compute_homographical_location((x, y))   
            self.tracker[frame]['interpolation'] = True
         
    # ball_locations is an array of ball locations from the model detection (not homographical)
    def update(
        self, 
        frame_id: int, 
        ball_locations: list, 
        allow_estimation: bool = True, 
        repeat_frame: bool = False
    ):
 
        frame_id = int(frame_id)
 
        if frame_id in self.court_points:
 
            self.HOMOGRAPHY_MATRIX = compute_homography(COURT_POINTS_PATH=self.COURT_POINTS_FILE, frame_id=frame_id)
 
        if repeat_frame:
 
            self.tracker[frame_id] = {'repeat_frame': True}  
            return self.is_estimation
 
        if not self.real_frames or self.real_frames[-1] != frame_id:
            self.real_frames.append(frame_id)
 
        self.is_estimation = False
        nearest_ball_location = {}
        prediction_indices = {}
 
        prev_frame_id = self.prev_real_frame(frame_id)                 
 
        if len(ball_locations) == 0:
 
            if allow_estimation:
                # if we don't detect a ball we estimate using a simple slope
                chosen_ball_location = self.no_ball_found(frame_id=frame_id)
            else:
                chosen_ball_location = (-1, -1)
                self.is_estimation = True                               
 
        elif len(ball_locations) == 1:
 
            """
            if there is only one ball location detected we check to see if it's plausible, else we estimate
            """
 
            # if the location found is a false positive
            if ball_locations[0] in self.false_positives:
 
                if allow_estimation:                                    
                    chosen_ball_location = self.no_ball_found(frame_id=frame_id)
                else:
                    chosen_ball_location = (-1, -1)
                    self.is_estimation = True
 
            else:
                
                self.consecutive_no_detections = 0
                # if we are on the first frame then just pick the first one
                if prev_frame_id is None:                               
 
                    chosen_ball_location = ball_locations[0]
 
                else:
 
                    prev_location = self.tracker[prev_frame_id]['vision model location']
 
                    curr_x, curr_y = ball_locations[0]
                    prev_x, prev_y = prev_location
 
                    if prev_x == -1 and prev_y == -1:
 
                        chosen_ball_location = curr_x, curr_y
 
                    # if the ball location is too far from the previous ball location then we rule it as not the ball
                    elif prev_x - self.MAX_DISPLACEMENT_PX > curr_x or curr_x > prev_x + self.MAX_DISPLACEMENT_PX:
                        chosen_ball_location = self.estimate_ball_location(frame_id=frame_id, predictions=ball_locations)
 
                    elif prev_y - self.MAX_DISPLACEMENT_PX > curr_y  or curr_y > prev_y + self.MAX_DISPLACEMENT_PX: 
                        chosen_ball_location = self.estimate_ball_location(frame_id=frame_id, predictions=ball_locations)
 
                    else:
 
                        chosen_ball_location = ball_locations[0]
 
        else:
            self.consecutive_no_detections = 0
 
            # rule out self positives
            valid_locations = [
                location
                for location in ball_locations
                if location not in self.false_positives
            ]
 
            if not valid_locations:
                if allow_estimation:
                    chosen_ball_location = self.no_ball_found(
                        frame_id=frame_id
                    )
                else:
                    chosen_ball_location = (-1, -1)
                    self.is_estimation = True
 
            elif prev_frame_id is None:                                 
                # There is no previous frame to compare against.
                chosen_ball_location = valid_locations[0]
 
            else:
                
                prev_x, prev_y = self.tracker[prev_frame_id]["vision model location"]   
 
                if (prev_x, prev_y) == (-1, -1):
                    chosen_ball_location = valid_locations[0]
                else:
                    chosen_ball_location = min(
                        valid_locations,
                        key=lambda location: (
                            abs(location[0] - prev_x)
                            + abs(location[1] - prev_y)
                        )
                    )
        
 
        # in case we have ties for detections that are n pixels away
        if len(prediction_indices) > 1: ties = nearest_ball_location
        else: ties = []
 
        if chosen_ball_location != (-1, -1):
            homography_location = self.compute_homographical_location(ball_location=chosen_ball_location)
        else: 
            homography_location = (-1, -1)
 
        # reset consecutive esimations if the curernt frame's ball location is not an estimation
        if self.is_estimation is False: self.consecutive_estimations = 0
 
        # track all detections in case we find a false positive
        self.detections_track[chosen_ball_location] = self.detections_track.get(chosen_ball_location, 0) + 1
 
        # if we found a false positive or a location that has been still for more than [self.MAX_FALSE_POSITIVE_COUNT] frames then we deem it as a false positive
        # also we don't count (-1, -1 as a false positive.)
        if (
            self.detections_track[chosen_ball_location] > self.MAX_FALSE_POSITIVE_COUNT and 
            not self.fixing_false_positives and 
            chosen_ball_location != (-1, -1)
        ):
 
            false_pos_x, false_pos_y = chosen_ball_location
            self.false_positives.update(
                [(false_pos_x, false_pos_y),
                (false_pos_x + 0.5, false_pos_y),
                (false_pos_x, false_pos_y + 0.5),
                (false_pos_x + 0.5, false_pos_y + 0.5),
                (false_pos_x - 0.5, false_pos_y),
                (false_pos_x, false_pos_y - 0.5),
                (false_pos_x - 0.5, false_pos_y - 0.5)]
            )
 
            
            # picked the false positive. Drop it and redo this frame after the replay.
            self.real_frames.pop()
            self.fix_false_positives()
            return self.update(frame_id=frame_id, ball_locations=ball_locations, allow_estimation=allow_estimation)
        
        # update self.tracker
        self.tracker[frame_id] = {
            'ball locations': ball_locations,                       # all predictions found
            'homography location': homography_location,             # homographic location of ball
            'vision model location': chosen_ball_location,          # vision model location of ball
            'estimation': self.is_estimation,                       # whether the chosen location is an estimation
            'ties': ties,                                           # if multiple detections are the same amount of pixels away we set ties and check future predictions to see which are most plausible
            'ball lost': False,
            'repeat_frame': False
        }  
 
        # we made it so if the starting frames don't detect a ball we mark it as (-1, -1) and now we want to fix it
        if self.no_location_found == True: self.fix_no_detections(last_frame=frame_id)
        if not self.tracker[frame_id]['estimation']: self.interpolate_estimations(frame_id=frame_id)
 
        return self.is_estimation
