import numpy as np
import cv2
import sys
import os

class BallTracker:

    def __init__(
            self, 
            homography_matrix,
            max_consecutive_predictions=30, 
            max_displacement_px=300, 
            max_false_positive_count=15, 
            bounce_angle_threshold=np.float64(40), 
            false_positives=set()
        ):

        self.HOMOGRAPHY_MATRIX = homography_matrix                      # homography matrix
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
        self.false_positives = false_positives
        self.MAX_FALSE_POSITIVE_COUNT = max_false_positive_count
        self.fixing_false_positives = False

        self.first_frame_where_ball_is_found = 1
        self.false_positives_count = 0

    def estimate_ball_location(self, frame_id: int, predictions: list):

        # if it's the first frame then the ball isn't found yet
        if frame_id == 1:

            self.no_location_found = True
            chosen_ball_location = (-1, -1)

        # if it's the second frame we just use the last location
        elif frame_id == 2:

            chosen_ball_location = self.tracker[frame_id - 1]['vision model location']

        # if frame number > 2 then we check if the ball's been found yet. if not then (-1, -1), else we use estimations from previous locations
        elif frame_id > 2:

            if self.tracker[frame_id - 1]['vision model location'] == (-1, -1) or self.tracker[frame_id - 2]['vision model location'] == (-1, -1):

                chosen_ball_location = (-1, -1)

            else:

                x1, y1 = self.tracker[frame_id - 2]['vision model location']
                x2, y2 = self.tracker[frame_id - 1]['vision model location']

                x_estimation = x2 + (x2 - x1)
                y_estimation = y2 + (y2 - y1)

                chosen_ball_location = (x_estimation, y_estimation)

        # if the ball hasn't been found yet we don't increment consecutive_estimations
        if chosen_ball_location != (-1, -1): self.consecutive_estimations += 1
        self.is_estimation = True

        if self.consecutive_estimations >= self.MAX_CONSECUTIVE_ESTIMATIONS:

            # start frame is our current frame number - the max estimations so ex: (Frame 650 - 30 = 620)
            start_frame, i = frame_id - self.MAX_CONSECUTIVE_ESTIMATIONS, 1
            temp_ball_tracker = BallTracker(
                homography_matrix=self.HOMOGRAPHY_MATRIX, 
                false_positives=self.false_positives
            )


            # print(f"Creating a new ball tracker for frames {start_frame} - {frame_id}")
            while start_frame < frame_id:

                temp_ball_tracker.update(
                    frame_id=i, 
                    ball_locations=self.tracker[start_frame]['ball locations'],
                    allow_estimation=False
                )
                start_frame += 1
                i += 1

            estimation_counter = 0
            for _, frame_data in temp_ball_tracker.tracker.items():

                if frame_data['estimation']:

                    estimation_counter += 1

            if estimation_counter > self.MAX_ESTIMATIONS_IN_REDO:

                # print(f"Lost the ball. There were a total of {estimation_counter}/{len(temp_ball_tracker.tracker)} estimations")
                start_frame -= (i - 1)
                i = 1
                while start_frame < frame_id:

                    self.tracker[start_frame] = {
                        'ball locations': temp_ball_tracker.tracker[i]['ball locations'],
                        'homography location': (-1, -1),
                        'vision model location': (-1, -1),
                        'estimation': True,
                        'ties': [],
                        'ball lost': True
                    }
                    start_frame += 1
                    i += 1
                    self.consecutive_estimations = 0

            else:

                start_frame -= (i - 1)
                i = 1
                while start_frame < frame_id:

                    self.tracker[start_frame] = temp_ball_tracker.tracker[i]
                    start_frame += 1
                    i += 1

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

        if  (
                (last_frame < 3) or 
                (self.tracker[last_frame - 1]['vision model location'] == (-1, -1)) or 
                (self.tracker[last_frame]['vision model location'] == (-1, -1)) or
                (self.tracker[last_frame]['ball lost'])
            ): return

        self.first_frame_where_ball_is_found = last_frame

        cur_x, cur_y = self.tracker[last_frame]['vision model location']
        prev_x, prev_y = self.tracker[last_frame - 1]['vision model location']

        x_change = cur_x - prev_x
        y_change = cur_y - prev_y

        changing_index = last_frame - 2
        while changing_index in self.tracker and self.tracker[changing_index]['vision model location'] == (-1, -1):

            next_x, next_y = self.tracker[changing_index + 1]['vision model location'] 
            self.tracker[changing_index]['vision model location'] = (next_x - x_change, next_y - y_change)
            self.tracker[changing_index]['homography location'] = self.compute_homographical_location(self.tracker[changing_index]['vision model location'])

            changing_index -= 1

        self.no_location_found = False

    def no_ball_found(self, frame_id: int, ball_locations: list = []) -> tuple:

        self.consecutive_no_detections += 1
        return self.estimate_ball_location(frame_id=frame_id, predictions=ball_locations)

    def fix_false_positives(self):

        if self.fixing_false_positives: return

        self.fixing_false_positives = True

        try:

            frames = list(self.tracker.items())

            for frame_id, frame_data in frames:
                
                self.update(
                    frame_id=frame_id,
                    ball_locations=frame_data['ball locations']
                )

            self.false_positives_count += 1

        finally:

            self.fixing_false_positives = False

    def interpolate_estimations(self, frame_id):

        start_frame = frame_id - 1
        while start_frame > 0 and self.tracker[start_frame]['estimation']:

            start_frame -= 1

        if start_frame <= 0 or start_frame == frame_id - 1: return

        frame_gap = frame_id - start_frame

        start_pos = self.tracker[start_frame]['vision model location']
        end_pos = self.tracker[frame_id]['vision model location']

        for frame in range(start_frame + 1, frame_id):

            ratio = (frame - start_frame) / frame_gap

            x = start_pos[0] + ratio * (end_pos[0] - start_pos[0])
            y = start_pos[1] + ratio * (end_pos[1] - start_pos[1])

            self.tracker[frame]['vision model location'] = [x, y]
            self.tracker[frame]['interpolation'] = True
         
    # ball_locations is an array of ball locations from the model detection (not homographical)
    def update(self, frame_id: int, ball_locations: list, allow_estimation=True):

        self.is_estimation = False
        nearest_ball_location = {}
        prediction_indices = {}

        if len(ball_locations) == 0:

            if allow_estimation:
                # if we don't detect a ball we estimate using a simple slope
                chosen_ball_location = self.no_ball_found(frame_id=frame_id)
            else:
                chosen_ball_location = (-1, -1)

        elif len(ball_locations) == 1:

            """
            if there is only one ball location detected we check to see if it's plausible, else we estimate
            """

            # if the location found is a false positive
            if ball_locations[0] in self.false_positives:

                chosen_ball_location = self.no_ball_found(frame_id=frame_id)

            else:
                
                self.consecutive_no_detections = 0
                # if we are on the first frame then just pick the first one
                if frame_id <= 1:

                    chosen_ball_location = ball_locations[0]

                else:

                    prev_location = self.tracker[frame_id - 1]['vision model location']

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

            """
            simple tracker for multiple ball predictions: chose closest from last frame

            create a dictionary for each ball location and compare

            use px

            ex 
                prev ball location: (100, 100)

                preds = [(102, 103), (101, 105)]

                pred[0] = |102 - 100| + |103 - 100| = 2 + 3 = 5px away
                pred[1] = |101 - 100| + |105 - 100| = 1 + 5 = 6px away

                we keep a dictionary
            """

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

            elif frame_id <= 1 or frame_id - 1 not in self.tracker:
                # There is no previous frame to compare against.
                chosen_ball_location = valid_locations[0]

            else:
                
                prev_x, prev_y = self.tracker[
                    frame_id - 1
                ]["vision model location"]

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


            self.fix_false_positives()
        
        # update self.tracker
        self.tracker[int(frame_id)] = {
            'ball locations': ball_locations,                       # all predictions found
            'homography location': homography_location,             # homographic location of ball
            'vision model location': chosen_ball_location,          # vision model location of ball
            'estimation': self.is_estimation,                       # whether the chosen location is an estimation
            'ties': ties,                                           # if multiple detections are the same amount of pixels away we set ties and check future predictions to see which are most plausible
            'ball lost': False
        }  

        # we made it so if the starting frames don't detect a ball we mark it as (-1, -1) and now we want to fix it
        if self.no_location_found == True: self.fix_no_detections(last_frame=frame_id)
        if not self.tracker[frame_id]['estimation']: self.interpolate_estimations(frame_id=frame_id)

        # print(f"------------------")

        return self.is_estimation

