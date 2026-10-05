import cv2
import json
import json
import numpy as np
from pathlib import Path
import pickle
from concurrent.futures import ThreadPoolExecutor, wait, FIRST_COMPLETED

import xgboost as xgb

from dotenv import load_dotenv
import os
import time, random

import threading

from inference_sdk import InferenceHTTPClient
from inference_sdk.http.errors import HTTPCallErrorError, HTTPClientError

from backend.scripts.side_functions import run_predictions, get_bounces
from backend.scripts.ball_tracker import BallTracker
from backend.scripts.status import update_status

load_dotenv()

import json

MAX_WORKERS = int(os.getenv("MAX_WORKERS"))
<<<<<<< HEAD

=======
WORKSPACE_NAME = os.getenv("WORKSPACE_NAME")
WORKFLOW_ID = os.getenv("WORKFLOW_ID")
WORKFLOW_ID_WITH_COURT_POINTS = os.getenv("WORKFLOW_ID_WITH_COURT_POINTS")
MAX_ATTEMPTS = int(os.getenv("MAX_ATTEMPTS"))

def is_repeat(frame, prev):
    a = cv2.cvtColor(cv2.resize(frame, (480, 270)), cv2.COLOR_BGR2GRAY)
    b = cv2.cvtColor(cv2.resize(prev, (480, 270)), cv2.COLOR_BGR2GRAY)
    return np.count_nonzero(cv2.absdiff(a, b) > 20) < 20

>>>>>>> ca4022552e9814a1150864c076423b9c8b17c841
def predict(
        video_path: Path,
        OUTPUT_DIR: Path,
        MODEL_PATH: Path, 
        STATUS_PATH: Path,
        api_key: str,
        MAX_WORKERS: int = MAX_WORKERS,
        MEAN_DIFF_THRESHOLD: float = 0.1
    ) -> dict:
    
    parts = video_path.parts

    VIDEO_FILENAME = parts[-1].split(".")[0]

    # example: input/make/dunk/make4.mp4
    INPUT_VIDEO = str(video_path)

    cap = cv2.VideoCapture(INPUT_VIDEO)
    if not cap.isOpened():
        print(f"Could not open video {INPUT_VIDEO}")
        os._exit(1)
    
    fps = cap.get(cv2.CAP_PROP_FPS)

    total_frames = cap.get(cv2.CAP_PROP_FRAME_COUNT)

    client = InferenceHTTPClient.init(
        api_url="https://serverless.roboflow.com",
        api_key=api_key
    )

    court_detection_points_by_frame = {}
    predictions_by_frame = {}

    def predict_frame(
        frame_id: int, 
        frame: np.array, 
    ) -> tuple:

        if frame_id == 1 or frame_id % 50 == 0:
            
            for attempt in range(MAX_ATTEMPTS):

<<<<<<< HEAD
                data = client.run_workflow(
                    workflow_id=f"tennis-object-detection-with-court-points",
                    workspace_name="noahs-workspace-kg24g",
                    images={"image": frame},
                )[0]
=======
                try:
>>>>>>> ca4022552e9814a1150864c076423b9c8b17c841

                    data = client.run_workflow(
                        workflow_id=WORKFLOW_ID_WITH_COURT_POINTS,
                        workspace_name=WORKSPACE_NAME,
                        images={"image": frame},
                    )[0]
                    break

                except HTTPCallErrorError as e:

                    delay = min(2 ** attempt, 30) + random.random()

                    print(
                        f" Roboflow request failed. "
                        f"Retrying in {delay}s (on attempt {attempt}) "
                    )
                    time.sleep(delay)

                except HTTPClientError as e:

                    delay = min(2 ** attempt, 30) + random.random()
                    
                    print(
                        f" Too many requests. "
                        f"Retrying in {delay}s (on attempt {attempt}) "
                    )
                    time.sleep(delay)

        else:

<<<<<<< HEAD
            try:

                data = client.run_workflow(
                    workflow_id=f"tennis-object-detection",
                    workspace_name="noahs-workspace-kg24g",
                    images={"image": frame},
                )[0]

            except HTTPCallErrorError as e:
=======
            for attempt in range(MAX_ATTEMPTS):
>>>>>>> ca4022552e9814a1150864c076423b9c8b17c841
            
                try:

                    data = client.run_workflow(
                        workflow_id=WORKFLOW_ID,
                        workspace_name=WORKSPACE_NAME,
                        images={"image": frame},
                    )[0]
                    break

                except HTTPCallErrorError as e:

                    delay = min(2 ** attempt, 30) + random.random()

                    print(
                        f" Roboflow request failed. "
                        f"Retrying in {delay}s (on attempt {attempt}) "
                    )
                    time.sleep(delay)

                except HTTPClientError as e:

                    delay = min(2 ** attempt, 30) + random.random()
                    
                    print(
                        f" Too many requests. "
                        f"Retrying in {delay}s (on attempt {attempt}) "
                    )
                    time.sleep(delay)
        
        if not data:

            return frame_id, None

        result = data.get("predictions", {}).get("predictions", [])

        court_detection_data = data.get("court_detection_predictions", {})
        if court_detection_data:

            if court_detection_data.get("predictions", []):

                court_detection_points = court_detection_data.get('predictions', [])[0].get("keypoints", [])
                court_detection_points_by_frame[frame_id] = court_detection_points      

        return frame_id, result

    MAX_PENDING = MAX_WORKERS * 3

    # bounded concurrency
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:

        pending = set()
        frame_id = 1
        completed_frames = 0
        video_finished = False
        previous_frame = None
        repeat_frames = set()
<<<<<<< HEAD
=======
        failed_frames = []
>>>>>>> ca4022552e9814a1150864c076423b9c8b17c841

        while pending or not video_finished:

            while len(pending) < MAX_PENDING and not video_finished:

                ret, frame = cap.read()
                if not ret:
                    video_finished = True
                    break

                if previous_frame is not None:
<<<<<<< HEAD
                
                    diff = cv2.absdiff(previous_frame, frame)
                    diff_mean = np.mean(diff)
        
                    if diff_mean < MEAN_DIFF_THRESHOLD:
                        repeat_frames.add(frame_id)
=======

                    if is_repeat(frame=frame, prev=previous_frame):

                        repeat_frames.add(frame_id)
                        frame_id += 1
>>>>>>> ca4022552e9814a1150864c076423b9c8b17c841
                        continue

                future = executor.submit(
                    predict_frame,
                    frame_id=frame_id,
                    frame=frame.copy(),
                )

                pending.add(future)
                previous_frame = frame
                frame_id += 1

            if not pending:
                break

            # wait until one frame finishes
            done, pending = wait(
                pending,
                return_when=FIRST_COMPLETED
            )

            for future in done:

                result_frame_id, preds = future.result()

                if preds is None:

                    failed_frames.append(result_frame_id)
                    preds = []

                predictions_by_frame[result_frame_id] = preds

                completed_frames += 1

                update_status(
                    status_file=STATUS_PATH,
                    status="in progress",
                    stage="Processing Frames",
                    current_frame=completed_frames + len(repeat_frames),
                    total_frames=total_frames
                )

            for frame in repeat_frames:

                predictions_by_frame[frame] = "Repeat Frame"

    cap.release()

    PREDICTIONS_DIRECTORY = OUTPUT_DIR / "predictions"
    if not os.path.isdir(PREDICTIONS_DIRECTORY):
        os.makedirs(PREDICTIONS_DIRECTORY, exist_ok=True)
    predictions_text_path = f"{OUTPUT_DIR}/predictions/{VIDEO_FILENAME}_predictions.txt"
    with open(predictions_text_path, "w") as f:

        json.dump(predictions_by_frame, f)

    COURT_POINTS_DIRECTORY = OUTPUT_DIR / "court_points"
    if not os.path.isdir(COURT_POINTS_DIRECTORY):
        os.makedirs(COURT_POINTS_DIRECTORY, exist_ok=True)
    court_points_path = f"{OUTPUT_DIR}/court_points/{VIDEO_FILENAME}_court_points.json"    
    with open(court_points_path, "w") as f:

        json.dump(court_detection_points_by_frame, f)

    BALL_TRACKER_CLASS_FILE = os.path.join(OUTPUT_DIR, "BallTracking", VIDEO_FILENAME, f"{VIDEO_FILENAME}_ball_tracking_class.pkl")
    BALL_TRACKER_FILE = os.path.join(OUTPUT_DIR, "BallTracking", VIDEO_FILENAME, f"{VIDEO_FILENAME}_ball_tracking.json")

    predictions_by_frame, ball_tracker = run_predictions(
        COURT_POINTS_INPUT_FILE=court_points_path, 
        PREDICTIONS_INPUT_FILE=predictions_text_path, 
        ball_tracker=BallTracker(COURT_POINTS_FILE=court_points_path)
    )

    with open(BALL_TRACKER_CLASS_FILE, "wb") as f:

        pickle.dump(ball_tracker, f)

    with open(BALL_TRACKER_FILE, "w") as f:

        json.dump(ball_tracker.tracker, f)

    with open(BALL_TRACKER_CLASS_FILE, "rb") as f:

        ball_tracker = pickle.load(f)

    XGBoost_model = xgb.XGBClassifier()
    XGBoost_model.load_model(MODEL_PATH)

    bounce_detection_dict = get_bounces(
        VIDEO_FILENAME = VIDEO_FILENAME,
        OUTPUT_PATH = OUTPUT_DIR,
        MODEL = XGBoost_model,
        BALL_TRACKER_PREDICTIONS = ball_tracker.tracker
    )

    update_status(
        status_file=STATUS_PATH,
        status="finished",
        stage="",
        current_frame=total_frames,
        total_frames=total_frames
    )

    return bounce_detection_dict, fps

<<<<<<< HEAD
def validate_video(
    INPUT_PATH: Path,
    VIDEO_FILENAME: str,
    ALREADY_VALIDATED_PATH: Path,
    STATUS_PATH: Path,
    mean_diff_threshold: float = 0.1,
):

    VIDEO_PATH = os.path.join(INPUT_PATH, f"{VIDEO_FILENAME}.mp4")
    VALIDATED_VIDEOS_DIRECTORY = os.path.join(INPUT_PATH, 'validated_videos')
    if not os.path.isdir(VALIDATED_VIDEOS_DIRECTORY):
        os.makedirs(VALIDATED_VIDEOS_DIRECTORY, exist_ok=True)
    VALIDATED_VIDEO_PATH = os.path.join(VALIDATED_VIDEOS_DIRECTORY, f"{VIDEO_FILENAME}.mp4")

    cap = cv2.VideoCapture(VIDEO_PATH)
    total_frames = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    previous_frame = None
    repeat_frames = 0
    frame_id = 1

    all_repeat_frames = set()

    out = cv2.VideoWriter(
        filename=VALIDATED_VIDEO_PATH,
        fourcc=cv2.VideoWriter_fourcc(*"mp4v"),
        fps=fps,
        frameSize=(width, height)
    )

    while True:

        ret, frame = cap.read()
        if not ret:
            break

        if previous_frame is not None:

            diff = cv2.absdiff(previous_frame, frame)
            diff_mean = np.mean(diff)

            if diff_mean < mean_diff_threshold:
                all_repeat_frames.add(frame_id)
                repeat_frames += 1
            else:
                out.write(frame)

        if repeat_frames > 0.25 * total_frames:
        
            print(f"Video is not valid for uploading. Try Again.")
            cap.release()
            return 1

        previous_frame = frame.copy()

        update_status(
            status_file=STATUS_PATH,
            status="in progress",
            stage="Validating Video",
            current_frame=frame_id,
            total_frames=total_frames
        )

        frame_id += 1

    cap.release()
    out.release()

    with open(ALREADY_VALIDATED_PATH, "a") as f:
        f.write(f"{VIDEO_FILENAME}\n") 
    
    return 0           

=======
>>>>>>> ca4022552e9814a1150864c076423b9c8b17c841
def analyze_video(
    VIDEO_FILENAME: str,
    INPUT_PATH: Path,
    OUTPUT_PATH: Path,
    JOB_ID: str,
    MODEL_PATH: Path,
) -> tuple:

    ROBOFLOW_API_KEY = os.getenv("ROBOFLOW_API_KEY")

    STATUS_PATH = OUTPUT_PATH / "status.json"

    VIDEO_PATH = Path(os.path.join(INPUT_PATH, f"{VIDEO_FILENAME}.mp4"))
    if not os.path.isfile(VIDEO_PATH):
        print(f"Could not find file: {VIDEO_PATH}")
        os._exit(1)

    BALL_TRACKING_DIRECTORY = os.path.join(OUTPUT_PATH, "BallTracking", f"{VIDEO_FILENAME}")

    if not os.path.isdir(BALL_TRACKING_DIRECTORY):

        os.makedirs(BALL_TRACKING_DIRECTORY, exist_ok=True)

<<<<<<< HEAD
    with open(VALIDATED_VIDEOS, "r") as f:

        validated_videos_arr = f.read()
        validated_videos_arr = validated_videos_arr.split("\n")

    validated = False
    for vid_name in validated_videos_arr:

        if VIDEO_FILENAME == vid_name:
            validated = True
            break

    if not validated:

        status_code = validate_video(
            INPUT_PATH=INPUT_PATH, 
            VIDEO_FILENAME=VIDEO_FILENAME, 
            STATUS_PATH=STATUS_PATH,
            ALREADY_VALIDATED_PATH=VALIDATED_VIDEOS
        )

        if status_code != 0: 

            print(f"Video file is corrupted.")
            VALIDATED_VIDEO_PATH = os.path.join(INPUT_PATH, 'validated_videos', f"{VIDEO_FILENAME}.mp4")
            os.remove(VALIDATED_VIDEO_PATH)
            os.remove(VIDEO_PATH)
            os._exit(1)

    VIDEO_PATH = Path(os.path.join(INPUT_PATH, "validated_videos", f"{VIDEO_FILENAME}.mp4"))

=======
>>>>>>> ca4022552e9814a1150864c076423b9c8b17c841
    bounce_detection_dict, fps = predict(
        video_path=VIDEO_PATH,
        OUTPUT_DIR=OUTPUT_PATH,
        MODEL_PATH=MODEL_PATH,
        STATUS_PATH=STATUS_PATH,
        api_key=ROBOFLOW_API_KEY, 
    )

    results = {
        "ok": True,
        "Bounce Detection Dictionary": bounce_detection_dict,
        "fps": fps,
        "video filename": VIDEO_FILENAME,
        "job id": JOB_ID
    }

    results_path = OUTPUT_PATH.parent / "results.json"

    with open(results_path, "w") as f:
        json.dump(results, f)
