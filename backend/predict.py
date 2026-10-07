import cv2
import json
import json
import numpy as np
from pathlib import Path
import pickle
from concurrent.futures import ThreadPoolExecutor, wait, FIRST_COMPLETED
import logging

import xgboost as xgb

from dotenv import load_dotenv
import os
import time, random

from inference_sdk import InferenceHTTPClient
from inference_sdk.http.errors import HTTPClientError

from backend.scripts.side_functions import run_predictions, get_bounces
from backend.scripts.ball_tracker import BallTracker
from backend.scripts.status import update_status

load_dotenv()

import json

MAX_WORKERS = int(os.getenv("MAX_WORKERS"))
WORKSPACE_NAME = os.getenv("WORKSPACE_NAME")
WORKFLOW_ID = os.getenv("WORKFLOW_ID")
WORKFLOW_ID_WITH_COURT_POINTS = os.getenv("WORKFLOW_ID_WITH_COURT_POINTS")
MAX_ATTEMPTS = int(os.getenv("MAX_ATTEMPTS"))

logger = logging.getLogger(__name__)

def is_repeat(frame, prev):
    a = cv2.cvtColor(cv2.resize(frame, (480, 270)), cv2.COLOR_BGR2GRAY)
    b = cv2.cvtColor(cv2.resize(prev, (480, 270)), cv2.COLOR_BGR2GRAY)
    return np.count_nonzero(cv2.absdiff(a, b) > 20) < 20

def predict(
        video_path: Path,
        OUTPUT_DIR: Path,
        MODEL_PATH: Path, 
        STATUS_PATH: Path,
        api_key: str,
        MAX_WORKERS: int = MAX_WORKERS,
    ) -> dict:
    
    parts = video_path.parts

    VIDEO_FILENAME = parts[-1].split(".")[0]

    # example: input/make/dunk/make4.mp4
    INPUT_VIDEO = str(video_path)

    cap = cv2.VideoCapture(INPUT_VIDEO)
    if not cap.isOpened():
        raise ValueError("Could not open video: %s", INPUT_VIDEO)
    
    fps = cap.get(cv2.CAP_PROP_FPS)

    total_frames = cap.get(cv2.CAP_PROP_FRAME_COUNT)

    start = time.time()

    update_status(
        status_file=STATUS_PATH,
        status="in progress",
        stage="Processing Frames",
        current_frame=0,
        total_frames=total_frames
    )

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

        data = None

        h, w = frame.shape[:2]
        scale = 1.0
        if w > 1280:
            scale = 1280 / w
            frame_small = cv2.resize(frame, (1280, int(h * scale)), interpolation=cv2.INTER_AREA)
        else:
            frame_small = frame

        if frame_id == 1 or frame_id % 50 == 0:
            
            for attempt in range(MAX_ATTEMPTS):

                try:

                    data = client.run_workflow(
                        workflow_id=WORKFLOW_ID_WITH_COURT_POINTS,
                        workspace_name=WORKSPACE_NAME,
                        images={"image": frame_small},
                    )[0]
                    break

                except HTTPClientError as e:

                    delay = min(2 ** attempt, 30) + random.random()
                    
                    logging.warning(
                        " Roboflow request failed. "
                        "Retrying in %ds (on attempt %d)", delay, attempt
                    )

                    time.sleep(delay)

        else:

            for attempt in range(MAX_ATTEMPTS):
            
                try:

                    data = client.run_workflow(
                        workflow_id=WORKFLOW_ID,
                        workspace_name=WORKSPACE_NAME,
                        images={"image": frame_small},
                    )[0]
                    break

                except HTTPClientError as e:

                    delay = min(2 ** attempt, 30) + random.random()
                    
                    logging.warning(
                        " Roboflow request failed. "
                        "Retrying in %ds (on attempt %d)", delay, attempt
                    )
                    time.sleep(delay)
        
        if not data:

            return frame_id, None

        result = data.get("predictions", {}).get("predictions", [])
        for pred in result:

            pred["x"] /= scale
            pred["y"] /= scale
            pred["width"] /= scale
            pred["height"] /= scale  

        court_detection_data = data.get("court_detection_predictions", {})
        if court_detection_data:

            if court_detection_data.get("predictions", []):

                court_detection_points = court_detection_data.get('predictions', [])[0].get("keypoints", [])
                for court_point in court_detection_points:

                    court_point["x"] /= scale
                    court_point["y"] /= scale

                court_detection_points_by_frame[frame_id] = court_detection_points    

        return frame_id, result

    MAX_PENDING = MAX_WORKERS * 2

    # bounded concurrency
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:

        pending = set()
        frame_id = 1
        completed_frames = 0
        video_finished = False
        previous_frame = None
        repeat_frames = set()
        failed_frames = []

        while pending or not video_finished:

            while len(pending) < MAX_PENDING and not video_finished:

                ret, frame = cap.read()
                if not ret:
                    video_finished = True
                    break

                if previous_frame is not None:

                    if is_repeat(frame=frame, prev=previous_frame):

                        repeat_frames.add(frame_id)
                        frame_id += 1
                        continue

                future = executor.submit(
                    predict_frame,
                    frame_id=frame_id,
                    frame=frame,
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

                try:
                    result_frame_id, preds = future.result()
                except Exception as e:
                    logging.error("frame failed: %s", e)
                    continue
                
                if preds is None:

                    failed_frames.append(result_frame_id)
                    preds = []

                for pred in preds:
                    x, y, w, h = pred["x"], pred["y"], pred["width"], pred["height"]
                    pred["box"] = [int(x - w / 2), int(y - h / 2), int(x + w / 2), int(y + h / 2)]

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

    update_status(
        status_file=STATUS_PATH,
        status="in progress",
        stage="Processing Frames",
        current_frame=total_frames,
        total_frames=total_frames
    )

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

    update_status(
        status_file=STATUS_PATH,
        status="in progress",
        stage="Detecting Bounces and Hits",
        current_frame=0,
        total_frames=total_frames
    )

    bounce_detection_dict = get_bounces(
        VIDEO_FILENAME = VIDEO_FILENAME,
        OUTPUT_PATH = OUTPUT_DIR,
        MODEL = XGBoost_model,
        BALL_TRACKER_PREDICTIONS = ball_tracker.tracker,
        PREDICTIONS_BY_FRAME = predictions_by_frame,
        STATUS_PATH=STATUS_PATH
    )

    update_status(
        status_file=STATUS_PATH,
        status="finished",
        stage="",
        current_frame=total_frames,
        total_frames=total_frames
    )

    end = time.time()

    video_duration = np.round(total_frames / fps, 2)
    elapsed = np.round(end - start, 2)

    logging.info("For a %lf second video using %d max workers it took %lf seconds (%s.mp4)", video_duration, MAX_WORKERS, elapsed, VIDEO_FILENAME)

    return bounce_detection_dict, fps

def analyze_video(
    VIDEO_FILENAME: str,
    INPUT_PATH: Path,
    OUTPUT_PATH: Path,
    JOB_ID: str,
    MODEL_PATH: Path,
) -> tuple:

    ROBOFLOW_API_KEY = os.getenv("ROBOFLOW_API_KEY")
    STATUS_PATH = OUTPUT_PATH / "status.json"
    BALL_TRACKING_DIRECTORY = os.path.join(OUTPUT_PATH, "BallTracking", f"{VIDEO_FILENAME}")

    try:

        VIDEO_PATH = Path(os.path.join(INPUT_PATH, f"{VIDEO_FILENAME}.mp4"))
        if not os.path.isfile(VIDEO_PATH):
            raise FileNotFoundError("Could not find file: %s", VIDEO_PATH)

        if not os.path.isdir(BALL_TRACKING_DIRECTORY):

            os.makedirs(BALL_TRACKING_DIRECTORY, exist_ok=True)

        bounce_detection_dict, fps = predict(
            video_path=VIDEO_PATH,
            OUTPUT_DIR=OUTPUT_PATH,
            MODEL_PATH=MODEL_PATH,
            STATUS_PATH=STATUS_PATH,
            api_key=ROBOFLOW_API_KEY, 
        )

    except Exception:
        
        logger.exception("job %s failed", JOB_ID)
        update_status(
            status_file=STATUS_PATH, 
            status="failed",
            stage="Could not process video", 
            current_frame=0, 
            total_frames=1
        )
        raise

    results = {
        "ok": True,
        "Bounce Detection Dictionary": bounce_detection_dict,
        "fps": fps,
        "video filename": VIDEO_FILENAME,
        "job id": JOB_ID
    }

    results_path = OUTPUT_PATH.parent / "results.json"

    with open(results_path, "w") as f:
        json.dump(results, f, indent=4)
