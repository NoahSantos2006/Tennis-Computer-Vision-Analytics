from fastapi import FastAPI, UploadFile, File, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse

from pathlib import Path
import shutil
import json
import os

from uuid import uuid4

from backend.predict import analyze_video
from backend.scripts.status import status_lock

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
STORAGE_DIRECTORY = PROJECT_ROOT / "storage"

JOBS_DIR = STORAGE_DIRECTORY / "jobs"
shutil.rmtree(JOBS_DIR, ignore_errors=True)
os.makedirs(JOBS_DIR)

MODEL_PATH = PROJECT_ROOT / "backend" / "models" / "model.ubj"

def create_job():

    job_id = str(uuid4())

    job_dir = JOBS_DIR / job_id
    input_dir = job_dir / "input"
    output_dir = job_dir / "output"

    input_dir.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)

    return job_id, input_dir, output_dir

@app.get("/")
def home():
    return {'message': 'CourtVision is running'}

@app.post("/analyze")
async def analyze(
    background_tasks: BackgroundTasks,
    video: UploadFile = File(...)
): # File(...) Ellipsis: means this must be an uploaded file and it's required

    try:

        job_id, input_dir, output_dir = create_job()

        video_path = input_dir / video.filename
        VIDEO_FILENAME = video.filename.split(".")[0]
        
        # copying the uploaded video's data into a real file
        with open(video_path, "wb") as buffer:
            shutil.copyfileobj(video.file, buffer)

        print(f"File updated and copied into {video_path}")

        background_tasks.add_task(
            func=analyze_video,
            VIDEO_FILENAME=VIDEO_FILENAME,
            INPUT_PATH=input_dir,
            OUTPUT_PATH=output_dir,
            JOB_ID=job_id,
            MODEL_PATH=MODEL_PATH
        )

        return {
            "ok": True,
            "video filename": video.filename,
            'job id': job_id
        }

    except Exception as e:

        print(f"Exception: {e}")

        return {
            "ok": False
        }

@app.get("/video/{job_id}/{filename}")
def get_video(job_id: str, filename: str):

    video_path = JOBS_DIR / job_id / "input" / f"{filename}.mp4"

    if not video_path.is_file():
        raise HTTPException(
            status_code=404,
            detail="Video not found"
        )

    return FileResponse(
        video_path,
        media_type="video/mp4"
    )

@app.get("/jobs/{job_id}/status")
def get_job_status(job_id: str):

    status_file = JOBS_DIR / job_id / "output" / "status.json"
    
    if not status_file.is_file():

        raise HTTPException(
            status_code=404,
            detail="Job status not found"
        )

    with status_lock:

        with open(status_file, "r") as f:
            status = json.load(f)

    return status

@app.get("/jobs/{job_id}/results")
def get_results(job_id: str):

    results_path = JOBS_DIR / job_id / "results.json"

    with open(results_path, "r") as f:

        results = json.load(f)
        results['Bounce Detection Dictionary'] = {int(k): v for k, v in results['Bounce Detection Dictionary'].items()}

    return results