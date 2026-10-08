from fastapi import FastAPI, UploadFile, File, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse

from pathlib import Path
import os
import logging

from uuid import uuid4

from backend.scripts.sweep import sweeper_loop
from contextlib import asynccontextmanager
import asyncio

import modal

PROJECT_ROOT = Path(__file__).resolve().parent.parent
STORAGE_DIRECTORY = PROJECT_ROOT / "storage"

JOBS_DIR = STORAGE_DIRECTORY / "jobs"
os.makedirs(JOBS_DIR, exist_ok=True)

MODEL_PATH = PROJECT_ROOT / "backend" / "models" / "model.ubj"

run_job = modal.Function.from_name("tennis-cv-analytics", "run_job")
progress = modal.Dict.from_name("tennis-cv-analytics-progress", create_if_missing=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

def save_call_id(job_id: str, call_id: str):

    job_dir = JOBS_DIR / job_id
    job_dir.mkdir(parents=True, exist_ok=True)
    (job_dir / "call_id.txt").write_text(call_id)

def read_call_id(job_id: str) -> str:
    path = JOBS_DIR / job_id / "call_id.txt"
    if not path.is_file():
        raise HTTPException(status_code=404, detail="Job not found")
    return path.read_text()

@asynccontextmanager
async def lifespan(app):
    task = asyncio.create_task(sweeper_loop(JOBS_DIR=JOBS_DIR))
    yield
    task.cancel()

app = FastAPI(lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "https://tennis-computer-vision-analytics.vercel.app"
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/health")
def health():

    return {"status": "ok"}

@app.get("/")
def home():
    return {'message': 'Tennis Computer Vision Analytics is running'}

@app.post("/analyze")
async def analyze(
    video: UploadFile = File(...)
): # File(...) Ellipsis: means this must be an uploaded file and it's required

    try:

        job_id = str(uuid4())
        video_bytes = await video.read()

        input_dir = JOBS_DIR / job_id / "input"
        input_dir.mkdir(parents=True, exist_ok=True)
        (input_dir / video.filename).write_bytes(video_bytes) 

        # reads uploaded video, send it to Modal to start analyzing and keep ticket
        call = await run_job.spawn.aio(job_id, video.filename, video_bytes)   # returns right away
        # save call.object_id somewhere (e.g. storage/jobs/{job_id}/call_id.txt)
        save_call_id(job_id=job_id, call_id=call.object_id)

        return {"ok": True, "video filename": video.filename, "job id": job_id}

    except Exception as e:

        logger.exception("Failed to start job %s for %s", job_id, video.filename)

        raise HTTPException(
            status_code=500,
            detail="Analysis of video failed."
        )

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

    if not (JOBS_DIR / job_id / "call_id.txt").is_file():

        raise HTTPException(
            status_code=404,
            detail="Job not found"
        )

    try:

        status = progress.get(job_id)
    
    except Exception as e:

        logger.exception("Could not read status for job %s", job_id)

        raise HTTPException(
            status_code=500,
            detail="Could not read job status"
        )

    if status is None:
        return {
            "status": "queued", 
            "stage": "starting", 
            "progress": 0,
            "current frame": 0, 
            "total frames": 1
        }
    
    return status

@app.get("/jobs/{job_id}/results")
def get_results(job_id: str):

    call = modal.FunctionCall.from_id(read_call_id(job_id))

    try:

        return call.get(timeout=0)
    
    except TimeoutError:

        raise HTTPException(
            status_code=202, 
            detail="Still Running"
        )
    
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail="Job failed"
        )

