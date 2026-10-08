import json
from pathlib import Path
import os
import logging

import modal

app = modal.App("tennis-cv-analytics")

image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("libgl1", "libglib2.0-0")               # OpenCV needs these
    .pip_install_from_requirements("requirements.txt")
    .add_local_dir("backend", remote_path="/root/backend")  # your code + model.ubj
)

# Progress shared between Modal and Render (see step 4)
progress = modal.Dict.from_name("tennis-cv-analytics-progress", create_if_missing=True)

logger = logging.getLogger(__name__)

@app.function(
    image=image,
    cpu=4,             # vs 0.1 on Render
    memory=2048,       # MB
    timeout=30 * 60,
    secrets=[modal.Secret.from_name("tennis-cv-analytics-env")],  # your .env values
)
def run_job(job_id: str, filename: str, video_bytes: bytes) -> dict:

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    from backend.predict import analyze_video

    root = Path("/tmp/jobs") / job_id
    input_dir, output_dir = root / "input", root / "output"
    input_dir.mkdir(parents=True)
    output_dir.mkdir(parents=True)
    (input_dir / filename).write_bytes(video_bytes)

    analyze_video(
        VIDEO_FILENAME=filename.rsplit(".", 1)[0],
        INPUT_PATH=input_dir,
        OUTPUT_PATH=output_dir,
        JOB_ID=job_id,
        MODEL_PATH=Path("/root/backend/models/model.ubj"),
    )

    return json.loads((root / "results.json").read_text())