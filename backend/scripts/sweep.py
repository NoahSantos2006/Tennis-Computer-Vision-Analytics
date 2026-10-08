import asyncio, shutil, time
from pathlib import Path
import os
from dotenv import load_dotenv
import logging

from backend.scripts.status import progress

load_dotenv()

MAX_AGE = int(os.getenv("MAX_AGE"))
CHECK_EVERY = int(os.getenv("CHECK_EVERY"))


logger = logging.getLogger(__name__)   

def sweep_once(JOBS_DIR: Path):
    cutoff = time.time() - MAX_AGE
    for job_dir in JOBS_DIR.iterdir():

        marker = job_dir / "call_id.txt"
        if marker.is_file() and marker.stat().st_mtime < cutoff:
            shutil.rmtree(job_dir, ignore_errors=True)
            progress.pop(job_dir.name, None)   # also clear its status from the Dict

async def sweeper_loop(JOBS_DIR: Path):
    while True:
        try:
            await asyncio.to_thread(sweep_once, JOBS_DIR=JOBS_DIR)
        except Exception as e:
            print(f"sweep failed: {e}")
        await asyncio.sleep(CHECK_EVERY)