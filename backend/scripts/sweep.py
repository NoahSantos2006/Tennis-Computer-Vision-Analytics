import asyncio, shutil, time
from pathlib import Path
import os

MAX_AGE = int(os.getenv("MAX_AGE"))
CHECK_EVERY = int(os.getenv("CHECK_EVERY"))

def sweep_once(JOBS_DIR: Path):
    cutoff = time.time() - MAX_AGE
    for job_id in JOBS_DIR.iterdir():

        results_path = JOBS_DIR / job_id / "results.json"

        if results_path.is_file() and results_path.stat().st_mtime < cutoff:
            shutil.rmtree(job_id, ignore_errors=True)

async def sweeper_loop():
    while True:
        try:
            await asyncio.to_thread(sweep_once)
        except Exception as e:
            print(f"sweep failed: {e}")
        await asyncio.sleep(CHECK_EVERY)