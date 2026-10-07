import modal
from pathlib import Path

progress = modal.Dict.from_name("tennis-cv-analytics-progress", create_if_missing=True)

def update_status(
    job_id: str,
    status: str,
    stage: str,
    current_frame: int,
    total_frames: int,
) -> None:
    
    current_progress = (current_frame / total_frames) * 100 if total_frames else 0

    status_dict = {
        "status": status,
        "stage": stage,
        "current frame": current_frame,
        "total frames": total_frames,
        "progress": current_progress,
    }

    progress[job_id] = status_dict

