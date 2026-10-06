import os
import json
import threading

status_lock = threading.Lock()

def update_status(
    status_file,
    status,
    stage,
    current_frame,
    total_frames,
):
    progress = (current_frame / total_frames) * 100

    status_dict = {
        "status": status,
        "stage": stage,
        "current frame": current_frame,
        "total frames": total_frames,
        "progress": progress,
    }

    with status_lock:

        with open(status_file, "w") as f:

            json.dump(status_dict, f, indent=4)

