"""FastAPI routes, end to end, with Roboflow replaced by a fake.

The upload runs the real pipeline (repeat-frame skipping, ball tracking,
homography, XGBoost). Only the network call is faked: FakeRoboflow "detects"
the ball by finding the white blob in each frame.
"""
import json
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np
import pytest
from fastapi.testclient import TestClient

import backend.main as main
import backend.predict as predict
import backend.scripts.status as status

from tests.conftest import VIDEO_KEYPOINTS

WIDTH, HEIGHT, FRAMES = 640, 360, 30
PLAYER_BOXES = [(300, 40, 340, 110), (280, 250, 330, 345)]  # far player, near player


def ball_position(i):
    # Ball travels down the court and bounces at frame 15.
    x = 200 + 8 * i
    y = 80 + 12 * i if i < 15 else 80 + 12 * 15 - 9 * (i - 15)
    return x, y


def make_video(path: Path, frames=FRAMES, repeat_every=0):

    out = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), 30, (WIDTH, HEIGHT))

    for i in range(frames):

        img = np.zeros((HEIGHT, WIDTH, 3), dtype=np.uint8)
        for x1, y1, x2, y2 in PLAYER_BOXES:

            cv2.rectangle(img, (x1, y1), (x2, y2), (90, 90, 90), -1)

        # repeat_every > 0 freezes some frames to exercise repeat-frame skipping
        j = i - 1 if repeat_every and i % repeat_every == 0 and i > 0 else i
        cv2.circle(img, ball_position(j), 7, (255, 255, 255), -1)
        out.write(img)

    out.release()


class FakeRoboflow:
    calls = []

    @classmethod
    def init(cls, api_url, api_key):
        return cls()

    def run_workflow(self, workflow_id, workspace_name, images):
        
        FakeRoboflow.calls.append(workflow_id)
        img = images["image"]
        ys, xs = np.nonzero(img[:, :, 0] > 200)
        preds = [
            {"class": "player", "class_id": 1, "confidence": 0.9,
             "x": (x1 + x2) / 2, "y": (y1 + y2) / 2, "width": x2 - x1, "height": y2 - y1}
            for x1, y1, x2, y2 in PLAYER_BOXES
        ]

        if len(xs):

            preds.append({"class": "ball", "class_id": 0, "confidence": 0.8,
                          "x": float(xs.mean()), "y": float(ys.mean()), "width": 14, "height": 14})
            
        result = {"predictions": {"predictions": preds}}

        if workflow_id == "detect-with-court":

            result["court_detection_predictions"] = {"predictions": [{"keypoints": VIDEO_KEYPOINTS}]}

        return [result]


@pytest.fixture
def jobs_dir(tmp_path, monkeypatch):
    jobs = tmp_path / "jobs"
    jobs.mkdir()
    monkeypatch.setattr(main, "JOBS_DIR", jobs)
    return jobs


class FakeModalCall:
    """Stands in for the modal.FunctionCall that run_job.spawn returns."""

    finished = {}  # call id -> results dict

    def __init__(self, call_id):
        self.object_id = call_id

    @classmethod
    def from_id(cls, call_id):
        return cls(call_id)

    def get(self, timeout=None):
        return FakeModalCall.finished[self.object_id]


class FakeRunJob:
    """Runs the job right here instead of on Modal, writing into the test's jobs folder."""

    def __init__(self, jobs_dir):
        self.jobs_dir = jobs_dir
        self.spawn = SimpleNamespace(aio=self._spawn)

    async def _spawn(self, job_id, filename, video_bytes):
        job_dir = self.jobs_dir / job_id
        input_dir, output_dir = job_dir / "input", job_dir / "output"
        input_dir.mkdir(parents=True, exist_ok=True)
        output_dir.mkdir(parents=True, exist_ok=True)
        (input_dir / filename).write_bytes(video_bytes)

        predict.analyze_video(
            VIDEO_FILENAME=filename.rsplit(".", 1)[0],
            INPUT_PATH=input_dir,
            OUTPUT_PATH=output_dir,
            JOB_ID=job_id,
            MODEL_PATH=main.MODEL_PATH,
        )
        call_id = f"fc-{job_id}"
        FakeModalCall.finished[call_id] = json.loads((job_dir / "results.json").read_text())
        return FakeModalCall(call_id)


@pytest.fixture
def client(jobs_dir, monkeypatch):
    FakeRoboflow.calls = []
    fake_progress = {}                                          # one shared dict, like the real Dict
    monkeypatch.setattr(status, "progress", fake_progress)      # where update_status writes
    monkeypatch.setattr(main, "progress", fake_progress)        # where /status reads

    monkeypatch.setattr(predict, "InferenceHTTPClient", FakeRoboflow)
    # Never call the real Modal from tests: run the job locally instead.
    monkeypatch.setattr(main, "run_job", FakeRunJob(jobs_dir))
    monkeypatch.setattr(main.modal, "FunctionCall", FakeModalCall)
    # No `with` block: skips the lifespan, so the job sweeper doesn't run.
    return TestClient(main.app)


def upload(client, tmp_path, name="rally.mp4", **video_kwargs):
    video = tmp_path / name
    make_video(video, **video_kwargs)
    with open(video, "rb") as f:
        return client.post("/analyze", files={"video": (name, f, "video/mp4")})


def test_health(client):
    assert client.get("/health").json() == {"status": "ok"}
    assert client.get("/").status_code == 200


def test_upload_runs_the_pipeline_to_finished(client, tmp_path, jobs_dir):
    r = upload(client, tmp_path)
    assert r.status_code == 200
    body = r.json()
    assert body["ok"] is True
    job_id = body["job id"]

    # TestClient runs background tasks before returning, so the job is already done.
    status = client.get(f"/jobs/{job_id}/status").json()
    assert status["status"] == "finished"
    assert status["progress"] == pytest.approx(100)

    results = client.get(f"/jobs/{job_id}/results").json()
    assert results["job id"] == job_id
    assert results["video filename"] == "rally"
    assert results["fps"] == pytest.approx(30)
    for frame_id, event in results["Bounce Detection Dictionary"].items():
        assert 1 <= int(frame_id) <= FRAMES
        assert event["label"] in (1, 2)

    # Frame 1 asks for court keypoints, the rest use the cheaper workflow.
    assert FakeRoboflow.calls.count("detect-with-court") == 1
    assert FakeRoboflow.calls.count("detect") == FRAMES - 1


def test_player_predictions_carry_box_and_court_location(client, tmp_path, jobs_dir):
    """Regression: without pred['box'] players get no court location and hits disappear."""
    job_id = upload(client, tmp_path).json()["job id"]

    preds_file = jobs_dir / job_id / "output" / "predictions" / "rally_predictions.txt"
    preds = json.loads(preds_file.read_text())
    players = [p for frame in preds.values() if frame != "Repeat Frame" for p in frame if p["class"] != "ball"]

    assert players, "fake detector should have returned players"
    for p in players:
        assert len(p["box"]) == 4
        assert len(p["homography location"]) == 2


def test_every_event_has_a_court_location(client, tmp_path, monkeypatch):
    """Regression: hits once came back with a null location, which crashed TennisCourt.jsx.

    The synthetic rally is too short for the model to fire confidently, so force
    one bounce and two hits and check the real location code fills them in.
    """
    import backend.scripts.side_functions as side_functions

    forced = iter([np.array([10]), np.array([5, 20])])  # bounces, then hits
    monkeypatch.setattr(side_functions, "pick_events", lambda frames, p, threshold: next(forced))

    job_id = upload(client, tmp_path).json()["job id"]
    events = client.get(f"/jobs/{job_id}/results").json()["Bounce Detection Dictionary"]

    assert {int(f): e["label"] for f, e in events.items()} == {5: 2, 10: 1, 20: 2}
    for frame_id, event in events.items():
        location = event["homography location"]
        assert location is not None, f"frame {frame_id} has no location"
        assert len(location) == 2 and all(isinstance(v, (int, float)) for v in location)


def test_repeated_frames_are_not_sent_to_roboflow(client, tmp_path, jobs_dir):
    job_id = upload(client, tmp_path, repeat_every=5).json()["job id"]

    assert len(FakeRoboflow.calls) < FRAMES
    preds = json.loads((jobs_dir / job_id / "output" / "predictions" / "rally_predictions.txt").read_text())
    assert "Repeat Frame" in preds.values()
    assert client.get(f"/jobs/{job_id}/status").json()["status"] == "finished"


def test_uploaded_video_can_be_played_back(client, tmp_path):
    
    job_id = upload(client, tmp_path).json()["job id"]

    r = client.get(f"/video/{job_id}/rally")
    assert r.status_code == 200
    assert r.headers["content-type"] == "video/mp4"


def test_unknown_job_status_is_404(client):
    assert client.get("/jobs/not-a-job/status").status_code == 404


def test_unknown_video_is_404(client):
    assert client.get("/video/not-a-job/rally").status_code == 404


def test_unknown_job_results_is_404(client):
    r = client.get("/jobs/not-a-job/results")
    assert r.status_code == 404
