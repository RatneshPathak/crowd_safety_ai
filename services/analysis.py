"""Analysis orchestration for the Streamlit front end.

Runs the existing OpenCV pipeline (``src/main.py``) in a background thread,
exposes *real* progress (frames processed / frames planned), supports
cancellation, writes every run to its own folder, converts the output video
to browser-playable H.264 when an ffmpeg binary is available, and cleans up
temporary files.

Nothing in here imports Streamlit, so it can be tested on its own.
"""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import threading
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Callable, Optional

import cv2

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))
if str(ROOT) not in sys.path:
    sys.path.insert(1, str(ROOT))

import config  # noqa: E402  (project config, resolved from ROOT)

ALLOWED_EXTENSIONS = (".mp4", ".avi", ".mov", ".mkv", ".m4v", ".webm")
OUTPUT_ROOT = ROOT / getattr(config, "OUTPUT_DIR", "data/outputs")
DEFAULT_VIDEO = ROOT / getattr(config, "VIDEO_PATH", "data/videos/merged_crowd_demo.mp4")
RESULT_FILE = "result.json"
VIDEO_FILE = "dashboard.mp4"


class VideoError(Exception):
    """A user-facing problem with the selected video."""


@dataclass
class VideoInfo:
    width: int
    height: int
    fps: float
    frames: int
    duration_sec: float
    size_mb: float

    def as_dict(self):
        return self.__dict__.copy()


def validate_video(path) -> VideoInfo:
    """Open the file with OpenCV and read one frame, or raise VideoError."""
    path = Path(path)
    if not path.is_file():
        raise VideoError(f"Video file not found: {path.name}")
    if path.suffix.lower() not in ALLOWED_EXTENSIONS:
        raise VideoError(
            f"'{path.suffix or 'no extension'}' is not a supported format. "
            f"Use one of: {', '.join(ALLOWED_EXTENSIONS)}."
        )
    if path.stat().st_size == 0:
        raise VideoError("The file is empty (0 bytes).")
    cap = cv2.VideoCapture(str(path))
    try:
        if not cap.isOpened():
            raise VideoError(
                "OpenCV could not open this file. The container or codec may "
                "be unsupported - re-encode it as H.264 MP4 and try again."
            )
        ok, _ = cap.read()
        if not ok:
            raise VideoError(
                "The file opened but no frame could be decoded. It may be "
                "corrupt or use a codec this OpenCV build cannot decode."
            )
        fps = cap.get(cv2.CAP_PROP_FPS) or 0.0
        frames = max(int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0), 0)
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    finally:
        cap.release()
    eff_fps = fps if 0 < fps <= 240 else 25.0
    return VideoInfo(w, h, round(fps, 3), frames,
                     round(frames / eff_fps, 2) if frames else 0.0,
                     round(path.stat().st_size / 1e6, 2))


def find_ffmpeg() -> Optional[str]:
    exe = shutil.which("ffmpeg")
    if exe:
        return exe
    try:
        import imageio_ffmpeg
        return imageio_ffmpeg.get_ffmpeg_exe()
    except Exception:
        return None


def transcode_h264(src: Path, dst: Path, timeout: int = 900) -> bool:
    """Re-encode to H.264/yuv420p so every browser can play it."""
    exe = find_ffmpeg()
    if not exe:
        return False
    cmd = [exe, "-y", "-loglevel", "error", "-i", str(src),
           "-c:v", "libx264", "-preset", "veryfast", "-crf", "23",
           "-pix_fmt", "yuv420p", "-movflags", "+faststart", "-an", str(dst)]
    flags = getattr(subprocess, "CREATE_NO_WINDOW", 0)
    try:
        r = subprocess.run(cmd, capture_output=True, timeout=timeout,
                           creationflags=flags)
    except (subprocess.TimeoutExpired, OSError):
        return False
    return r.returncode == 0 and dst.is_file() and dst.stat().st_size > 0


# --------------------------------------------------------------------------
# Jobs
# --------------------------------------------------------------------------
@dataclass
class Job:
    run_id: str
    video_name: str
    max_frames: int
    run_dir: Path
    status: str = "queued"      # queued|running|encoding|done|failed|cancelled
    frame: int = 0
    total: int = 0              # planned frames (0 = unknown)
    latest: dict = field(default_factory=dict)
    error: str = ""
    result: Optional[dict] = None
    started: float = field(default_factory=time.time)
    finished: float = 0.0
    cancel_event: threading.Event = field(default_factory=threading.Event)
    thread: Optional[threading.Thread] = None
    # real throughput samples (frames, wall-time) for ETA
    _t0: float = 0.0

    @property
    def active(self) -> bool:
        return self.status in ("queued", "running", "encoding")

    @property
    def fraction(self) -> float:
        if self.status == "done":
            return 1.0
        if self.status == "encoding":
            return 1.0
        if self.total > 0:
            return min(self.frame / self.total, 1.0)
        return 0.0

    @property
    def eta_sec(self) -> Optional[float]:
        if self.status != "running" or self.frame < 5 or self.total <= 0:
            return None
        elapsed = time.time() - self._t0
        rate = self.frame / elapsed if elapsed > 0 else 0
        return (self.total - self.frame) / rate if rate > 0 else None

    @property
    def elapsed_sec(self) -> float:
        return (self.finished or time.time()) - self.started

    def cancel(self):
        self.cancel_event.set()


def new_run_id() -> str:
    return datetime.now().strftime("run_%Y%m%d_%H%M%S_") + uuid.uuid4().hex[:4]


def start_job(source_path, display_name: str, max_frames: int = 0,
              cleanup_source: bool = False,
              output_root: Optional[Path] = None) -> Job:
    """Validate the video and start processing in a daemon thread.

    ``max_frames`` of 0 means the whole video.
    Raises VideoError synchronously if the file is unusable.
    """
    source_path = Path(source_path)
    try:
        info = validate_video(source_path)
    except VideoError:
        if cleanup_source:
            source_path.unlink(missing_ok=True)
        raise
    output_root = Path(output_root or OUTPUT_ROOT)
    run_id = new_run_id()
    run_dir = output_root / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    planned = info.frames if info.frames else 0
    if max_frames and planned:
        planned = min(planned, max_frames)
    elif max_frames and not planned:
        planned = max_frames
    job = Job(run_id=run_id, video_name=display_name,
              max_frames=int(max_frames), run_dir=run_dir, total=planned)
    job.thread = threading.Thread(
        target=_run, args=(job, source_path, info, cleanup_source),
        name=f"analysis-{run_id}", daemon=True)
    job.thread.start()
    return job


def _run(job: Job, source: Path, info: VideoInfo, cleanup_source: bool):
    raw_path = job.run_dir / "raw_output.mp4"
    try:
        from main import EnhancedCrowdSafetySystem  # existing pipeline

        job.status = "running"
        job._t0 = time.time()
        system = EnhancedCrowdSafetySystem()

        def on_progress(frame_num, planned, latest):
            job.frame = frame_num
            if planned:
                job.total = planned
            job.latest = latest

        summary = system.process_video(
            video_path=str(source),
            output_path=str(raw_path),
            display=False,
            max_frames=job.max_frames or None,
            progress_callback=on_progress,
            should_cancel=job.cancel_event.is_set,
        )

        if job.cancel_event.is_set():
            job.status = "cancelled"
            return
        if summary is None or not system.records:
            raise RuntimeError(
                "The pipeline produced no frames. The video may be "
                "unreadable or too short.")

        playable = False
        video_name = None
        if raw_path.is_file() and raw_path.stat().st_size > 0:
            job.status = "encoding"
            final = job.run_dir / VIDEO_FILE
            if transcode_h264(raw_path, final):
                playable = True
                video_name = VIDEO_FILE
            else:
                raw_path.replace(final)  # keep the mp4v file; download works
                video_name = VIDEO_FILE
        raw_path.unlink(missing_ok=True)

        result = {
            "run_id": job.run_id,
            "created": datetime.now().isoformat(timespec="seconds"),
            "source_name": job.video_name,
            "max_frames": job.max_frames,
            "video_info": info.as_dict(),
            "summary": summary,
            "video_file": video_name,
            "browser_playable": playable,
            "records": system.records,
            "events": system.events,
            "thresholds": {
                "risk_low": config.LOW_RISK_THRESHOLD,
                "risk_high": config.HIGH_RISK_THRESHOLD,
                "high_risk_index": 0.7,
                "anomaly": 0.7,
                "motion_alert_px": 15.0,
                "density_alert": 0.8,
            },
        }
        (job.run_dir / RESULT_FILE).write_text(
            json.dumps(result), encoding="utf-8")
        job.result = result
        job.status = "done"
    except Exception as exc:  # never let a bad video kill the dashboard
        job.error = f"{type(exc).__name__}: {exc}"
        job.status = "failed"
    finally:
        job.finished = time.time()
        if job.status in ("failed", "cancelled"):
            shutil.rmtree(job.run_dir, ignore_errors=True)
        else:
            raw_path.unlink(missing_ok=True)
        if cleanup_source:
            try:
                source.unlink(missing_ok=True)
            except OSError:
                pass


# --------------------------------------------------------------------------
# Saved runs
# --------------------------------------------------------------------------
def list_runs(output_root: Optional[Path] = None) -> list[dict]:
    """Lightweight metadata for every saved run, newest first."""
    root = Path(output_root or OUTPUT_ROOT)
    runs = []
    if not root.is_dir():
        return runs
    for d in sorted(root.glob("run_*"), reverse=True):
        f = d / RESULT_FILE
        if not f.is_file():
            continue
        try:
            data = json.loads(f.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        runs.append({
            "run_id": data.get("run_id", d.name),
            "created": data.get("created", ""),
            "source_name": data.get("source_name", "unknown"),
            "frames": data.get("summary", {}).get("frames_processed", 0),
        })
    return runs


def load_run(run_id: str, output_root: Optional[Path] = None) -> Optional[dict]:
    root = Path(output_root or OUTPUT_ROOT)
    f = root / Path(run_id).name / RESULT_FILE   # .name blocks path tricks
    if not f.is_file():
        return None
    try:
        data = json.loads(f.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    data["video_path"] = str(f.parent / data["video_file"]) \
        if data.get("video_file") else None
    return data


def delete_run(run_id: str, output_root: Optional[Path] = None) -> bool:
    root = Path(output_root or OUTPUT_ROOT)
    d = root / Path(run_id).name
    if d.is_dir() and d.name.startswith("run_"):
        shutil.rmtree(d, ignore_errors=True)
        return True
    return False
