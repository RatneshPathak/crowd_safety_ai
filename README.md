# Crowd Safety AI — Safety Operations Center

Hackathon Problem Statement 1. Team lead: **Ratnesh Pathak** (MSc IT, Thakur College of Science & Commerce) ·
Raunak Gupta (anomaly detection) · Ayush Jha (density + motion) · Ritesh Mishra (visualization).

An OpenCV pipeline scores every frame of a crowd video, and a Streamlit "operations center" presents the
result: annotated dashboard video with a scrubbable event timeline, linked analytics charts, and an event log.

## Run (Windows PowerShell)
```powershell
cd path\to\crowd_safety_ai
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
streamlit run app.py
```
Optional: put a demo clip at `data\videos\merged_crowd_demo.mp4` to enable the "Explore the demo video" button.
Original desktop mode still works: `python src\main.py`. A synthetic test clip: `python tests\make_test_video.py`.

## What each number really is
| Signal | Computation | Limitation |
|---|---|---|
| Density proxy | mean of blurred, min-max-normalised grayscale, 10×10 grid, 0–1 | **Brightness, not people.** Not a head-count |
| Motion | mean Farneback optical-flow magnitude, px/frame at 960×540 | includes camera shake / flicker |
| Risk index | 0.6·density + 0.4·min(motion/20,1) → class (0.40/0.70) → 0–1 index | heuristic, **not a probability** |
| Anomaly score | deviation from running mean of motion & density, flag > 0.70 | not a learned model; sustained shifts keep flagging |

No trained model, no measured accuracy. It does not count or identify people or detect specific incidents.

## Features
Real per-frame progress and cancel · every run saved in its own `data/outputs/run_*` folder (never overwritten) ·
H.264 re-encode for browser playback (via `imageio-ffmpeg`) · file validation with readable errors ·
per-frame CSV and events JSON export · reduced-motion and keyboard support.

## Layout
`app.py` entry · `src/` original CV pipeline · `services/analysis.py` background runner · `ui/` theme, components, charts ·
`tests/` synthetic clip + service tests.

## Changes to original files (`src/`)
- `main.py`: optional `max_frames`, `progress_callback`, `should_cancel`; returns a summary; keeps full per-frame `records` and merged `events`; fixes zero-FPS/zero-frame-count crashes; the mock `model_accuracy = 92.5` removed.
- `visualizer.py`: removed the fake accuracy box and the `random()` anomaly value (now the real score); removed hard-coded "Mumbai Train Analysis" label and emoji that rendered as `??`.
- `dashboard.py`: relabelled "density" as proxy; replaced the decorative JET legend (no heat-map is drawn) with the real risk bands.
- README previously listed a phone number and wrong thresholds/grid; both corrected/removed.
