"""Service-layer checks. Run: python tests/test_service.py  (no Streamlit needed)."""
import sys, tempfile, time
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from services import analysis as svc
sys.path.insert(0, str(ROOT / "tests"))
from make_test_video import make

tmp = Path(tempfile.mkdtemp()); out = tmp / "out"
clip = make(str(tmp / "clip.mp4"), seconds=4)

def wait(j):
    while j.active: time.sleep(0.1)
    return j

# 1 invalid inputs -> friendly VideoError
bad = tmp / "bad.mp4"; bad.write_bytes(b"not a video")
for p, label in ((bad, "corrupt"), (tmp / "x.txt", "wrong ext"), (tmp / "missing.mp4", "missing")):
    if label == "wrong ext": p.write_text("x")
    try: svc.start_job(p, p.name, 0, output_root=out); raise SystemExit(f"{label}: no error")
    except svc.VideoError as e: print("ok", label, "->", str(e)[:60])

# 2 frame limit + two runs get separate folders, no stale reuse
j1 = wait(svc.start_job(clip, "a.mp4", 30, output_root=out))
j2 = wait(svc.start_job(clip, "a.mp4", 60, output_root=out))
assert j1.status == j2.status == "done", (j1.error, j2.error)
assert j1.run_id != j2.run_id
assert len(j1.result["records"]) == 30 and len(j2.result["records"]) == 60
assert len(svc.list_runs(out)) == 2
print("ok two runs", len(j1.result["records"]), len(j2.result["records"]), "playable", j1.result["browser_playable"])

# 3 cancel discards partial output
j3 = svc.start_job(clip, "c.mp4", 0, output_root=out)
while j3.frame < 5: time.sleep(0.05)
j3.cancel(); wait(j3)
assert j3.status == "cancelled" and not j3.run_dir.exists()
print("ok cancel cleaned up")

# 4 cleanup_source removes temp upload; delete_run works
up = tmp / "up.mp4"; up.write_bytes(clip.read_bytes() if isinstance(clip, Path) else Path(clip).read_bytes())
j4 = wait(svc.start_job(up, "up.mp4", 20, cleanup_source=True, output_root=out))
assert not up.exists() and j4.status == "done"
assert svc.delete_run(j4.run_id, out) and svc.load_run(j4.run_id, out) is None
print("ok temp cleanup + delete; ALL PASSED")
