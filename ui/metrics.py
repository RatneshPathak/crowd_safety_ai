"""Pure-Python helpers that turn a saved run (result dict) into numbers.

Everything is derived from the per-frame ``records`` written by the pipeline.
No value is invented; anything missing is returned as ``None``.
"""
from __future__ import annotations

import json
from collections import Counter

import numpy as np

HIGH_RISK_INDEX = 0.7        # same threshold the pipeline's counter uses
ANOMALY_THRESHOLD = 0.7


def series(result: dict) -> dict:
    recs = (result or {}).get("records") or []
    def col(k, dtype=float):
        return np.array([r[k] for r in recs], dtype=dtype)
    if not recs:
        z = np.array([])
        return dict(frame=z, t=z, density=z, motion=z, risk=z, risk_raw=z,
                    anomaly_score=z, anomaly=np.array([], dtype=bool),
                    risk_label=[])
    return dict(
        frame=col("frame", int), t=col("t_sec"), density=col("density"),
        motion=col("motion"), risk=col("risk"), risk_raw=col("risk_raw"),
        anomaly_score=col("anomaly_score"), anomaly=col("anomaly", bool),
        risk_label=[r["risk_label"] for r in recs],
    )


def _stats(a: np.ndarray) -> dict | None:
    if a.size == 0:
        return None
    return dict(latest=float(a[-1]), mean=float(a.mean()), peak=float(a.max()),
                low=float(a.min()), p95=float(np.percentile(a, 95)))


def summarize(result: dict) -> dict | None:
    s = series(result)
    n = int(s["frame"].size)
    if n == 0:
        return None
    ev = result.get("events", [])
    by_cat = Counter(e["category"] for e in ev)
    return dict(
        frames=n,
        duration=float(s["t"][-1]),
        density=_stats(s["density"]),
        motion=_stats(s["motion"]),
        risk=_stats(s["risk"]),
        anomaly=_stats(s["anomaly_score"]),
        high_risk_frames=int((s["risk"] > HIGH_RISK_INDEX).sum()),
        anomaly_frames=int(s["anomaly"].sum()),
        label_counts=dict(Counter(s["risk_label"])),
        event_count=len(ev),
        events_by_category=dict(by_cat),
        processing_fps=(n / result["summary"]["elapsed_sec"]
                        if result.get("summary", {}).get("elapsed_sec") else None),
    )


def smooth(a: np.ndarray, window: int) -> np.ndarray:
    """Centered moving average (display only; raw data is never altered)."""
    window = int(window)
    if window <= 1 or a.size < window:
        return a
    k = np.ones(window) / window
    pad = window // 2
    ext = np.pad(a, (pad, window - 1 - pad), mode="edge")
    return np.convolve(ext, k, mode="valid")


def downsample(a: np.ndarray, max_points: int = 240) -> np.ndarray:
    """Bucket-mean downsample for sparklines."""
    if a.size <= max_points:
        return a
    edges = np.linspace(0, a.size, max_points + 1).astype(int)
    return np.array([a[edges[i]:edges[i + 1]].mean() for i in range(max_points)])


def js_payload(result: dict, max_samples: int = 4000) -> str:
    """Compact JSON for the browser (playhead readout + timeline markers)."""
    s = series(result)
    n = s["frame"].size
    step = max(1, int(np.ceil(n / max_samples)))
    sl = slice(0, None, step)
    r3 = lambda a: [round(float(x), 3) for x in a[sl]]
    fps = (result.get("video_info") or {}).get("fps") or 25
    events = [dict(s=e["start_sec"], e=e["end_sec"], sev=e["severity"],
                   label=e["category"]) for e in result.get("events", [])]
    return json.dumps(dict(
        run_id=result.get("run_id"), fps=fps,
        duration=float(s["t"][-1]) if n else 0.0, events=events,
        records=dict(t=[round(float(x), 3) for x in s["t"][sl]],
                     f=[int(x) for x in s["frame"][sl]],
                     density=r3(s["density"]), motion=r3(s["motion"]),
                     risk=r3(s["risk"]), anomaly=r3(s["anomaly_score"])),
    ), separators=(",", ":"))


def to_csv(result: dict) -> str:
    cols = ["frame", "t_sec", "density", "motion", "risk_raw", "risk",
            "risk_label", "anomaly_score", "anomaly"]
    lines = [",".join(cols)]
    for r in result.get("records", []):
        lines.append(",".join(str(r[c]) for c in cols))
    return "\n".join(lines)


def fmt_dur(sec: float | None) -> str:
    if sec is None:
        return "–"
    sec = max(0, int(round(sec)))
    m, s = divmod(sec, 60)
    return f"{m}:{s:02d}"
