"""HTML builders for the Crowd Safety AI interface.

Every function returns an HTML string for ``st.markdown(..., unsafe_allow_html=True)``.
User-supplied text (file names etc.) is always escaped.
"""
from __future__ import annotations

import html
from typing import Optional

import numpy as np

from . import metrics as M

e = html.escape

_ICONS = {
    "frames": '<rect x="3" y="5" width="18" height="14" rx="2"/><path d="M7 5v14M17 5v14M3 10h4M3 14h4M17 10h4M17 14h4"/>',
    "density": '<path d="M3 20h18"/><path d="M5 20V13M9 20V8M13 20V11M17 20V5"/>',
    "risk": '<path d="M12 3l9 16H3z"/><path d="M12 10v4M12 17h.01"/>',
    "motion": '<path d="M3 12h4l2-6 4 12 2-6h6"/>',
    "alert": '<path d="M18 8a6 6 0 10-12 0c0 7-3 8-3 8h18s-3-1-3-8"/><path d="M10 21a2 2 0 004 0"/>',
    "anomaly": '<circle cx="12" cy="12" r="9"/><circle cx="12" cy="12" r="3"/><path d="M12 3v3M12 18v3M3 12h3M18 12h3"/>',
    "video": '<rect x="3" y="5" width="13" height="14" rx="2"/><path d="M16 10l5-3v10l-5-3"/>',
    "upload": '<path d="M12 16V4M7 9l5-5 5 5"/><path d="M4 16v3a1 1 0 001 1h14a1 1 0 001-1v-3"/>',
    "chart": '<path d="M4 4v16h16"/><path d="M8 15l3-4 3 2 5-7"/>',
    "check": '<path d="M5 12l5 5L20 7"/>',
    "x": '<path d="M6 6l12 12M18 6L6 18"/>',
    "cpu": '<rect x="6" y="6" width="12" height="12" rx="2"/><path d="M9 2v4M15 2v4M9 18v4M15 18v4M2 9h4M2 15h4M18 9h4M18 15h4"/>',
    "inbox": '<path d="M3 13l3-8h12l3 8v6H3z"/><path d="M3 13h5l1 3h6l1-3h5"/>',
    "info": '<circle cx="12" cy="12" r="9"/><path d="M12 11v5M12 8h.01"/>',
}


def icon(name: str, cls: str = "ci-ico") -> str:
    return (f'<svg class="{cls}" viewBox="0 0 24 24" aria-hidden="true">'
            f'{_ICONS.get(name, _ICONS["info"])}</svg>')


LOGO = (
    '<svg class="ci-logo" viewBox="0 0 40 40" aria-hidden="true">'
    '<defs><linearGradient id="cig" x1="0" y1="0" x2="1" y2="1">'
    '<stop offset="0" stop-color="#9be8ff"/><stop offset="1" stop-color="#3b82f6"/></linearGradient></defs>'
    '<rect x="1.5" y="1.5" width="37" height="37" rx="11" fill="#0d1626" stroke="url(#cig)" stroke-width="1.5"/>'
    '<circle cx="20" cy="20" r="3.2" fill="url(#cig)"/>'
    '<circle cx="20" cy="20" r="8.5" fill="none" stroke="url(#cig)" stroke-width="1.4" stroke-dasharray="3 3.2"/>'
    '<path d="M20 5.5v4.5M20 30v4.5M5.5 20H10M30 20h4.5" stroke="#5ad1ff" stroke-width="1.5" stroke-linecap="round"/>'
    '</svg>'
)


def _tone(v: Optional[float], warn: float, crit: float) -> str:
    if v is None:
        return ""
    return "crit" if v > crit else "warn" if v > warn else "ok"


# ---------------------------------------------------------------- top bar
def topbar(state_label: str, state_tone: str, live: bool, detail: str = "") -> str:
    det = f' <b>{e(detail)}</b>' if detail else ""
    return (
        '<div class="ci-topbar">'
        f'<div class="ci-brand">{LOGO}<div class="ci-wordmark">CROWD SAFETY AI'
        '<small>CROWD INTELLIGENCE // SAFETY OPERATIONS CENTER</small></div></div>'
        f'<div class="ci-chip" role="status"><span class="ci-dot {state_tone}{" live" if live else ""}"></span>'
        f'{e(state_label)}{det}</div></div>'
    )


# ------------------------------------------------------------------- hero
def hero(state_rows: list[tuple[str, str, str, str]]) -> str:
    """state_rows: (icon, label, tag_text, tag_tone) built from real app state."""
    words = [("See", ""), ("risk", ""), ("before", ""), ("it", ""), ("escalates.", "grad")]
    h1 = "".join(f'<span class="w {c}" style="--i:{i}">{w}</span>' for i, (w, c) in enumerate(words))
    rows = "".join(
        f'<div class="ci-status-row"><span>{icon(ic)}{e(lbl)}</span>'
        f'<span class="ci-tag {tone}">{e(tag)}</span></div>'
        for ic, lbl, tag, tone in state_rows)
    return (
        '<div class="ci-hero"><div>'
        '<div class="ci-hero-eyebrow ci-rise"><span class="ci-tag info">COMPUTER VISION</span>'
        '<span class="ci-eyebrow">Density · Motion · Anomaly · Risk</span></div>'
        f'<h1 aria-label="See risk before it escalates.">{h1}</h1>'
        '<p class="ci-lede ci-rise" style="--i:6">Upload a crowd video and the OpenCV pipeline scores every frame '
        'for a brightness-based density proxy, optical-flow motion and a heuristic risk index, then renders an '
        'annotated dashboard video with an event timeline you can scrub.</p></div>'
        '<aside class="ci-card ci-status-panel static ci-rise" style="--i:4" aria-label="System status">'
        '<div class="ci-eyebrow">System status · measured from this machine</div>'
        f'<div style="margin-top:8px">{rows}</div></aside></div>'
    )


def section(eyebrow: str, title: str, right_html: str = "") -> str:
    return (f'<div class="ci-section"><div><div class="ci-eyebrow">{e(eyebrow)}</div>'
            f'<h2>{e(title)}</h2></div>{right_html}</div>')


# --------------------------------------------------------------- sparkline
def sparkline(values, w: int = 160, h: int = 30) -> str:
    a = np.asarray(values, dtype=float)
    if a.size < 2:
        return ""
    a = M.downsample(a, 90)
    lo, hi = float(a.min()), float(a.max())
    rng = (hi - lo) or 1.0
    xs = np.linspace(1, w - 1, a.size)
    ys = h - 3 - (a - lo) / rng * (h - 6)
    d = "M" + " L".join(f"{x:.1f},{y:.1f}" for x, y in zip(xs, ys))
    area = d + f" L{w - 1},{h} L1,{h} Z"
    return (f'<svg class="ci-spark" viewBox="0 0 {w} {h}" preserveAspectRatio="none" role="img" '
            f'aria-label="Recorded history, {a.size} samples">'
            '<defs><linearGradient id="ci-spark-fill" x1="0" y1="0" x2="0" y2="1">'
            '<stop offset="0" stop-color="currentColor"/><stop offset="1" stop-color="transparent"/></linearGradient></defs>'
            f'<path class="area" d="{area}"/><path d="{d}"/></svg>')


def metric_card(i: int, ico: str, label: str, value: Optional[float], dec: int,
                unit: str, desc: str, tone: str = "", spark=None,
                key: str = "") -> str:
    if value is None:
        val = '<span class="ci-empty-val">—</span>'
    else:
        txt = f"{value:,.{dec}f}"
        val = (f'<span data-count-to="{value:.6f}" data-dec="{dec}" '
               f'data-count-key="{e(key)}">{txt}</span>')
        if unit:
            val += f"<small>{e(unit)}</small>"
    foot = sparkline(spark) if spark is not None else ""
    return (
        f'<article class="ci-card ci-metric {tone}" style="--i:{i}">'
        f'<div class="ci-metric-top"><span class="ci-metric-label">{e(label)}</span>'
        f'<span class="ci-metric-ico">{icon(ico)}</span></div>'
        f'<div class="ci-metric-value">{val}</div>'
        f'<div class="ci-metric-desc">{e(desc)}</div>'
        f'<div class="ci-metric-foot">{foot}</div></article>'
    )


def metric_grid(result: Optional[dict]) -> str:
    """Six cards from real data, or the same six in an honest empty state."""
    S = M.summarize(result) if result else None
    ser = M.series(result) if result else None
    rid = (result or {}).get("run_id", "none")
    if not S:
        cards = [
            ("frames", "Frames analyzed", "Number of frames the pipeline processed."),
            ("density", "Density proxy", "Mean normalized pixel brightness (0–1). Not a head-count."),
            ("risk", "Risk index", "Heuristic 0–1 index from density proxy and motion."),
            ("motion", "Motion", "Mean optical-flow magnitude in px/frame."),
            ("alert", "High-risk frames", "Frames with risk index above 0.70."),
            ("anomaly", "Anomaly frames", "Frames whose deviation score exceeds 0.70."),
        ]
        return '<div class="ci-metrics">' + "".join(
            metric_card(i, ic, lb, None, 0, "", ds + " Run an analysis to populate.")
            for i, (ic, lb, ds) in enumerate(cards)) + "</div>"
    d, m, r, a = S["density"], S["motion"], S["risk"], S["anomaly"]
    cards = [
        metric_card(0, "frames", "Frames analyzed", S["frames"], 0, "frames",
                    f"{M.fmt_dur(S['duration'])} of video · processed at "
                    f"{S['processing_fps']:.1f} fps" if S["processing_fps"] else
                    f"{M.fmt_dur(S['duration'])} of video", "", None, f"{rid}:frames"),
        metric_card(1, "density", "Density proxy", d["latest"], 3, "idx",
                    f"Latest frame. Mean {d['mean']:.3f}, peak {d['peak']:.3f}. Brightness-based; not a head-count.",
                    "", ser["density"], f"{rid}:density"),
        metric_card(2, "risk", "Risk index", r["latest"], 2, "0–1",
                    f"Latest frame. Mean {r['mean']:.2f}, peak {r['peak']:.2f}. Heuristic, not a probability.",
                    _tone(r["latest"], 0.5, 0.7), ser["risk"], f"{rid}:risk"),
        metric_card(3, "motion", "Motion", m["latest"], 2, "px/frame",
                    f"Latest frame. Mean {m['mean']:.2f}, peak {m['peak']:.2f}. Farneback optical flow.",
                    "", ser["motion"], f"{rid}:motion"),
        metric_card(4, "alert", "High-risk frames", S["high_risk_frames"], 0, "frames",
                    f"{S['high_risk_frames'] / S['frames']:.1%} of frames had a risk index above 0.70.",
                    "crit" if S["high_risk_frames"] else "ok", None, f"{rid}:hr"),
        metric_card(5, "anomaly", "Anomaly frames", S["anomaly_frames"], 0, "frames",
                    f"Peak deviation score {a['peak']:.2f}; flagged above 0.70 against the running mean.",
                    "warn" if S["anomaly_frames"] else "ok", ser["anomaly_score"], f"{rid}:an"),
    ]
    return '<div class="ci-metrics">' + "".join(cards) + "</div>"


# ------------------------------------------------------------- states
def empty_state(title: str, text: str, ico: str = "inbox") -> str:
    return (f'<div class="ci-empty ci-rise">{icon(ico)}<h3>{e(title)}</h3><p>{e(text)}</p></div>')


def error_box(title: str, detail: str) -> str:
    return (f'<div class="ci-error" role="alert"><b>{e(title)}</b><br>{e(detail)}</div>')


def note(text_html: str) -> str:
    """text_html is trusted markup written in this repo, not user input."""
    return f'<div class="ci-note">{text_html}</div>'


def skeleton_block(h: int = 120) -> str:
    return f'<div class="ci-skel" style="height:{h}px" aria-hidden="true"></div>'


def video_skeleton() -> str:
    return ('<div class="ci-card static" aria-busy="true"><div class="ci-skel" style="height:min(46vh,380px)"></div>'
            '<div class="ci-skel" style="height:14px;width:60%;margin-top:14px"></div></div>')


# ------------------------------------------------------------ progress
def progress_panel(job, expanded: bool = True) -> str:
    pct = job.fraction * 100
    status = job.status
    label = {"queued": "Preparing", "running": "Analyzing frames",
             "encoding": "Encoding browser-ready video", "done": "Complete",
             "failed": "Failed", "cancelled": "Cancelled"}.get(status, status)
    steps = [("Validate", "done" if status != "queued" else "on"),
             ("Analyze frames", "on" if status == "running" else ("done" if status in ("encoding", "done") else "")),
             ("Encode H.264", "on" if status == "encoding" else ("done" if status == "done" else "")),
             ("Ready", "done" if status == "done" else "")]
    steps_html = "".join(
        f'<div class="ci-step {c}">{icon("check") if c == "done" else ""}{e(n)}</div>' for n, c in steps)
    frames_txt = (f"{job.frame:,} / {job.total:,} frames" if job.total else f"{job.frame:,} frames")
    eta = M.fmt_dur(job.eta_sec) if job.eta_sec is not None else "–"
    live = ""
    if expanded and job.latest:
        L = job.latest
        live = (
            '<div class="ci-live-grid">'
            f'<div class="ci-live-cell"><span>Density proxy</span><b>{L["density"]:.3f}</b></div>'
            f'<div class="ci-live-cell"><span>Motion px/f</span><b>{L["motion"]:.2f}</b></div>'
            f'<div class="ci-live-cell"><span>Risk index</span><b>{L["risk"]:.2f}</b></div>'
            f'<div class="ci-live-cell"><span>Class</span><b>{e(str(L["risk_label"]))}</b></div>'
            f'<div class="ci-live-cell"><span>Anomaly</span><b>{L["anomaly_score"]:.2f}</b></div></div>')
    elif expanded:
        live = '<div class="ci-skel" style="height:58px;margin-top:14px"></div>'
    return (
        '<section class="ci-card static ci-progress" aria-live="polite">'
        f'<div class="ci-progress-head"><div><div class="ci-eyebrow">{e(label)} · {e(job.video_name)}</div>'
        f'<div class="ci-progress-pct">{pct:.0f}%</div></div>'
        f'<div class="ci-tag info">{frames_txt} · elapsed {M.fmt_dur(job.elapsed_sec)} · ETA {eta}</div></div>'
        f'<div class="ci-bar{" done" if status == "done" else ""}" role="progressbar" aria-valuemin="0" '
        f'aria-valuemax="100" aria-valuenow="{pct:.0f}"><i style="width:{max(pct, 2):.1f}%"></i></div>'
        f'<div class="ci-steps">{steps_html}</div>{live}</section>'
    )


# ------------------------------------------------------- video workspace
def run_data_tag(result: dict) -> str:
    return f'<div id="ci-run-data" hidden data-json="{e(M.js_payload(result), quote=True)}"></div>'


# --------------------------------------------------------------- events
def event_rows(events: list[dict], limit: int = 200) -> str:
    if not events:
        return empty_state("No events recorded",
                           "No frame crossed an alert threshold in this run, so there is nothing to list. "
                           "That is a statement about these proxy metrics, not a safety guarantee.", "check")
    out = []
    for ev in sorted(events, key=lambda x: x["start_frame"])[:limit]:
        span = (f'{ev["start_sec"]:.2f}s' if ev["start_frame"] == ev["end_frame"]
                else f'{ev["start_sec"]:.2f}s → {ev["end_sec"]:.2f}s')
        out.append(
            f'<div class="ci-ev {e(ev["severity"])}"><i></i><div><h4>{e(ev["category"])}</h4>'
            f'<p>{e(ev["message"])}</p></div><div class="meta">{span}<br>'
            f'frames {ev["start_frame"]}–{ev["end_frame"]} · {ev["frames"]} fr<br>'
            f'peak {ev["peak_value"]:.2f}</div></div>')
    more = (f'<p class="ci-sub">Showing the first {limit} of {len(events)} events.</p>'
            if len(events) > limit else "")
    return "".join(out) + more


def run_list(runs: list[dict], limit: int = 6) -> str:
    if not runs:
        return empty_state("No saved analyses yet",
                           "Each completed analysis is saved in its own folder under data/outputs, "
                           "so earlier results are never overwritten.", "inbox")
    return "".join(
        f'<div class="ci-run"><div><b>{e(r["source_name"])}</b><br>'
        f'<span>{e(r["created"].replace("T", " "))}</span></div>'
        f'<span>{r["frames"]:,} frames</span></div>' for r in runs[:limit])


# ----------------------------------------------------------------- about
def methodology_table() -> str:
    rows = [
        ("Density proxy", "<code>density_estimation.py</code>",
         "Mean of the blurred, min-max-normalised grayscale frame, averaged over a 10×10 grid, scaled 0–1.",
         "It measures scene brightness, not people. A bright empty platform scores higher than a dark packed one. "
         "Never read it as a head-count."),
        ("Motion", "<code>motion_analysis.py</code>",
         "Mean Farneback optical-flow magnitude between consecutive frames at 960×540, in px/frame.",
         "Includes camera shake and lighting flicker. Not per-person speed."),
        ("Risk index", "<code>main.py</code>, <code>risk_classifier.py</code>",
         "Raw = 0.6·density + 0.4·min(motion/20, 1). Class by thresholds 0.40 / 0.70, then re-mapped to a "
         "0–1 index (class base 0.25 / 0.50 / 0.75 + 0.3·density + 0.2·motion term).",
         "A hand-tuned heuristic, not a calibrated probability of an incident. The index can exceed 0.70 in a frame "
         "whose class is MEDIUM; the high-risk counter uses the index."),
        ("Anomaly score", "<code>anomaly_detection.py</code>",
         "Mean of |motion − running mean| and |density − running mean|, capped at 1. Needs 10 frames of history.",
         "The running mean covers the whole clip so far and motion is in px/frame, so a sustained change keeps "
         "flagging frames. It is a deviation score, not a learned anomaly model."),
        ("Events", "<code>main.py</code>",
         "Consecutive frames over a threshold are merged into one episode with start/end frame, time and peak value.",
         "Times are frame ÷ container FPS. Thresholds: risk index 0.70, anomaly 0.70, motion 15 px/f, density 0.80."),
    ]
    body = "".join(f"<tr><td>{a}</td><td>{b}</td><td>{c}</td><td>{d}</td></tr>" for a, b, c, d in rows)
    return ('<div class="ci-table-wrap"><table class="ci-kv"><thead><tr><th>Signal</th><th>Source</th>'
            f'<th>How it is computed</th><th>Limitation</th></tr></thead><tbody>{body}</tbody></table></div>')


def pipeline_flow() -> str:
    steps = [("Load", "OpenCV VideoCapture, resized to 960×540"),
             ("Preprocess", "Grayscale, Gaussian blur, min-max normalise"),
             ("Density proxy", "10×10 grid mean intensity"),
             ("Motion", "Farneback optical flow"),
             ("Anomaly + risk", "Running-mean deviation, weighted score, bands"),
             ("Render", "1920×1080 dashboard video, H.264 re-encode")]
    return '<div class="ci-flow">' + "".join(f"<div><b>{a}</b>{b}</div>" for a, b in steps) + "</div>"


def not_claimed() -> str:
    return note(
        "<b>What this system does not do.</b> It does not count or identify people, track individuals, "
        "detect stampedes or specific behaviours, or predict emergencies. It has no trained model and no measured "
        "accuracy. Treat every number as a relative, uncalibrated indicator of how this footage changes over time.")
