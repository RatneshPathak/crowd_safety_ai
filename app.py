"""Crowd Safety AI - Safety Operations Center (Streamlit front end).

The computer-vision pipeline lives in ``src/`` and is unchanged in spirit;
this file only orchestrates it (``services/analysis.py``) and presents the
results (``ui/``).

Run:  streamlit run app.py
"""
from __future__ import annotations

import importlib.util
import json
import sys
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
for _p in (str(ROOT), str(ROOT / "src")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import streamlit as st

st.set_page_config(
    page_title="Crowd Safety AI · Safety Operations Center",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="collapsed",
)

from services import analysis as svc          # noqa: E402
from ui import charts as CH                   # noqa: E402
from ui import components as C                # noqa: E402
from ui import fx                             # noqa: E402
from ui import metrics as M                   # noqa: E402

PAGES = ["Overview", "Analyze", "Analytics", "Events & Alerts", "About"]
FRAME_LIMITS = [60, 150, 300, 600, 1500, 3000, 0]
UPLOAD_DIR = Path(tempfile.gettempdir()) / "csai_uploads"
PIPELINE_MODULES = ["video_loader", "preprocessing", "density_estimation", "motion_analysis",
                    "anomaly_detection", "risk_classifier", "visualizer", "dashboard"]

ss = st.session_state


def _rel(path: Path) -> str:
    try:
        return str(Path(path).relative_to(ROOT))
    except ValueError:
        return str(path)


# ============================================================ state & helpers
def _init_state() -> None:
    defaults = dict(nav="Overview", job=None, run=None, error=None, upload_path=None,
                    upload_sig=None, upload_info=None, upload_error=None, upload_name=None,
                    effects=True, chart_mode="Detailed", smooth=1, frame_limit=300,
                    source="Upload a video", purged=False)
    for k, v in defaults.items():
        ss.setdefault(k, v)
    if not ss.purged:                      # drop stale temp uploads (>24h)
        ss.purged = True
        try:
            for f in UPLOAD_DIR.glob("*"):
                if time.time() - f.stat().st_mtime > 86400:
                    f.unlink(missing_ok=True)
        except OSError:
            pass


@st.cache_data(ttl=20, show_spinner=False)
def _env_status() -> dict:
    import cv2
    mods = sum(1 for m in PIPELINE_MODULES if importlib.util.find_spec(m) is not None)
    demo = svc.DEFAULT_VIDEO
    return dict(
        cv2=cv2.__version__, modules=mods, modules_total=len(PIPELINE_MODULES),
        demo=demo.is_file(), demo_mb=round(demo.stat().st_size / 1e6, 1) if demo.is_file() else 0,
        ffmpeg=bool(svc.find_ffmpeg()),
    )


def _go(page: str) -> None:
    ss.nav = page


def _open_run(run_id: str) -> None:
    data = svc.load_run(run_id)
    if data:
        ss.run, ss.error = data, None
        ss.nav = "Analyze"
    else:
        ss.error = f"Could not read saved run {run_id}."


def _delete_run(run_id: str) -> None:
    svc.delete_run(run_id)
    if ss.run and ss.run.get("run_id") == run_id:
        ss.run = None


def _use_demo() -> None:
    ss.source = ss.w_source = "Project demo video"
    ss.nav = "Analyze"


def _clear_upload() -> None:
    if ss.upload_path:
        Path(ss.upload_path).unlink(missing_ok=True)
    ss.upload_path = ss.upload_sig = ss.upload_info = ss.upload_error = None


def _finalize_job() -> None:
    """Move a finished background job into session state exactly once."""
    job = ss.job
    if job is None or job.active:
        return
    if job.status == "done":
        data = svc.load_run(job.run_id)
        ss.run, ss.error = data, None
        st.toast(f"Analysis complete · {job.frame:,} frames", icon="✅")
    elif job.status == "failed":
        ss.error = job.error or "The analysis failed for an unknown reason."
        st.toast("Analysis failed", icon="⚠️")
    elif job.status == "cancelled":
        st.toast("Analysis cancelled. Partial output was discarded.", icon="⏹️")
    ss.job = None


def _start_analysis(frame_limit: int) -> None:
    ss.error = None
    try:
        if ss.source == "Project demo video":
            job = svc.start_job(svc.DEFAULT_VIDEO, svc.DEFAULT_VIDEO.name, frame_limit)
        else:
            if not ss.upload_path:
                ss.error = "Choose a video file first."
                return
            path, name = ss.upload_path, ss.upload_name
            ss.upload_path = ss.upload_sig = ss.upload_info = None   # job owns the temp file now
            job = svc.start_job(path, name, frame_limit, cleanup_source=True)
        ss.job = job
    except svc.VideoError as exc:
        ss.error = str(exc)
    except Exception as exc:  # defensive: never crash the dashboard
        ss.error = f"{type(exc).__name__}: {exc}"


# =================================================================== chrome
def _header() -> None:
    env = _env_status()
    job, run = ss.job, ss.run
    if job is not None and job.active:
        label, tone, live, detail = "Analysis running", "info", True, job.video_name
    elif run:
        label, tone, live, detail = "Result ready", "ok", False, run["source_name"]
    else:
        label, tone, live, detail = "Idle · no analysis loaded", "", False, ""
    st.markdown(C.topbar(label, tone, live, detail), unsafe_allow_html=True)

    with st.container(key="nav_row"):
        c1, c2, c3 = st.columns([6, 1.4, 1.6], vertical_alignment="center")
        with c1:
            st.radio("Navigate", PAGES, key="nav", horizontal=True, label_visibility="collapsed")
        with c2:
            with st.popover("⚙ Settings"):
                st.toggle("Ambient background effects", key="effects",
                          help="Particles and light drift. Also off automatically when your OS requests reduced motion.")
                st.radio("Chart view", ["Overview", "Detailed"], key="chart_mode", horizontal=True,
                         help="Overview: four compact charts. Detailed: one coordinated, linked-axis panel.")
                st.slider("Chart smoothing (frames)", 1, 31, key="smooth", step=2,
                          help="Moving average for display only. Raw data stays visible underneath and is never changed.")
        with c3:
            st.button("＋ New analysis", type="primary", key="btn_new", on_click=_go, args=("Analyze",))
    return None


def _job_strip(expanded: bool) -> None:
    job = ss.job
    if job is None:
        return

    @st.fragment(run_every=0.6)
    def live():
        j = ss.job
        if j is None:
            return
        if not j.active:
            st.rerun()          # finalize on a full rerun
            return
        st.markdown(C.progress_panel(j, expanded), unsafe_allow_html=True)
        st.button("Cancel analysis", key="btn_cancel", on_click=j.cancel,
                  help="Stops after the current frame and discards partial output.")

    if job.active:
        live()


# ================================================================== pages
def page_overview() -> None:
    env = _env_status()
    runs = svc.list_runs()
    job, run = ss.job, ss.run
    state_tag = ("Running", "info") if (job and job.active) else (
        (f"{run['summary']['frames_processed']:,} frames", "ok") if run else ("Idle", ""))
    rows = [
        ("cpu", f"Pipeline modules (OpenCV {env['cv2']})",
         f"{env['modules']}/{env['modules_total']} found", "ok" if env["modules"] == env["modules_total"] else "crit"),
        ("video", "Demo video", f"found · {env['demo_mb']} MB" if env["demo"] else "not found",
         "ok" if env["demo"] else "warn"),
        ("frames", "Browser-ready H.264 encoder", "ffmpeg ready" if env["ffmpeg"] else "unavailable",
         "ok" if env["ffmpeg"] else "warn"),
        ("chart", "Saved analyses", str(len(runs)), "info" if runs else ""),
        ("anomaly", "Current analysis", state_tag[0], state_tag[1]),
    ]
    st.markdown(C.hero(rows), unsafe_allow_html=True)

    with st.container(key="hero_actions"):
        a, b = st.columns(2)
        a.button("▶  Start video analysis", type="primary", key="cta_start", on_click=_go, args=("Analyze",))
        if env["demo"]:
            b.button("Explore the demo video", key="cta_demo", on_click=_use_demo)
        elif runs:
            b.button("View latest results", key="cta_latest", on_click=_open_run, args=(runs[0]["run_id"],))
        else:
            b.button("Demo video not found", key="cta_demo_off", disabled=True,
                     help=f"Place a video at {_rel(svc.DEFAULT_VIDEO)} to enable this.")

    _job_strip(expanded=False)

    st.markdown(C.section("Latest analysis", "Signals at a glance",
                          f'<span class="ci-tag">{C.e(run["source_name"])}</span>' if run else ""),
                unsafe_allow_html=True)
    st.markdown(C.metric_grid(run), unsafe_allow_html=True)

    if run:
        left, right = st.columns([1.6, 1], gap="large")
        with left:
            st.markdown(C.section("Trend", "Risk index over video time"), unsafe_allow_html=True)
            st.plotly_chart(CH.risk_fig(run, ss.smooth, 280), config=CH.PLOT_CONFIG, key="ov_risk")
        with right:
            st.markdown(C.section("Class mix", "Frames by pipeline class"), unsafe_allow_html=True)
            st.plotly_chart(CH.risk_class_fig(run, 200), config=CH.PLOT_CONFIG, key="ov_mix")
            st.button("Open full workspace →", key="ov_open", on_click=_go, args=("Analyze",))

    st.markdown(C.section("History", "Recent analyses"), unsafe_allow_html=True)
    _run_history(runs, limit=4)

    st.markdown(C.section("Method", "How each signal is built"), unsafe_allow_html=True)
    st.markdown(C.pipeline_flow(), unsafe_allow_html=True)
    st.markdown(C.not_claimed(), unsafe_allow_html=True)


def _run_history(runs: list[dict], limit: int) -> None:
    if not runs:
        st.markdown(C.run_list([]), unsafe_allow_html=True)
        return
    with st.container(key="run_history"):
        for r in runs[:limit]:
            c1, c2, c3 = st.columns([7, 1.2, 1.2], vertical_alignment="center")
            c1.markdown(C.run_list([r], 1), unsafe_allow_html=True)
            c2.button("Open", key=f"open_{r['run_id']}", on_click=_open_run, args=(r["run_id"],))
            c3.button("Delete", key=f"del_{r['run_id']}", on_click=_delete_run, args=(r["run_id"],),
                      help="Permanently removes this saved run's folder.")


def _source_panel() -> None:
    env = _env_status()
    left, right = st.columns([1, 1.15], gap="large")
    with left, st.container(key="source_card"):
        st.markdown('<div class="ci-eyebrow">Step 1</div><div class="ci-h2">Choose a video</div>',
                    unsafe_allow_html=True)
        options = ["Upload a video"] + (["Project demo video"] if env["demo"] else [])
        if ss.source not in options:
            ss.source = options[0]
        if len(options) > 1:
            ss.source = st.radio("Source", options, index=options.index(ss.source), key="w_source",
                                 horizontal=True, label_visibility="collapsed")

        upload = None
        if ss.source == "Upload a video":
            upload = st.file_uploader(
                "Video file", type=[e.lstrip(".") for e in svc.ALLOWED_EXTENSIONS],
                label_visibility="collapsed",
                help="MP4, AVI, MOV, MKV, M4V or WebM. Re-encode to H.264 MP4 if OpenCV cannot open it.")
            _sync_upload(upload)
        else:
            st.markdown(C.note(f"Using <b>{C.e(svc.DEFAULT_VIDEO.name)}</b> from the project's data folder."),
                        unsafe_allow_html=True)

        st.markdown('<div class="ci-eyebrow" style="margin-top:6px">Step 2</div><div class="ci-h2">Frames to analyze</div>',
                    unsafe_allow_html=True)
        limit = st.select_slider("Frames to analyze", options=FRAME_LIMITS, value=ss.frame_limit,
                                 key="w_frame_limit", label_visibility="collapsed",
                                 format_func=lambda v: "Full video" if v == 0 else f"{v:,}")
        ss.frame_limit = limit
        prev = ss.run
        if prev and prev["summary"].get("elapsed_sec"):
            fps = prev["summary"]["frames_processed"] / prev["summary"]["elapsed_sec"]
            target = limit or (prev["video_info"].get("frames") or 0)
            if target:
                st.caption(f"Last run processed ≈{fps:.1f} frames/s on this machine → "
                           f"about {M.fmt_dur(target / fps)} for {target:,} frames.")
        ready = (ss.source == "Project demo video") or bool(ss.upload_path and not ss.upload_error)
        st.button("▶  Run crowd analysis", type="primary", key="btn_run", disabled=not ready,
                  on_click=_start_analysis, args=(limit,))

    with right, st.container(key="preview_card"):
        st.markdown('<div class="ci-eyebrow">Selected source</div><div class="ci-h2">Preview &amp; validation</div>',
                    unsafe_allow_html=True)
        if ss.source == "Project demo video":
            try:
                info = svc.validate_video(svc.DEFAULT_VIDEO)
                st.video(str(svc.DEFAULT_VIDEO))
                _info_tags(info)
            except svc.VideoError as exc:
                st.markdown(C.error_box("Demo video problem", str(exc)), unsafe_allow_html=True)
        elif ss.upload_error:
            st.markdown(C.error_box("This file cannot be analyzed", ss.upload_error), unsafe_allow_html=True)
        elif ss.upload_path and ss.upload_info:
            st.video(ss.upload_path)
            _info_tags(ss.upload_info)
        else:
            st.markdown(C.empty_state("No video selected",
                                      "Upload a file on the left. It is validated with OpenCV before analysis starts, "
                                      "so unsupported codecs are reported here instead of failing mid-run.", "upload"),
                        unsafe_allow_html=True)


def _info_tags(info) -> None:
    st.markdown(
        '<div style="display:flex;gap:8px;flex-wrap:wrap;margin-top:12px">'
        f'<span class="ci-tag ok">valid</span><span class="ci-tag">{info.width}×{info.height}</span>'
        f'<span class="ci-tag">{info.fps:g} fps</span><span class="ci-tag">{info.frames:,} frames</span>'
        f'<span class="ci-tag">{M.fmt_dur(info.duration_sec)} long</span>'
        f'<span class="ci-tag">{info.size_mb} MB</span></div>', unsafe_allow_html=True)


def _sync_upload(upload) -> None:
    """Persist an uploaded file once, validate it, and clean up on removal."""
    if upload is None:
        if ss.upload_path or ss.upload_error:
            _clear_upload()
        return
    sig = (upload.name, upload.size)
    if sig == ss.upload_sig:
        return
    _clear_upload()
    UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
    path = UPLOAD_DIR / f"{int(time.time() * 1000)}_{Path(upload.name).name}"
    path.write_bytes(upload.getvalue())
    ss.upload_sig, ss.upload_name = sig, upload.name
    try:
        ss.upload_info = svc.validate_video(path)
        ss.upload_path = str(path)
    except svc.VideoError as exc:
        ss.upload_error = str(exc)
        path.unlink(missing_ok=True)


def page_analyze() -> None:
    st.markdown(C.section("Analyze", "Video intelligence workspace"), unsafe_allow_html=True)
    _job_strip(expanded=True)

    if ss.error:
        st.markdown(C.error_box("Analysis problem", ss.error), unsafe_allow_html=True)

    job_active = ss.job is not None and ss.job.active
    if not job_active:
        _source_panel()

    run = ss.run
    if job_active:
        st.markdown(C.video_skeleton(), unsafe_allow_html=True)
        return
    if not run:
        st.markdown(C.section("Result", "Annotated dashboard video"), unsafe_allow_html=True)
        st.markdown(C.empty_state("No analysis yet",
                                  "Run an analysis and the annotated dashboard video appears here with a "
                                  "scrubbable event timeline and a live per-frame readout.", "video"),
                    unsafe_allow_html=True)
        st.markdown(C.section("Signals", "Metrics"), unsafe_allow_html=True)
        st.markdown(C.metric_grid(None), unsafe_allow_html=True)
        return

    S = M.summarize(run)
    st.markdown(C.section("Result", run["source_name"],
                          f'<span class="ci-tag ok">{S["frames"]:,} frames · {M.fmt_dur(S["duration"])}</span>'),
                unsafe_allow_html=True)
    vpath = run.get("video_path")
    if vpath and Path(vpath).is_file():
        with st.container(key="video_stage"):
            st.markdown(C.run_data_tag(run), unsafe_allow_html=True)
            st.video(vpath)
        if not run.get("browser_playable"):
            st.markdown(C.error_box(
                "Preview may not play in your browser",
                "No H.264 encoder was available, so the video is MPEG-4 Part 2. Download it to watch locally, "
                "or run `pip install imageio-ffmpeg` and analyze again."), unsafe_allow_html=True)
    else:
        st.markdown(C.error_box("Output video missing",
                                "The metrics were saved but the rendered video file could not be found."),
                    unsafe_allow_html=True)

    st.markdown(C.metric_grid(run), unsafe_allow_html=True)

    left, right = st.columns([1.5, 1], gap="large")
    with left:
        st.markdown(C.section("Trend", "Risk index with flagged intervals"), unsafe_allow_html=True)
        st.plotly_chart(CH.risk_fig(run, ss.smooth, 300), config=CH.PLOT_CONFIG, key="an_risk")
    with right:
        st.markdown(C.section("Alerts", "Most significant events"), unsafe_allow_html=True)
        top = sorted(run.get("events", []), key=lambda x: (x["severity"] != "high", -x["frames"]))[:4]
        st.markdown(C.event_rows(top), unsafe_allow_html=True)
        st.button("All events →", key="an_events", on_click=_go, args=("Events & Alerts",))

    d1, d2, _ = st.columns([1, 1, 2])
    if vpath and Path(vpath).is_file():
        with open(vpath, "rb") as fh:
            d1.download_button("⬇ Dashboard video", fh.read(), file_name=f"{run['run_id']}_dashboard.mp4",
                               mime="video/mp4", key="dl_video")
    d2.download_button("⬇ Per-frame CSV", M.to_csv(run), file_name=f"{run['run_id']}_frames.csv",
                       mime="text/csv", key="dl_csv")

    st.markdown(C.section("History", "Saved analyses"), unsafe_allow_html=True)
    _run_history(svc.list_runs(), limit=8)


def _need_run(page_label: str) -> bool:
    if ss.run:
        return True
    st.markdown(C.section(page_label, "Nothing to show yet"), unsafe_allow_html=True)
    st.markdown(C.empty_state("No analysis loaded",
                              "This view is built entirely from a completed analysis. Run one, or open a saved run.",
                              "chart"), unsafe_allow_html=True)
    st.button("Go to Analyze →", key=f"need_{page_label}", on_click=_go, args=("Analyze",))
    return False


def page_analytics() -> None:
    if not _need_run("Analytics"):
        return
    run, w = ss.run, ss.smooth
    S = M.summarize(run)
    st.markdown(C.section("Analytics", f"{run['source_name']} · {ss.chart_mode.lower()} view",
                          f'<span class="ci-tag">smoothing {w} fr</span>' if w > 1 else ""),
                unsafe_allow_html=True)
    if ss.chart_mode == "Detailed":
        st.plotly_chart(CH.stacked_fig(run, w, 780), config=CH.PLOT_CONFIG, key="ax_stack")
    else:
        c1, c2 = st.columns(2, gap="large")
        c1.plotly_chart(CH.density_fig(run, w, 280), config=CH.PLOT_CONFIG, key="ax_d")
        c2.plotly_chart(CH.risk_fig(run, w, 280), config=CH.PLOT_CONFIG, key="ax_r")
        c3, c4 = st.columns(2, gap="large")
        c3.plotly_chart(CH.motion_fig(run, w, 280), config=CH.PLOT_CONFIG, key="ax_m")
        c4.plotly_chart(CH.anomaly_fig(run, w, 280), config=CH.PLOT_CONFIG, key="ax_a")

    st.markdown(C.section("Distribution", "Summary statistics"), unsafe_allow_html=True)
    rows = []
    for label, k, dec in (("Density proxy (0–1)", "density", 3), ("Risk index (0–1)", "risk", 2),
                          ("Motion (px/frame)", "motion", 2), ("Anomaly score (0–1)", "anomaly", 2)):
        s = S[k]
        rows.append(f"<tr><td>{label}</td>" + "".join(
            f"<td>{s[x]:.{dec}f}</td>" for x in ("low", "mean", "p95", "peak", "latest")) + "</tr>")
    st.markdown('<div class="ci-card static ci-table-wrap"><table class="ci-kv"><thead><tr><th>Signal</th>'
                '<th>Min</th><th>Mean</th><th>P95</th><th>Peak</th><th>Last frame</th></tr></thead><tbody>'
                + "".join(rows) + "</tbody></table></div>", unsafe_allow_html=True)
    l, r = st.columns([1.2, 1], gap="large")
    with l:
        st.plotly_chart(CH.risk_class_fig(run, 200), config=CH.PLOT_CONFIG, key="ax_mix")
    with r:
        st.markdown(C.note(
            "<b>Reading these charts.</b> The x-axis is video time (frame ÷ container FPS), not processing time. "
            "Values are unvalidated proxies; compare shapes and changes within one video rather than absolute "
            "levels across different cameras."), unsafe_allow_html=True)


def page_events() -> None:
    if not _need_run("Events & Alerts"):
        return
    run = ss.run
    S = M.summarize(run)
    events = run.get("events", [])
    st.markdown(C.section("Events & alerts", run["source_name"],
                          '<span class="ci-tag">offline analysis · no live alerts</span>'),
                unsafe_allow_html=True)
    rid = run["run_id"]
    cat = S["events_by_category"]
    cards = "".join([
        C.metric_card(0, "alert", "Event episodes", len(events), 0, "events",
                      "Consecutive frames over a threshold merged into one episode.", "warn" if events else "ok",
                      None, f"{rid}:ev"),
        C.metric_card(1, "risk", "High risk index", cat.get("High risk index", 0), 0, "episodes",
                      f"{S['high_risk_frames']:,} frames above 0.70 in total.",
                      "crit" if cat.get("High risk index") else "ok", None, f"{rid}:evh"),
        C.metric_card(2, "anomaly", "Statistical anomaly", cat.get("Statistical anomaly", 0), 0, "episodes",
                      f"{S['anomaly_frames']:,} frames with deviation score above 0.70.",
                      "warn" if cat.get("Statistical anomaly") else "ok", None, f"{rid}:eva"),
        C.metric_card(3, "motion", "Motion / density", cat.get("High motion", 0) + cat.get("High density proxy", 0),
                      0, "episodes", "Motion above 15 px/frame or density proxy above 0.80.", "", None, f"{rid}:evm"),
    ])
    st.markdown(f'<div class="ci-metrics">{cards}</div>', unsafe_allow_html=True)
    st.markdown(C.note("<b>Active vs historical.</b> This is a recorded-video analysis, so there are no "
                       "<i>active</i> alerts. Everything below is a historical event from the analyzed footage; "
                       "the counters above are aggregates. Timestamps are frame ÷ container FPS."),
                unsafe_allow_html=True)

    st.markdown(C.section("Timeline", "When events occurred"), unsafe_allow_html=True)
    st.plotly_chart(CH.events_timeline_fig(run, 260), config=CH.PLOT_CONFIG, key="ev_tl")

    st.markdown(C.section("Log", "Event episodes"), unsafe_allow_html=True)
    f1, f2 = st.columns(2)
    sev = f1.multiselect("Severity", ["high", "medium"], default=["high", "medium"], key="ev_sev")
    cats = sorted({e["category"] for e in events})
    pick = f2.multiselect("Category", cats, default=cats, key="ev_cat")
    shown = [e for e in events if e["severity"] in sev and e["category"] in pick]
    st.markdown(C.event_rows(shown), unsafe_allow_html=True)
    if events:
        st.download_button("⬇ Events JSON", json.dumps(events, indent=2), file_name=f"{rid}_events.json",
                           mime="application/json", key="dl_events")


def page_about() -> None:
    env = _env_status()
    st.markdown(C.section("About", "Methodology, limits and project"), unsafe_allow_html=True)
    st.markdown(C.not_claimed(), unsafe_allow_html=True)
    st.markdown(C.section("Pipeline", "From frame to dashboard"), unsafe_allow_html=True)
    st.markdown(C.pipeline_flow(), unsafe_allow_html=True)
    st.markdown(C.section("Signals", "What each number means"), unsafe_allow_html=True)
    st.markdown(f'<div class="ci-card static">{C.methodology_table()}</div>', unsafe_allow_html=True)

    try:
        import plotly
        plotly_v = plotly.__version__
    except Exception:
        plotly_v = "n/a"
    st.markdown(C.section("Environment", "This installation"), unsafe_allow_html=True)
    st.markdown(
        '<div class="ci-card static"><table class="ci-kv"><tbody>'
        f'<tr><td>Python</td><td><code>{sys.version.split()[0]}</code></td></tr>'
        f'<tr><td>Streamlit</td><td><code>{st.__version__}</code></td></tr>'
        f'<tr><td>OpenCV</td><td><code>{env["cv2"]}</code></td></tr>'
        f'<tr><td>Plotly</td><td><code>{plotly_v}</code></td></tr>'
        f'<tr><td>H.264 encoder</td><td>{"available" if env["ffmpeg"] else "not found (output stays MPEG-4 Part 2)"}</td></tr>'
        f'<tr><td>Saved runs folder</td><td><code>{C.e(_rel(svc.OUTPUT_ROOT))}</code></td></tr>'
        '</tbody></table></div>', unsafe_allow_html=True)
    st.markdown(C.section("Project", "Credits"), unsafe_allow_html=True)
    st.markdown(
        '<div class="ci-card static"><p class="ci-sub" style="margin:0">Built for Hackathon Problem Statement 1. '
        'Team lead <b>Ratnesh Pathak</b> · Raunak Gupta (anomaly detection) · Ayush Jha (density and motion) · '
        'Ritesh Mishra (visualization and overlays). The OpenCV pipeline in <code>src/</code> is the team\'s '
        'original work; the operations-center interface wraps it without replacing it.</p></div>',
        unsafe_allow_html=True)


# ==================================================================== main
def main() -> None:
    _init_state()
    fx.inject_theme()
    _finalize_job()
    _header()
    fx.inject_fx(bool(ss.effects))

    page = ss.nav
    {"Overview": page_overview, "Analyze": page_analyze, "Analytics": page_analytics,
     "Events & Alerts": page_events, "About": page_about}.get(page, page_overview)()

    if page != "Overview" and page != "Analyze":
        _job_strip(expanded=False)

    st.markdown(
        '<div class="ci-foot">Crowd Safety AI is an experimental computer-vision demo. Density, motion, risk and anomaly '
        'values are uncalibrated heuristics computed from pixels. They are not head-counts, probabilities of an incident, '
        'or a substitute for trained safety staff.</div>', unsafe_allow_html=True)


main()
