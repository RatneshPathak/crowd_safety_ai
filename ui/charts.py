"""Plotly figures for the analytics views.

All figures are built from the per-frame ``records`` of a saved run. X axes
are video time in seconds (frame / container FPS). Colours are consistent
across every chart: one hue per signal, semantic bands only where a
threshold really exists in the pipeline.
"""
from __future__ import annotations

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from . import metrics as M

C = dict(density="#5ad1ff", motion="#8b9cff", risk="#e8eef8", anomaly="#fbbf24",
         ok="#34d399", warn="#fbbf24", crit="#fb7185", grid="rgba(148,170,205,0.10)",
         axis="#8493ad", text="#e8eef8")

PLOT_CONFIG = {"displaylogo": False, "responsive": True,
               "modeBarButtonsToRemove": ["select2d", "lasso2d", "autoScale2d"],
               "toImageButtonOptions": {"format": "png", "scale": 2}}


def _rgba(hex_color: str, a: float) -> str:
    h = hex_color.lstrip("#")
    r, g, b = (int(h[i:i + 2], 16) for i in (0, 2, 4))
    return f"rgba({r},{g},{b},{a})"


def _base_layout(fig: go.Figure, height: int, title: str | None = None) -> go.Figure:
    fig.update_layout(
        height=height, margin=dict(l=8, r=12, t=40 if title else 14, b=8),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Inter, system-ui, sans-serif", color=C["axis"], size=12),
        title=dict(text=title, x=0.0, xanchor="left",
                   font=dict(size=14, color=C["text"])) if title else None,
        hovermode="x unified",
        hoverlabel=dict(bgcolor="#02050a", bordercolor="rgba(148,170,205,0.3)",
                        font=dict(family="JetBrains Mono, monospace", size=12, color=C["text"])),
        legend=dict(orientation="h", y=1.12, x=1, xanchor="right",
                    bgcolor="rgba(0,0,0,0)", font=dict(size=11)),
        showlegend=False,
        transition=dict(duration=450, easing="cubic-in-out"),
    )
    fig.update_xaxes(gridcolor=C["grid"], zeroline=False, linecolor="rgba(148,170,205,0.2)",
                     tickfont=dict(size=11), automargin=True)
    fig.update_yaxes(gridcolor=C["grid"], zeroline=False, tickfont=dict(size=11),
                     automargin=True)
    return fig


def _add_signal(fig, x, y, name, color, unit, fmt=".3f", raw=None, row=None, fill=True):
    """Main line (optionally smoothed) with a faint raw trace underneath."""
    kw = dict(row=row, col=1) if row else {}
    if raw is not None:
        fig.add_trace(go.Scatter(x=x, y=raw, mode="lines", name=f"{name} (raw)",
                                 line=dict(color=_rgba(color, 0.28), width=1),
                                 hovertemplate=f"raw %{{y:{fmt}}} {unit}<extra></extra>"), **kw)
    fig.add_trace(go.Scatter(
        x=x, y=y, mode="lines", name=name,
        line=dict(color=color, width=2.2, shape="linear"),
        fill="tozeroy" if fill else None, fillcolor=_rgba(color, 0.10),
        hovertemplate=f"{name}: %{{y:{fmt}}} {unit}<extra></extra>"), **kw)


def _empty(msg: str, height: int) -> go.Figure:
    fig = go.Figure()
    fig.add_annotation(text=msg, x=0.5, y=0.5, xref="paper", yref="paper",
                       showarrow=False, font=dict(color=C["axis"], size=13))
    fig.update_xaxes(visible=False)
    fig.update_yaxes(visible=False)
    return _base_layout(fig, height)


def _event_vrects(fig, events, row=None):
    for ev in events:
        if ev["category"] not in ("High risk index", "Statistical anomaly"):
            continue
        kw = dict(row=row, col=1) if row else {}
        fig.add_vrect(x0=ev["start_sec"], x1=max(ev["end_sec"], ev["start_sec"] + 0.04),
                      fillcolor=_rgba(C["crit"] if ev["severity"] == "high" else C["warn"], 0.10),
                      line_width=0, layer="below", **kw)


# ------------------------------------------------------------------ singles
def density_fig(result, smooth_w=1, height=300, show_events=False):
    s = M.series(result)
    if s["t"].size == 0:
        return _empty("No density measurements recorded.", height)
    y = M.smooth(s["density"], smooth_w)
    fig = go.Figure()
    _add_signal(fig, s["t"], y, "Density proxy", C["density"], "idx",
                raw=s["density"] if smooth_w > 1 else None)
    fig.update_yaxes(title_text="Density proxy (0–1)", range=[0, max(1.0, float(s["density"].max()) * 1.05)])
    fig.update_xaxes(title_text="Video time (s)")
    return _base_layout(fig, height)


def risk_fig(result, smooth_w=1, height=300, show_events=True):
    s = M.series(result)
    if s["t"].size == 0:
        return _empty("No risk measurements recorded.", height)
    y = M.smooth(s["risk"], smooth_w)
    fig = go.Figure()
    fig.add_hrect(y0=0.7, y1=1.0, fillcolor=_rgba(C["crit"], 0.07), line_width=0, layer="below")
    fig.add_hrect(y0=0.5, y1=0.7, fillcolor=_rgba(C["warn"], 0.05), line_width=0, layer="below")
    fig.add_hline(y=0.7, line=dict(color=_rgba(C["crit"], 0.6), width=1, dash="dot"),
                  annotation_text="0.70 high-risk index", annotation_position="top left",
                  annotation_font=dict(size=10, color=C["crit"]))
    if show_events:
        _event_vrects(fig, (result.get("events") or [])[:150])
    _add_signal(fig, s["t"], y, "Risk index", C["risk"], "", fmt=".2f",
                raw=s["risk"] if smooth_w > 1 else None, fill=False)
    fig.update_yaxes(title_text="Risk index (0–1)", range=[0, 1.02])
    fig.update_xaxes(title_text="Video time (s)")
    return _base_layout(fig, height)


def motion_fig(result, smooth_w=1, height=300, show_events=False):
    s = M.series(result)
    if s["t"].size == 0:
        return _empty("No motion measurements recorded.", height)
    y = M.smooth(s["motion"], smooth_w)
    fig = go.Figure()
    fig.add_hline(y=15, line=dict(color=_rgba(C["warn"], 0.5), width=1, dash="dot"),
                  annotation_text="15 px/frame alert", annotation_position="top left",
                  annotation_font=dict(size=10, color=C["warn"]))
    _add_signal(fig, s["t"], y, "Motion", C["motion"], "px/frame", fmt=".2f",
                raw=s["motion"] if smooth_w > 1 else None)
    fig.update_yaxes(title_text="Mean flow magnitude (px/frame)", rangemode="tozero")
    fig.update_xaxes(title_text="Video time (s)")
    return _base_layout(fig, height)


def anomaly_fig(result, smooth_w=1, height=300, show_events=False):
    s = M.series(result)
    if s["t"].size == 0:
        return _empty("No anomaly scores recorded.", height)
    y = M.smooth(s["anomaly_score"], smooth_w)
    fig = go.Figure()
    fig.add_hline(y=0.7, line=dict(color=_rgba(C["warn"], 0.6), width=1, dash="dot"),
                  annotation_text="0.70 flag threshold", annotation_position="top left",
                  annotation_font=dict(size=10, color=C["warn"]))
    _add_signal(fig, s["t"], y, "Anomaly score", C["anomaly"], "", fmt=".2f",
                raw=s["anomaly_score"] if smooth_w > 1 else None)
    fig.update_yaxes(title_text="Deviation score (0–1)", range=[0, 1.05])
    fig.update_xaxes(title_text="Video time (s)")
    return _base_layout(fig, height)


# --------------------------------------------------------------- combined
def stacked_fig(result, smooth_w=1, height=760):
    """Four coordinated panels sharing one time axis (detailed view)."""
    s = M.series(result)
    if s["t"].size == 0:
        return _empty("No data recorded.", height)
    fig = make_subplots(rows=4, cols=1, shared_xaxes=True, vertical_spacing=0.045,
                        row_heights=[0.25, 0.30, 0.25, 0.20])
    t = s["t"]
    _add_signal(fig, t, M.smooth(s["density"], smooth_w), "Density proxy", C["density"], "idx",
                raw=s["density"] if smooth_w > 1 else None, row=1)
    fig.add_hrect(y0=0.7, y1=1.0, fillcolor=_rgba(C["crit"], 0.07), line_width=0, layer="below", row=2, col=1)
    fig.add_hrect(y0=0.5, y1=0.7, fillcolor=_rgba(C["warn"], 0.05), line_width=0, layer="below", row=2, col=1)
    _add_signal(fig, t, M.smooth(s["risk"], smooth_w), "Risk index", C["risk"], "", fmt=".2f",
                raw=s["risk"] if smooth_w > 1 else None, row=2, fill=False)
    _add_signal(fig, t, M.smooth(s["motion"], smooth_w), "Motion", C["motion"], "px/frame", fmt=".2f",
                raw=s["motion"] if smooth_w > 1 else None, row=3)
    _add_signal(fig, t, M.smooth(s["anomaly_score"], smooth_w), "Anomaly score", C["anomaly"], "", fmt=".2f",
                raw=s["anomaly_score"] if smooth_w > 1 else None, row=4)
    for r, ttl in enumerate(["Density proxy", "Risk index", "Motion px/f", "Anomaly"], start=1):
        fig.update_yaxes(title_text=ttl, row=r, col=1, title_font=dict(size=11))
    fig.update_yaxes(range=[0, 1.02], row=2, col=1)
    fig.update_yaxes(range=[0, 1.05], row=4, col=1)
    fig.update_xaxes(title_text="Video time (s)", row=4, col=1)
    fig.update_xaxes(showspikes=True, spikemode="across", spikesnap="cursor",
                     spikecolor="rgba(148,170,205,0.45)", spikethickness=1, spikedash="dot")
    return _base_layout(fig, height)


def events_timeline_fig(result, height=260):
    ev = result.get("events") or []
    dur = (M.series(result)["t"][-1] if result.get("records") else 0)
    if not ev:
        return _empty("No events to plot.", height)
    cats = sorted({e["category"] for e in ev})
    fig = go.Figure()
    for e in ev:
        col = C["crit"] if e["severity"] == "high" else C["warn"]
        x1 = max(e["end_sec"], e["start_sec"] + 0.05)
        fig.add_trace(go.Scatter(
            x=[e["start_sec"], x1], y=[e["category"]] * 2, mode="lines",
            line=dict(color=col, width=14), showlegend=False,
            hovertemplate=(f"{e['category']}<br>{e['start_sec']:.2f}s → {e['end_sec']:.2f}s"
                           f"<br>frames {e['start_frame']}–{e['end_frame']}"
                           f"<br>peak {e['peak_value']:.2f}<extra></extra>")))
    fig.update_yaxes(categoryorder="array", categoryarray=cats[::-1], title_text=None)
    fig.update_xaxes(title_text="Video time (s)", range=[0, max(dur, 0.1)])
    fig.update_layout(hovermode="closest")
    return _base_layout(fig, height)


def risk_class_fig(result, height=220):
    counts = (M.summarize(result) or {}).get("label_counts", {})
    if not counts:
        return _empty("No classes recorded.", height)
    order = [("LOW", C["ok"]), ("MEDIUM", C["warn"]), ("HIGH", C["crit"])]
    fig = go.Figure()
    total = sum(counts.values())
    for name, col in order:
        n = counts.get(name, 0)
        fig.add_trace(go.Bar(
            y=["Frames"], x=[n], name=name, orientation="h", marker_color=col,
            hovertemplate=f"{name}: %{{x}} frames ({n / total:.1%})<extra></extra>",
            text=[f"{name} {n / total:.0%}" if n / total > 0.08 else ""], textposition="inside",
            textfont=dict(color="#04101c", size=12)))
    fig.update_layout(barmode="stack", hovermode="closest", showlegend=False)
    fig.update_xaxes(title_text="Frames by pipeline class (raw composite, thresholds 0.40 / 0.70)")
    fig.update_yaxes(visible=False)
    return _base_layout(fig, height)
