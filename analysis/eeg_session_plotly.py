"""Self-contained HTML alternative to the MNE Qt browser for eyeballing a
whole recorded session.

Why this exists: over X11 forwarding, the Qt browser's amplitude scaling is
shared across all "eeg"-typed channels, so the raw block (~10^5 uV) and the
filtered block (~10^3 uV) can't both be legible at once, and pan/zoom
latency depends on the SSH link. This script instead renders one static
HTML file per participant -- scp/download it and open it in any local
browser, no X11/server/forwarding needed at all. Each channel gets its own
row with raw on one y-axis and filtered on a secondary y-axis (independent
scales, both always legible), plotly's native box-zoom/pan/range-slider for
interaction, and every task/transition/questionnaire boundary marked.

By default every channel's y-axis is auto-ranged with a robust
median +/- k*IQR limit (k=5, same technique used in the colleague's
ADS1299_BLE_gaoteng/tools/data_processing/ads1299_binary_to_csv.py
--plot), so one channel's outlier/artifact spike doesn't flatten its own
row -- or, since axes are independent per row already, doesn't make that
row look deceptively "big" next to a genuinely quieter channel. Override
with --y-min/--y-max for an identical, directly-comparable scale across
channels, and --x-start/--x-end to crop to a specific span (also gives
much finer time resolution within that span, since decimation then only
has to cover the cropped range instead of the whole session).

Usage:
    python -m analysis.eeg_session_plotly --pid P001                     # filtered only (default)
    python -m analysis.eeg_session_plotly --pid P001 --show raw
    python -m analysis.eeg_session_plotly --pid P001 --show both
    python -m analysis.eeg_session_plotly --pid P017 --max-points 4000
    python -m analysis.eeg_session_plotly --pid P001 --y-min -100 --y-max 100   # same scale, every channel
    python -m analysis.eeg_session_plotly --pid P001 --x-start 12 --x-end 15    # zoom to minutes 12-15
"""
import argparse

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from analysis.eeg_continuous_preprocess import CHANNELS, NATIVE_FS, resample_uniform, filter_continuous_uv
from analysis.eeg_session_viewer import load_full_session_uv
from analysis.plot_style import CHANNEL_COLOR, CATEGORICAL, INK_MUTED, GRIDLINE, SURFACE, INK_PRIMARY

PLOTS_DIR = __import__("pathlib").Path(__file__).parents[1] / "data" / "qc_plots" / "session_plotly"


def minmax_decimate(t: np.ndarray, x: np.ndarray, target_points: int):
    """Bucket into `target_points` chunks, keep each chunk's min and max
    sample (in time order) -- unlike plain stride decimation, this can't
    hide a brief clip spike or a transient artifact between kept samples."""
    n = len(x)
    if n <= target_points * 2:
        return t, x
    bucket = n // target_points
    n_full = bucket * target_points
    xt = x[:n_full].reshape(target_points, bucket)
    tt = t[:n_full].reshape(target_points, bucket)
    rows = np.arange(target_points)
    lo_idx, hi_idx = xt.argmin(axis=1), xt.argmax(axis=1)
    first_is_lo = lo_idx <= hi_idx
    out_t = np.empty(target_points * 2)
    out_x = np.empty(target_points * 2)
    out_t[0::2] = np.where(first_is_lo, tt[rows, lo_idx], tt[rows, hi_idx])
    out_x[0::2] = np.where(first_is_lo, xt[rows, lo_idx], xt[rows, hi_idx])
    out_t[1::2] = np.where(first_is_lo, tt[rows, hi_idx], tt[rows, lo_idx])
    out_x[1::2] = np.where(first_is_lo, xt[rows, hi_idx], xt[rows, lo_idx])
    return out_t, out_x


def robust_ylim(x: np.ndarray, k: float = 5.0):
    """median +/- k*IQR -- same technique as the colleague's
    ads1299_binary_to_csv.py --plot: keeps typical EEG visible while a rare
    artifact spike (electrode pop, filtfilt ringing) doesn't get to set the
    axis range for the whole channel."""
    med = np.median(x)
    q25, q75 = np.percentile(x, [25, 75])
    half = max((q75 - q25) * k, 1.0)
    return med - half, med + half


def task_spans(windows):
    spans = {}
    for w in windows:
        lo, hi = spans.get(w.task, (w.t_start, w.t_end))
        spans[w.task] = (min(lo, w.t_start), max(hi, w.t_end))
    return spans


def build_figure(pid: str, sec_uniform, filt, windows, max_points: int, show: str = "filtered",
                  y_min: float = None, y_max: float = None, robust_k: float = 5.0,
                  x_start: float = None, x_end: float = None) -> go.Figure:
    """show: "raw", "filtered", or "both". With one series, that series gets
    the primary (left) y-axis to itself; with "both", raw uses the left axis
    and filtered uses an independent secondary (right) axis, since the two
    are ~50x apart in scale and would otherwise flatten one of them.

    y_min/y_max: fixed range applied to every channel's primary axis (only
    meaningful with show != "both", since raw/filtered are on different
    scales -- ignored with a warning otherwise). Omit for the default: a
    robust per-channel median +/- robust_k*IQR range, so an outlier spike
    in one channel doesn't flatten its own row or exaggerate it next to a
    genuinely quieter channel.

    x_start/x_end: crop to this span (minutes into session) BEFORE
    decimation, so max_points buys full resolution within the crop instead
    of being spread across the whole session."""
    show_raw, show_filt = show in ("raw", "both"), show in ("filtered", "both")
    t0 = sec_uniform.wall_utc_s.iloc[0]
    t_min = (sec_uniform.wall_utc_s.to_numpy() - t0) / 60.0

    if x_start is not None or x_end is not None:
        lo = x_start if x_start is not None else t_min[0]
        hi = x_end if x_end is not None else t_min[-1]
        mask = (t_min >= lo) & (t_min <= hi)
        t_min, sec_uniform, filt = t_min[mask], sec_uniform.loc[mask].reset_index(drop=True), filt.loc[mask].reset_index(drop=True)

    apply_fixed_ylim = (y_min is not None or y_max is not None)
    if apply_fixed_ylim and show == "both":
        print("Warning: --y-min/--y-max ignored with --show both (raw and filtered are ~50x apart in scale); using robust per-axis ranges instead.")
        apply_fixed_ylim = False

    fig = make_subplots(
        rows=len(CHANNELS), cols=1, shared_xaxes=True, vertical_spacing=0.006,
        specs=[[{"secondary_y": True}]] * len(CHANNELS),
        subplot_titles=CHANNELS,
    )

    for i, ch in enumerate(CHANNELS, start=1):
        if show_raw:
            raw_vals = sec_uniform[ch].to_numpy()
            rt, rx = minmax_decimate(t_min, raw_vals, max_points)
            fig.add_trace(go.Scattergl(x=rt, y=rx, mode="lines", line=dict(color=INK_MUTED, width=0.8),
                                        name="raw", legendgroup="raw", showlegend=(i == 1),
                                        hovertemplate="t=%{x:.2f}min<br>raw=%{y:.0f}uV<extra></extra>"),
                           row=i, col=1, secondary_y=False)
            ylim = (y_min, y_max) if apply_fixed_ylim else robust_ylim(raw_vals, robust_k)
            fig.update_yaxes(range=ylim, row=i, col=1, secondary_y=False)
        if show_filt:
            filt_vals = filt[ch].to_numpy()
            ft, fx = minmax_decimate(t_min, filt_vals, max_points)
            fig.add_trace(go.Scattergl(x=ft, y=fx, mode="lines", line=dict(color=CHANNEL_COLOR[ch], width=1.1),
                                        name="filtered", legendgroup="filtered", showlegend=(i == 1),
                                        hovertemplate="t=%{x:.2f}min<br>filt=%{y:.0f}uV<extra></extra>"),
                           row=i, col=1, secondary_y=show_raw)  # share the primary axis alone, or take the secondary if raw is also shown
            ylim = (y_min, y_max) if (apply_fixed_ylim and not show_raw) else robust_ylim(filt_vals, robust_k)
            fig.update_yaxes(range=ylim, row=i, col=1, secondary_y=show_raw)
        fig.update_yaxes(title_text=ch, title_font=dict(size=9), tickfont=dict(size=7),
                          gridcolor=GRIDLINE, row=i, col=1, secondary_y=False)
        if show_raw and show_filt:
            fig.update_yaxes(showgrid=False, tickfont=dict(size=7), row=i, col=1, secondary_y=True)

    spans = task_spans(windows)
    for j, (task, (lo, hi)) in enumerate(sorted(spans.items(), key=lambda kv: kv[1][0])):
        color = CATEGORICAL[j % len(CATEGORICAL)]
        for i in range(1, len(CHANNELS) + 1):
            fig.add_vrect(x0=(lo - t0) / 60, x1=(hi - t0) / 60, fillcolor=color, opacity=0.06,
                          line_width=0, row=i, col=1)
    # task labels once, at the top row only
    for j, (task, (lo, hi)) in enumerate(sorted(spans.items(), key=lambda kv: kv[1][0])):
        fig.add_annotation(x=(lo + hi) / 2 / 60 - t0 / 60, y=1.0, yref="y domain", row=1, col=1,
                           text=task, showarrow=False, font=dict(size=8, color=INK_PRIMARY), yshift=14)

    markers = [w for w in windows if w.window_type in ("transition", "questionnaire")]
    if markers:
        mx = [(w.t_start - t0) / 60 for w in markers]
        mtext = [f"{w.task}/{w.window_type}" for w in markers]
        for i in range(1, len(CHANNELS) + 1):
            for x in mx:
                fig.add_vline(x=x, line=dict(color=INK_MUTED, width=0.6, dash="dot"), row=i, col=1)
        fig.add_trace(go.Scattergl(x=mx, y=[0] * len(mx), mode="markers", marker=dict(size=1, opacity=0),
                                    text=mtext, hovertemplate="%{text}<extra></extra>", showlegend=False),
                       row=1, col=1, secondary_y=False)

    title = {
        "raw": f"{pid}: whole-session raw signal per channel",
        "filtered": f"{pid}: whole-session filtered signal per channel",
        "both": f"{pid}: whole-session raw (gray, left axis) vs. filtered (colored, right axis) per channel",
    }[show]
    fig.update_layout(
        height=170 * len(CHANNELS), width=1500, plot_bgcolor=SURFACE, paper_bgcolor=SURFACE,
        title=title,
        legend=dict(orientation="h", yanchor="bottom", y=1.01, xanchor="left", x=0),
        margin=dict(t=90, b=40),
    )
    fig.update_xaxes(title_text="Minutes into session", row=len(CHANNELS), col=1, rangeslider_visible=True)
    return fig


def main():
    parser = argparse.ArgumentParser(description="Static self-contained HTML whole-session EEG QC viewer.")
    parser.add_argument("--pid", required=True)
    parser.add_argument("--show", choices=["raw", "filtered", "both"], default="filtered",
                        help="which trace(s) to plot (default: filtered only)")
    parser.add_argument("--max-points", type=int, default=3000, help="min-max decimation target points per trace")
    parser.add_argument("--y-min", type=float, default=None, help="fixed lower y-limit, uV, applied to every channel (default: robust per-channel auto-range)")
    parser.add_argument("--y-max", type=float, default=None, help="fixed upper y-limit, uV, applied to every channel")
    parser.add_argument("--robust-k", type=float, default=5.0, help="default auto-range width: median +/- robust_k*IQR (uV)")
    parser.add_argument("--x-start", type=float, default=None, help="crop start, minutes into session")
    parser.add_argument("--x-end", type=float, default=None, help="crop end, minutes into session")
    args = parser.parse_args()

    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    sec, windows = load_full_session_uv(args.pid)
    sec_uniform = resample_uniform(sec, NATIVE_FS)
    filt = filter_continuous_uv(sec_uniform)

    fig = build_figure(args.pid, sec_uniform, filt, windows, args.max_points, show=args.show,
                        y_min=args.y_min, y_max=args.y_max, robust_k=args.robust_k,
                        x_start=args.x_start, x_end=args.x_end)
    if args.x_start is not None or args.x_end is not None:
        lo = f"{args.x_start:g}" if args.x_start is not None else "start"
        hi = f"{args.x_end:g}" if args.x_end is not None else "end"
        suffix = f"{args.show}_{lo}-{hi}min"
    else:
        suffix = args.show
    out = PLOTS_DIR / f"{args.pid}_session_{suffix}.html"
    fig.write_html(out, include_plotlyjs=True)
    print(f"Saved: {out}")
    print("Self-contained -- scp/download this one file and open it directly in any local browser (no server needed).")
    print("Interaction: click-drag to box-zoom (any axis), double-click to reset, scroll the bottom range-slider to pan,")
    print("click a legend entry to toggle a trace off across all 8 rows at once.")


if __name__ == "__main__":
    main()
