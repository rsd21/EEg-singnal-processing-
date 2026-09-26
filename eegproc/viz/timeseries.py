"""Multichannel time-series plots."""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np

from ..core.annotations import is_bad_description
from . import _style as S


def plot_raw(raw, start: float = 0.0, duration: float = 10.0, picks="data", scale: float | str = "auto",
             highlight: dict | None = None, quality=None, annotations: bool = True,
             title: str | None = None, max_channels: int = 40, display_filter: tuple | None = None,
             show: bool = False):
    """Stacked traces, one row per channel, like a clinical EEG viewer.

    Parameters
    ----------
    start, duration : window to show (s).
    scale : µV between rows, or ``'auto'`` (from the median robust amplitude).
    highlight : ``{channel: colour}`` to colour chosen traces.
    quality : a :class:`ChannelQualityReport`; traces are coloured by grade
        and the score is written next to the channel name.
    annotations : shade annotated spans (``BAD*`` segments in red).
    display_filter : ``(l_freq, h_freq)`` applied to the shown traces only
        (like the filter knobs of a clinical viewer), e.g. ``(0.5, 40)``.
    """
    idx = raw.pick_indices(picks)[:max_channels]
    a, b = raw.time_to_index(start), raw.time_to_index(start + duration)
    data = raw.data[idx, a:b]
    if display_filter is not None:
        from ..preprocessing.filters import filter_data

        pad = int(2 * raw.sfreq)
        lo, hi = max(0, a - pad), min(raw.n_times, b + pad)
        data = filter_data(raw.data[idx, lo:hi], raw.sfreq, *display_filter)[:, a - lo: a - lo + (b - a)]
    t = np.arange(a, b) / raw.sfreq
    x = data - np.median(data, axis=1, keepdims=True)
    if scale == "auto":
        amp = np.median(np.percentile(np.abs(x), 95, axis=1))
        scale = float(max(amp * 2.5, 1e-6))
    names = [raw.ch_names[i] for i in idx]
    colors = {}
    labels = {n: n for n in names}
    if quality is not None:
        for n in names:
            try:
                row = quality[n]
            except KeyError:
                continue
            if row["grade"] != "good":
                colors[n] = S.GRADE_STATUS[row["grade"]]
            labels[n] = f"{n}  {row['score']:.0f}"
    for bname in raw.bads:
        if bname in names:
            colors.setdefault(bname, S.STATUS["critical"])
            labels[bname] = labels[bname] + " (bad)"
    colors.update(highlight or {})

    with S.style():
        height = max(3.0, 0.28 * len(idx) + 1.2)
        fig, ax = plt.subplots(figsize=(11, height), layout="constrained")
        offsets = -np.arange(len(idx)) * scale
        for k, n in enumerate(names):
            c = colors.get(n, S.TRACE)
            ax.plot(t, np.clip(x[k], -1.1 * scale, 1.1 * scale) + offsets[k], color=c,
                    lw=0.9 if c != S.TRACE else 0.6)
        if annotations and len(raw.annotations):
            ann = raw.annotations.crop(start, start + duration, shift=False)
            for o, d, s in zip(ann.onset, ann.duration, ann.description):
                bad = is_bad_description(s)
                if d > 0:
                    ax.axvspan(o, o + d, color=S.STATUS["critical"] if bad else S.SEQ_BLUE[1],
                               alpha=0.12 if bad else 0.35, lw=0, zorder=0)
                else:
                    ax.axvline(o, color=S.MUTED, lw=0.8, zorder=0)
                ax.text(o, offsets[0] + 1.3 * scale, s[:14], fontsize=7, color=S.INK_2, rotation=0,
                        va="bottom", clip_on=True)
        ax.set_yticks(offsets)
        ax.set_yticklabels([labels[n] for n in names])
        ax.set_ylim(offsets[-1] - 1.6 * scale, offsets[0] + 2.0 * scale)
        ax.set_xlim(t[0] if len(t) else start, t[-1] if len(t) else start + duration)
        ax.set_xlabel("Time (s)")
        ax.spines["left"].set_visible(False)
        ax.tick_params(axis="y", length=0)
        # scale bar
        x0 = t[-1] - 0.02 * duration if len(t) else 0
        y0 = offsets[-1] - 1.3 * scale
        ax.plot([x0, x0], [y0, y0 + scale], color=S.INK, lw=1.5, clip_on=False)
        ax.text(x0 - 0.005 * duration, y0 + scale / 2, f"{scale:.3g} µV", ha="right", va="center",
                fontsize=7, color=S.INK_2)
        ax.set_title(title or f"{raw.filename.split('/')[-1] if raw.filename else 'EEG'}  "
                              f"{start:g}-{start + duration:g} s", loc="left")
    return S.finish(fig, show)
