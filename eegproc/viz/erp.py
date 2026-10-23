"""Event-related potentials and event-related (de)synchronisation."""

from __future__ import annotations

from typing import Iterable, Sequence

import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import uniform_filter1d

from . import _style as S
from .topomap import plot_topomap


def plot_evoked(evoked, picks: Sequence[str] | None = None, gfp: bool = True, ax=None,
                title: str | None = None, show: bool = False):
    """Butterfly plot (all channels grey) with chosen channels and GFP highlighted."""
    picks = list(picks or [])
    with S.style():
        if ax is None:
            fig, ax = plt.subplots(figsize=(8, 3.8), layout="constrained")
        fig = ax.figure
        t = evoked.times * 1000
        for k, n in enumerate(evoked.ch_names):
            if evoked.ch_types[k] != "eeg" or n in picks:
                continue
            ax.plot(t, evoked.data[k], color=S.AXIS, lw=0.7)
        for j, n in enumerate(picks):
            ax.plot(t, evoked.data[evoked.ch_names.index(n)], color=S.CATEGORICAL[j % 8], lw=2, label=n)
        if gfp:
            ax.plot(t, evoked.gfp(), color=S.INK, lw=1.2, label="GFP")
        ax.axvline(0, color=S.INK_2, lw=0.8)
        ax.axhline(0, color=S.AXIS, lw=0.8)
        ax.set_xlim(t[0], t[-1])
        ax.set_xlabel("Time (ms)")
        ax.set_ylabel("Amplitude (µV)")
        S.grid_y(ax)
        ax.legend(loc="upper left", ncol=min(5, len(picks) + 1))
        ax.set_title(title or f"Evoked response · {evoked.comment} (n={evoked.nave})", loc="left")
    return S.finish(fig, show)


def plot_evoked_topomaps(evoked, times: float | Iterable[float] | str = "peaks", n_peaks: int = 4,
                         show: bool = False):
    """Scalp maps at chosen latencies (``'peaks'`` = local maxima of the GFP)."""
    if isinstance(times, str):
        from scipy.signal import find_peaks

        g = evoked.gfp()
        post = evoked.times >= 0
        pk, prop = find_peaks(np.where(post, g, 0), distance=max(1, int(0.05 * evoked.sfreq)))
        pk = pk[np.argsort(g[pk])[::-1][:n_peaks]] if len(pk) else [int(np.argmax(g))]
        times = sorted(evoked.times[np.asarray(pk)])
    times = list(np.atleast_1d(times))
    lim = float(np.max(np.abs(evoked.data[[i for i, t in enumerate(evoked.ch_types) if t == "eeg"]])))
    with S.style():
        fig, axes = plt.subplots(1, len(times), figsize=(2.8 * len(times), 3.2), layout="constrained")
        axes = np.atleast_1d(axes)
        for ax, t in zip(axes, times):
            i = int(np.argmin(np.abs(evoked.times - t)))
            plot_topomap(evoked.data[:, i], evoked.ch_names, evoked.positions, ax=ax, vmin=-lim, vmax=lim,
                         diverging=True, title=f"{1000 * evoked.times[i]:.0f} ms", colorbar=ax is axes[-1],
                         cbar_label="µV")
    return S.finish(fig, show)


def plot_erds_timecourse(epochs, channels: Sequence[str], band=(8.0, 13.0), baseline=(-1.0, 0.0),
                         by_condition: bool = True, show: bool = False):
    """Band power over time relative to baseline (%), per condition."""
    from ..preprocessing.filters import filter_data

    conds = list(epochs.event_id) if by_condition else [None]
    with S.style():
        fig, axes = plt.subplots(1, len(channels), figsize=(4.2 * len(channels), 3.2), sharey=True,
                                 layout="constrained")
        axes = np.atleast_1d(axes)
        for ax, ch in zip(axes, channels):
            ci = epochs.ch_names.index(ch)
            for j, cond in enumerate(conds):
                ep = epochs[cond] if cond else epochs
                x = filter_data(ep.data[:, ci], ep.sfreq, *band) ** 2
                p = x.mean(axis=0)
                p = uniform_filter1d(p, max(1, int(0.25 * ep.sfreq)), mode="nearest")
                b = (ep.times >= baseline[0]) & (ep.times <= baseline[1])
                pct = 100 * (p - p[b].mean()) / p[b].mean()
                ax.plot(ep.times, pct, color=S.CATEGORICAL[j % 8], lw=2, label=cond or "all")
            ax.axhline(0, color=S.AXIS, lw=0.8)
            ax.axvline(0, color=S.INK_2, lw=0.8)
            ax.set_title(f"{ch} · {band[0]:g}-{band[1]:g} Hz", loc="left")
            ax.set_xlabel("Time (s)")
            S.grid_y(ax)
        axes[0].set_ylabel("Power change vs baseline (%)")
        axes[0].legend(loc="lower left")
    return S.finish(fig, show)
