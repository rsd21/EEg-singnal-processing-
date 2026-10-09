"""Power spectra, band power and spectrograms."""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np

from ..analysis.spectral import BANDS, Spectrum, compute_psd, spectrogram
from . import _style as S


def plot_spectrum(spec: Spectrum, fmax: float | None = 60.0, bands: dict | None = BANDS,
                  highlight: dict | None = None, db: bool = True, average: bool = True, ax=None,
                  title: str | None = None, highlight_labels: dict | None = None, show: bool = False):
    """Every channel as a thin grey line, the average in blue, chosen channels coloured.

    ``highlight`` maps channel -> colour (bad channels are highlighted by
    :func:`plot_psd` automatically); channels sharing a colour share one
    legend entry, optionally prefixed by ``highlight_labels[colour]``.
    """
    fmax = fmax or spec.freqs[-1]
    mask = spec.freqs <= fmax
    f = spec.freqs[mask]
    p = spec.data[:, mask]
    y = 10 * np.log10(np.maximum(p, 1e-20)) if db else p
    highlight = highlight or {}
    with S.style():
        if ax is None:
            fig, ax = plt.subplots(figsize=(8, 4), layout="constrained")
        fig = ax.figure
        if bands:
            for k, (name, (lo, hi)) in enumerate(bands.items()):
                if lo >= fmax:
                    continue
                if k % 2 == 0:
                    ax.axvspan(lo, min(hi, fmax), color=S.GRID, alpha=0.45, lw=0, zorder=0)
                ax.text((lo + min(hi, fmax)) / 2, 1.0, name, transform=ax.get_xaxis_transform(),
                        ha="center", va="bottom", fontsize=7, color=S.MUTED)
        for k, n in enumerate(spec.ch_names):
            if n in highlight:
                continue
            ax.plot(f, y[k], color=S.AXIS, lw=0.7, zorder=1)
        if average and len(spec.ch_names) > 1:
            good = [k for k, n in enumerate(spec.ch_names) if n not in highlight]
            if good:
                mean = 10 * np.log10(np.maximum(p[good].mean(axis=0), 1e-20)) if db else p[good].mean(axis=0)
                ax.plot(f, mean, color=S.CATEGORICAL[0], lw=2, zorder=3, label="average")
        groups: dict[str, list[str]] = {}
        for n, c in highlight.items():
            if n in spec.ch_names:
                groups.setdefault(c, []).append(n)
        for c, members in groups.items():
            label = ", ".join(members[:4]) + (f" +{len(members) - 4}" if len(members) > 4 else "")
            if highlight_labels and c in highlight_labels:
                label = f"{highlight_labels[c]}: {label}"
            for j, n in enumerate(members):
                ax.plot(f, y[spec.ch_names.index(n)], color=c, lw=1.4, zorder=4,
                        label=label if j == 0 else None)
        ax.set_xlim(f[0], fmax)
        ax.set_xlabel("Frequency (Hz)")
        ax.set_ylabel("Power (dB re 1 µV²/Hz)" if db else "Power (µV²/Hz)")
        S.grid_y(ax)
        if highlight or average:
            ax.legend(loc="upper right")
        ax.set_title(title or f"Power spectral density ({spec.method})", loc="left", pad=14)
    return S.finish(fig, show)


def plot_psd(raw, picks="eeg", fmax: float | None = 60.0, highlight_bads: bool = True,
             highlight: dict | None = None, show: bool = False, **psd_kwargs):
    """PSD of a :class:`RawEEG`; bad channels are drawn in red."""
    spec = compute_psd(raw, picks=picks, **psd_kwargs)
    hl = {}
    if highlight_bads:
        hl.update({b: S.STATUS["critical"] for b in raw.bads if b in spec.ch_names})
    hl.update(highlight or {})
    return plot_spectrum(spec, fmax=fmax, highlight=hl, show=show,
                         highlight_labels={S.STATUS["critical"]: "bad"})


def plot_band_power(raw, bands: dict | None = None, picks="eeg", relative: bool = True, show: bool = False):
    """Heatmap of band power: channels x bands (one-hue ramp, per-band columns)."""
    bands = bands or BANDS
    spec = compute_psd(raw, picks=picks, window_sec=2.0, average="median")
    bands = {k: v for k, v in bands.items() if v[0] < spec.freqs[-1]}
    bp = spec.band_power(bands, relative=relative)
    mat = np.column_stack([bp[b] for b in bands]) * (100 if relative else 1)
    with S.style():
        fig, ax = plt.subplots(figsize=(1.2 * len(bands) + 2.5, 0.22 * len(spec.ch_names) + 1.5))
        im = ax.imshow(mat, aspect="auto", cmap=S.CMAP_SEQ)
        ax.set_xticks(range(len(bands)))
        ax.set_xticklabels([f"{b}\n{lo:g}-{hi:g} Hz" for b, (lo, hi) in bands.items()])
        ax.set_yticks(range(len(spec.ch_names)))
        ax.set_yticklabels(spec.ch_names)
        for j in range(mat.shape[1]):
            i = int(np.nanargmax(mat[:, j]))
            ax.scatter(j, i, marker="o", s=26, facecolors="none", edgecolors=S.INK, linewidths=1.2)
        ax.tick_params(length=0)
        for spine in ax.spines.values():
            spine.set_visible(False)
        cb = fig.colorbar(im, ax=ax, fraction=0.05, pad=0.02)
        cb.outline.set_visible(False)
        cb.set_label("% of 1-45 Hz power" if relative else "µV²", fontsize=8)
        ax.set_title("Band power per channel (circle = strongest channel)", loc="left")
    return S.finish(fig, show)


def plot_spectrogram(raw, channel: str, fmax: float = 45.0, window_sec: float = 2.0, show: bool = False):
    """Time-frequency power of one channel (dB)."""
    x = raw[channel]
    t, f, sxx = spectrogram(x, raw.sfreq, window_sec=window_sec, fmax=fmax)
    with S.style():
        fig, ax = plt.subplots(figsize=(10, 3.2))
        db = 10 * np.log10(np.maximum(sxx, 1e-20))
        vmin, vmax = np.percentile(db, [2, 99.5])
        mesh = ax.pcolormesh(t, f, db, shading="auto", cmap=S.CMAP_SEQ, vmin=vmin, vmax=vmax)
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Frequency (Hz)")
        cb = fig.colorbar(mesh, ax=ax, fraction=0.03, pad=0.01)
        cb.outline.set_visible(False)
        cb.set_label("dB re 1 µV²/Hz", fontsize=8)
        ax.set_title(f"Spectrogram · {channel}", loc="left")
    return S.finish(fig, show)
