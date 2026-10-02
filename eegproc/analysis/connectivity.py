"""Functional connectivity between channels.

Methods:

``correlation``  Pearson correlation of band-passed signals
``coherence``    magnitude-squared coherence (Welch cross-spectra)
``imcoh``        imaginary part of coherency (insensitive to volume conduction)
``plv``          phase-locking value (Hilbert phases of band-passed signals)
``pli``          phase-lag index
``wpli``         weighted phase-lag index (Vinck et al., 2011)
``aec``          amplitude-envelope correlation

Zero-lag measures (correlation, coherence, PLV) are inflated by volume
conduction; lagged measures (imcoh, PLI, wPLI) are more conservative.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np
from scipy import signal

from ..preprocessing.filters import filter_data
from ..utils import format_table

METHODS = ("correlation", "coherence", "imcoh", "plv", "pli", "wpli", "aec")


class Connectivity:
    """A symmetric channel x channel connectivity matrix."""

    def __init__(self, matrix: np.ndarray, ch_names: Sequence[str], method: str,
                 band: tuple[float, float] | None, positions: dict | None = None):
        self.matrix = np.asarray(matrix, dtype=float)
        self.ch_names = list(ch_names)
        self.method = method
        self.band = band
        self.positions = dict(positions or {})

    def __repr__(self) -> str:
        band = f"{self.band[0]:g}-{self.band[1]:g} Hz" if self.band else "broadband"
        return f"<Connectivity {self.method} | {band} | {len(self.ch_names)} channels>"

    def node_strength(self) -> np.ndarray:
        """Sum of |connections| per channel (how much a channel is a hub)."""
        m = np.abs(self.matrix.copy())
        np.fill_diagonal(m, 0)
        return m.sum(axis=1) / max(1, len(m) - 1)

    def strongest(self, n: int = 10) -> list[dict]:
        m = self.matrix
        iu = np.triu_indices(len(m), 1)
        vals = m[iu]
        order = np.argsort(np.abs(vals))[::-1][:n]
        return [{"channel_a": self.ch_names[iu[0][k]], "channel_b": self.ch_names[iu[1][k]],
                 "value": float(vals[k])} for k in order]

    def summary(self, n: int = 10) -> str:
        return format_table(self.strongest(n))

    def plot(self, **kwargs):
        from ..viz.connectivity import plot_connectivity

        return plot_connectivity(self, **kwargs)


def spectral_connectivity(x: np.ndarray, sfreq: float, band: tuple[float, float],
                          method: str = "coherence", window_sec: float = 2.0,
                          overlap: float = 0.5, chunk: int = 32) -> np.ndarray:
    """Cross-spectral measures averaged over the frequency band.

    Segments are processed in chunks and only the band's frequency bins are
    kept, so memory does not grow with the recording length.
    """
    n_ch, n = x.shape
    nper = int(max(8, min(n, round(window_sec * sfreq))))
    step = max(1, int(round(nper * (1 - overlap))))
    starts = np.arange(0, n - nper + 1, step)
    freqs = np.fft.rfftfreq(nper, 1 / sfreq)
    fmask = (freqs >= band[0]) & (freqs <= band[1])
    if not fmask.any():
        raise ValueError(f"No frequencies in band {band} at this resolution")
    win = signal.windows.hann(nper, sym=False)
    n_f = int(fmask.sum())
    acc = np.zeros((n_f, n_ch, n_ch), dtype=complex if method in ("coherence", "imcoh") else float)
    acc_abs = np.zeros((n_f, n_ch, n_ch)) if method == "wpli" else None
    for c0 in range(0, len(starts), chunk):
        segs = np.stack([x[:, s: s + nper] for s in starts[c0: c0 + chunk]])
        segs = segs - segs.mean(axis=-1, keepdims=True)
        spec = np.fft.rfft(segs * win, axis=-1)[:, :, fmask]  # (seg, ch, f)
        cross = np.einsum("sif,sjf->fsij", spec, np.conj(spec))  # (f, seg, ch, ch)
        if method in ("coherence", "imcoh"):
            acc += cross.sum(axis=1)
        elif method == "pli":
            acc += np.sign(np.imag(cross)).sum(axis=1)
        elif method == "wpli":
            im = np.imag(cross)
            acc += im.sum(axis=1)
            acc_abs += np.abs(im).sum(axis=1)
        else:
            raise ValueError(method)
    n_seg = len(starts)
    if method in ("coherence", "imcoh"):
        sxy = acc / n_seg
        auto = np.real(np.einsum("fii->fi", sxy))
        norm = np.sqrt(auto[:, :, None] * auto[:, None, :])
        coh = sxy / np.where(norm > 0, norm, 1)
        per_f = np.abs(coh) ** 2 if method == "coherence" else np.abs(np.imag(coh))
    elif method == "pli":
        per_f = np.abs(acc / n_seg)
    else:
        per_f = np.abs(acc) / np.where(acc_abs > 0, acc_abs, 1)
    out = per_f.mean(axis=0)
    np.fill_diagonal(out, 1.0 if method == "coherence" else 0.0)
    return out


def compute_connectivity(inst, method: str = "coherence", band: tuple[float, float] = (8.0, 13.0),
                         picks="eeg", exclude_bads: bool = True, window_sec: float = 2.0,
                         tmin: float | None = None, tmax: float | None = None) -> Connectivity:
    """Channel-by-channel connectivity of a :class:`RawEEG` in a frequency band."""
    if method not in METHODS:
        raise ValueError(f"method must be one of {METHODS}")
    idx = inst.pick_indices(picks, exclude_bads=exclude_bads)
    names = [inst.ch_names[i] for i in idx]
    x = inst.get_data(idx, tmin=tmin, tmax=tmax)
    sfreq = inst.sfreq
    if method in ("coherence", "imcoh", "pli", "wpli"):
        mat = spectral_connectivity(x, sfreq, band, method, window_sec)
    else:
        xf = filter_data(x, sfreq, band[0], band[1])
        if method == "correlation":
            mat = np.corrcoef(xf)
        elif method == "plv":
            phase = np.exp(1j * np.angle(signal.hilbert(xf, axis=-1)))
            mat = np.abs(phase @ phase.conj().T) / phase.shape[1]
        elif method == "aec":
            env = np.abs(signal.hilbert(xf, axis=-1))
            mat = np.corrcoef(env)
    positions = {n: inst.positions[n] for n in names if n in inst.positions}
    return Connectivity(mat, names, method, band, positions)
