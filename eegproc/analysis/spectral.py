"""Spectral analysis: PSD, band power, peak frequencies, 1/f fits, time-frequency."""

from __future__ import annotations

from typing import Sequence

import numpy as np
from scipy import signal

from .._spectral_core import band_power as _band_power
from .._spectral_core import multitaper_psd, welch_psd
from ..utils import format_table

# Canonical EEG frequency bands (Hz).
BANDS: dict[str, tuple[float, float]] = {
    "delta": (1.0, 4.0),
    "theta": (4.0, 8.0),
    "alpha": (8.0, 13.0),
    "beta": (13.0, 30.0),
    "gamma": (30.0, 45.0),
}


class Spectrum:
    """Power spectral density per channel (µV²/Hz for voltage channels)."""

    def __init__(self, freqs: np.ndarray, psd: np.ndarray, ch_names: Sequence[str],
                 method: str = "welch", positions: dict | None = None):
        self.freqs = np.asarray(freqs, dtype=float)
        self.data = np.atleast_2d(np.asarray(psd, dtype=float))
        self.ch_names = list(ch_names)
        self.method = method
        self.positions = dict(positions or {})

    def __repr__(self) -> str:
        return (f"<Spectrum ({self.method}) | {len(self.ch_names)} channels | "
                f"{self.freqs[0]:.2f}-{self.freqs[-1]:.2f} Hz, df={self.resolution:.3g} Hz>")

    @property
    def resolution(self) -> float:
        return float(self.freqs[1] - self.freqs[0]) if len(self.freqs) > 1 else float("nan")

    def get(self, ch_name: str) -> np.ndarray:
        return self.data[self.ch_names.index(ch_name)]

    def to_db(self) -> np.ndarray:
        return 10 * np.log10(np.maximum(self.data, 1e-20))

    def band_power(self, bands: dict[str, tuple[float, float]] | None = None,
                   relative: bool = False, total_range: tuple[float, float] | None = None) -> dict[str, np.ndarray]:
        """Integrated power per band -> ``{band: array(n_channels)}`` (µV²).

        With ``relative=True`` each band is divided by the power in
        ``total_range`` (default: from the lowest to the highest band edge).
        """
        bands = bands or BANDS
        out = {name: _band_power(self.freqs, self.data, rng) for name, rng in bands.items()}
        if relative:
            lo = min(b[0] for b in bands.values()) if total_range is None else total_range[0]
            hi = max(b[1] for b in bands.values()) if total_range is None else total_range[1]
            total = _band_power(self.freqs, self.data, (lo, min(hi, self.freqs[-1])))
            out = {k: v / np.where(total > 0, total, np.nan) for k, v in out.items()}
        return out

    def band_power_table(self, bands=None, relative: bool = True) -> list[dict]:
        bp = self.band_power(bands, relative=relative)
        return [{"channel": c, **{b: float(v[i]) for b, v in bp.items()}} for i, c in enumerate(self.ch_names)]

    def peak_frequency(self, fmin: float = 7.0, fmax: float = 14.0, method: str = "max") -> np.ndarray:
        """Peak (``'max'``) or centre-of-gravity (``'cog'``) frequency in a band."""
        mask = (self.freqs >= fmin) & (self.freqs <= fmax)
        f, p = self.freqs[mask], self.data[:, mask]
        if method == "cog":
            return (p * f).sum(axis=1) / p.sum(axis=1)
        return f[np.argmax(p, axis=1)]

    def spectral_edge(self, edge: float = 0.95, fmax: float = 45.0) -> np.ndarray:
        """Frequency below which ``edge`` of the power (up to ``fmax``) lies."""
        mask = self.freqs <= fmax
        c = np.cumsum(self.data[:, mask], axis=1)
        c /= c[:, -1:]
        return np.array([self.freqs[mask][np.searchsorted(row, edge)] for row in c])

    def spectral_entropy(self, fmin: float = 1.0, fmax: float = 45.0) -> np.ndarray:
        """Normalised Shannon entropy of the spectrum (0 = one peak, 1 = white noise)."""
        mask = (self.freqs >= fmin) & (self.freqs <= fmax)
        p = self.data[:, mask]
        p = p / p.sum(axis=1, keepdims=True)
        h = -(p * np.log2(np.maximum(p, 1e-30))).sum(axis=1)
        return h / np.log2(mask.sum())

    def aperiodic(self, fmin: float = 2.0, fmax: float = 40.0) -> tuple[np.ndarray, np.ndarray]:
        """1/f fit ``log10 P = offset - exponent * log10 f`` -> ``(offset, exponent)``."""
        return fit_aperiodic(self.freqs, self.data, fmin, fmax)

    def average(self) -> "Spectrum":
        return Spectrum(self.freqs, self.data.mean(axis=0, keepdims=True), ["average"], self.method)

    def plot(self, **kwargs):
        from ..viz.spectra import plot_spectrum

        return plot_spectrum(self, **kwargs)


def compute_psd(inst, picks="data", method: str = "welch", fmin: float = 0.0,
                fmax: float | None = None, window_sec: float = 2.0, overlap: float = 0.5,
                average: str = "mean", bandwidth: float | None = None,
                reject_by_annotation: bool = True, sfreq: float | None = None,
                ch_names: Sequence[str] | None = None) -> Spectrum:
    """Power spectral density of a :class:`RawEEG`, :class:`Epochs` or array.

    Parameters
    ----------
    method : ``'welch'`` | ``'multitaper'`` | ``'periodogram'``
    window_sec, overlap : Welch segment length (s) and overlap fraction.
    average : ``'mean'`` or ``'median'`` (median is robust to artifacts).
    reject_by_annotation : skip ``BAD*`` segments of raw data.
    """
    from ..core.epochs import Epochs
    from ..core.raw import RawEEG

    positions = {}
    if isinstance(inst, RawEEG):
        idx = inst.pick_indices(picks)
        x = inst.data[idx]
        names = [inst.ch_names[i] for i in idx]
        fs = inst.sfreq
        positions = {n: inst.positions[n] for n in names if n in inst.positions}
        if reject_by_annotation and len(inst.annotations):
            good = ~inst.annotations.bad_mask(inst.n_times, inst.sfreq)
            if good.sum() >= window_sec * fs and not good.all():
                return _psd_segments(x, good, fs, names, method, fmin, fmax, window_sec, overlap,
                                     average, bandwidth, positions)
    elif isinstance(inst, Epochs):
        idx = inst._pick(None if picks in ("data", None) else picks)
        x = inst.data[:, idx]
        names = [inst.ch_names[i] for i in idx]
        fs = inst.sfreq
        positions = {n: inst.positions[n] for n in names if n in inst.positions}
    else:
        x = np.atleast_2d(np.asarray(inst, dtype=float))
        if sfreq is None:
            raise ValueError("sfreq is required for array input")
        fs = sfreq
        names = list(ch_names) if ch_names is not None else [f"Ch{i + 1}" for i in range(x.shape[-2])]
    freqs, p = _estimate(x, fs, method, fmin, fmax, window_sec, overlap, average, bandwidth)
    if p.ndim == 3:  # epochs: average over epochs
        p = p.mean(axis=0)
    return Spectrum(freqs, p, names, method, positions)


def _estimate(x, fs, method, fmin, fmax, window_sec, overlap, average, bandwidth):
    if method == "welch":
        return welch_psd(x, fs, window_sec=min(window_sec, x.shape[-1] / fs), overlap=overlap,
                         fmin=fmin, fmax=fmax, average=average)
    if method == "multitaper":
        return multitaper_psd(x, fs, bandwidth=bandwidth, fmin=fmin, fmax=fmax)
    if method == "periodogram":
        freqs, p = signal.periodogram(x, fs=fs, window="hann", axis=-1)
        fmax = fs / 2 if fmax is None else fmax
        keep = (freqs >= fmin) & (freqs <= fmax)
        return freqs[keep], p[..., keep]
    raise ValueError("method must be 'welch', 'multitaper' or 'periodogram'")


def _psd_segments(x, good, fs, names, method, fmin, fmax, window_sec, overlap, average, bandwidth,
                  positions) -> Spectrum:
    """PSD averaged over the clean stretches of a recording (weighted by length)."""
    edges = np.diff(np.concatenate([[0], good.astype(np.int8), [0]]))
    starts, ends = np.flatnonzero(edges == 1), np.flatnonzero(edges == -1)
    min_len = int(window_sec * fs)
    acc, weight, freqs = None, 0, None
    for a, b in zip(starts, ends):
        if b - a < min_len:
            continue
        f, p = _estimate(x[:, a:b], fs, method, fmin, fmax, window_sec, overlap, average, bandwidth)
        if freqs is None:
            freqs, acc = f, np.zeros_like(p)
        if p.shape != acc.shape:
            continue
        acc += p * (b - a)
        weight += b - a
    if acc is None:
        f, p = _estimate(x, fs, method, fmin, fmax, window_sec, overlap, average, bandwidth)
        return Spectrum(f, p, names, method, positions)
    return Spectrum(freqs, acc / weight, names, method, positions)


def fit_aperiodic(freqs: np.ndarray, psd: np.ndarray, fmin: float = 2.0, fmax: float = 40.0,
                  n_iter: int = 2) -> tuple[np.ndarray, np.ndarray]:
    """Robust 1/f fit (FOOOF-like, without a knee): returns (offset, exponent) per channel."""
    psd = np.atleast_2d(psd)
    mask = (freqs >= fmin) & (freqs <= fmax) & (freqs > 0)
    lf = np.log10(freqs[mask])
    offsets, exps = np.empty(psd.shape[0]), np.empty(psd.shape[0])
    for k, row in enumerate(psd[:, mask]):
        lp = np.log10(np.maximum(row, 1e-20))
        keep = np.ones_like(lp, dtype=bool)
        for _ in range(n_iter + 1):
            slope, icpt = np.polyfit(lf[keep], lp[keep], 1)
            resid = lp - (slope * lf + icpt)
            thr = np.percentile(resid, 60)
            new_keep = resid <= max(thr, 0.0) + 0.02
            if new_keep.sum() < 4 or np.array_equal(new_keep, keep):
                break
            keep = new_keep
        offsets[k], exps[k] = icpt, -slope
    return offsets, exps


def band_power_table(raw, bands: dict | None = None, relative: bool = True, **psd_kwargs) -> list[dict]:
    """Band power per channel as a list of rows (for printing or CSV)."""
    spec = compute_psd(raw, picks="eeg", **psd_kwargs)
    return spec.band_power_table(bands, relative=relative)


def spectrogram(x: np.ndarray, sfreq: float, window_sec: float = 2.0, overlap: float = 0.9,
                fmax: float | None = None) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Short-time Fourier power: returns ``(times, freqs, power[freq, time])``."""
    nperseg = int(max(8, min(len(x), round(window_sec * sfreq))))
    f, t, sxx = signal.spectrogram(np.asarray(x, dtype=float), fs=sfreq, window="hann",
                                   nperseg=nperseg, noverlap=int(overlap * nperseg), detrend="constant")
    if fmax is not None:
        keep = f <= fmax
        f, sxx = f[keep], sxx[keep]
    return t, f, sxx


def morlet_power(x: np.ndarray, sfreq: float, freqs: Sequence[float], n_cycles: float | Sequence[float] = 7.0,
                 output: str = "power") -> np.ndarray:
    """Time-frequency decomposition with complex Morlet wavelets.

    ``x`` has shape (..., n_times); the result has shape (..., n_freqs, n_times).
    ``output='power'`` returns |W|² (a sinusoid of amplitude A gives A²),
    ``'complex'`` the analytic coefficients, ``'phase'`` the angle.
    """
    x = np.asarray(x, dtype=float)
    freqs = np.atleast_1d(np.asarray(freqs, dtype=float))
    cycles = np.broadcast_to(np.asarray(n_cycles, dtype=float), freqs.shape)
    n = x.shape[-1]
    out = np.empty(x.shape[:-1] + (len(freqs), n), dtype=complex)
    for k, (f, c) in enumerate(zip(freqs, cycles)):
        sigma_t = c / (2 * np.pi * f)
        tw = np.arange(-3.5 * sigma_t, 3.5 * sigma_t + 1 / sfreq, 1 / sfreq)
        gauss = np.exp(-(tw ** 2) / (2 * sigma_t ** 2))
        # scaled so a sinusoid of amplitude A gives |W| = A (power A^2)
        wavelet = np.exp(2j * np.pi * f * tw) * gauss / (gauss.sum() / 2)
        conv = signal.fftconvolve(x, wavelet.reshape((1,) * (x.ndim - 1) + (-1,)), mode="same", axes=-1)
        out[..., k, :] = conv[..., :n]
    if output == "complex":
        return out
    if output == "phase":
        return np.angle(out)
    return np.abs(out) ** 2


def band_envelope(x: np.ndarray, sfreq: float, band: tuple[float, float]) -> np.ndarray:
    """Instantaneous amplitude (Hilbert envelope) in a frequency band."""
    from ..preprocessing.filters import filter_data

    return np.abs(signal.hilbert(filter_data(x, sfreq, band[0], band[1]), axis=-1))


def summarize_bands(spec: Spectrum, bands: dict | None = None) -> str:
    """Text table: for each band, the channel with the most (relative) power."""
    bands = bands or BANDS
    rel = spec.band_power(bands, relative=True)
    ab = spec.band_power(bands, relative=False)
    rows = []
    for b in bands:
        i = int(np.nanargmax(rel[b]))
        rows.append({"band": b, "range_hz": f"{bands[b][0]:g}-{bands[b][1]:g}",
                     "leading_channel": spec.ch_names[i], "relative_power": float(rel[b][i]),
                     "absolute_uV2": float(ab[b][i]), "median_relative": float(np.nanmedian(rel[b]))})
    return format_table(rows)
