"""Low-level spectral estimators shared by several modules (numpy/scipy only)."""

from __future__ import annotations

import numpy as np
from scipy import signal
from scipy.integrate import trapezoid


def welch_psd(x: np.ndarray, sfreq: float, window_sec: float = 2.0, overlap: float = 0.5,
              fmin: float = 0.0, fmax: float | None = None, average: str = "mean",
              window: str = "hann") -> tuple[np.ndarray, np.ndarray]:
    """Welch power spectral density of ``x`` (..., n_samples) in unit²/Hz."""
    x = np.asarray(x, dtype=float)
    n = x.shape[-1]
    nperseg = int(max(8, min(n, round(window_sec * sfreq))))
    noverlap = int(round(nperseg * overlap))
    freqs, psd = signal.welch(x, fs=sfreq, window=window, nperseg=nperseg, noverlap=noverlap,
                              axis=-1, average=average, detrend="constant")
    fmax = sfreq / 2 if fmax is None else fmax
    keep = (freqs >= fmin) & (freqs <= fmax)
    return freqs[keep], psd[..., keep]


def multitaper_psd(x: np.ndarray, sfreq: float, bandwidth: float | None = None,
                   fmin: float = 0.0, fmax: float | None = None) -> tuple[np.ndarray, np.ndarray]:
    """Multitaper PSD with DPSS (Slepian) tapers, eigenvalue-weighted average.

    ``bandwidth`` is the full frequency smoothing in Hz (default 4 / duration,
    i.e. time-bandwidth product NW = 2).
    """
    x = np.asarray(x, dtype=float)
    n = x.shape[-1]
    duration = n / sfreq
    bandwidth = 4.0 / duration if bandwidth is None else bandwidth
    nw = max(1.0, bandwidth * duration / 2.0)
    k = max(1, int(2 * nw) - 1)
    tapers, eigvals = signal.windows.dpss(n, nw, Kmax=k, return_ratios=True)  # unit energy
    weights = eigvals / eigvals.sum()
    flat = x.reshape(-1, n)
    psd = np.empty((flat.shape[0], n // 2 + 1))
    for i, row in enumerate(flat):  # one signal at a time keeps memory bounded
        spec = np.fft.rfft((row - row.mean()) * tapers, axis=-1)
        psd[i] = weights @ (np.abs(spec) ** 2) / sfreq
    psd = psd.reshape(x.shape[:-1] + (n // 2 + 1,))
    psd[..., 1:] *= 2.0  # one-sided
    if n % 2 == 0:
        psd[..., -1] /= 2.0
    freqs = np.fft.rfftfreq(n, 1.0 / sfreq)
    fmax = sfreq / 2 if fmax is None else fmax
    keep = (freqs >= fmin) & (freqs <= fmax)
    return freqs[keep], psd[..., keep]


def band_power(freqs: np.ndarray, psd: np.ndarray, band: tuple[float, float]) -> np.ndarray:
    """Integrate a PSD over ``band`` (inclusive) with the trapezoidal rule."""
    lo, hi = band
    mask = (freqs >= lo) & (freqs <= hi)
    if mask.sum() < 2:
        # Band narrower than the resolution: interpolate the PSD onto a fine grid.
        grid = np.linspace(lo, hi, 16)
        vals = np.apply_along_axis(lambda p: np.interp(grid, freqs, p), -1, psd)
        return trapezoid(vals, grid, axis=-1)
    return trapezoid(psd[..., mask], freqs[mask], axis=-1)
