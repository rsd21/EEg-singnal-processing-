"""Per-channel signal-quality metrics.

Every function takes a ``(n_channels, n_samples)`` array in microvolts and
returns one value per channel. They are used by
:func:`eegproc.preprocessing.find_bad_channels` (PREP-style detection) and
by :func:`eegproc.quality.assess_channel_quality` (0-100 quality score).
"""

from __future__ import annotations

import numpy as np
from scipy import stats

from .._spectral_core import welch_psd
from ..utils import mad, robust_zscore, sliding_windows


def robust_std(x: np.ndarray) -> np.ndarray:
    """0.7413 x inter-quartile range: the std of a Gaussian, insensitive to spikes."""
    q75, q25 = np.percentile(x, [75, 25], axis=-1)
    return 0.7413 * (q75 - q25)


def _windows(x: np.ndarray, sfreq: float, window: float, max_windows: int | None = None) -> np.ndarray:
    win = max(2, int(round(window * sfreq)))
    w = sliding_windows(x, win)  # (n_ch, n_win, win)
    if max_windows is not None and w.shape[1] > max_windows:
        keep = np.linspace(0, w.shape[1] - 1, max_windows).astype(int)
        w = w[:, keep]
    return w


def flat_fraction(x: np.ndarray, sfreq: float, window: float = 1.0, threshold: float = 0.5) -> np.ndarray:
    """Fraction of windows whose standard deviation is below ``threshold`` µV."""
    w = _windows(x, sfreq, window)
    if w.shape[1] == 0:
        return (np.std(x, axis=-1) < threshold).astype(float)
    return (w.std(axis=-1) < threshold).mean(axis=1)


def clipping_fraction(x: np.ndarray, min_run: int = 3) -> np.ndarray:
    """Fraction of samples in plateaus at the channel's own min/max (amplifier saturation)."""
    out = np.zeros(x.shape[0])
    for k, row in enumerate(x):
        if row.size < min_run:
            continue
        lo, hi = row.min(), row.max()
        if lo == hi:
            out[k] = 1.0
            continue
        at_edge = (row == lo) | (row == hi)
        if at_edge.sum() < min_run:
            continue
        # count only samples belonging to runs of >= min_run
        edges = np.diff(np.concatenate([[0], at_edge.astype(np.int8), [0]]))
        starts, ends = np.flatnonzero(edges == 1), np.flatnonzero(edges == -1)
        lengths = ends - starts
        out[k] = lengths[lengths >= min_run].sum() / row.size
    return out


def window_correlations(x: np.ndarray, sfreq: float, window: float = 1.0,
                        max_windows: int = 600) -> np.ndarray:
    """Correlation matrices of short windows: ``(n_windows, n_ch, n_ch)``."""
    w = _windows(x, sfreq, window, max_windows)  # (n_ch, n_win, win)
    w = np.transpose(w, (1, 0, 2)).astype(float)
    w = w - w.mean(axis=-1, keepdims=True)
    norm = np.linalg.norm(w, axis=-1, keepdims=True)
    safe = np.where(norm > 1e-12, norm, 1.0)
    z = np.where(norm > 1e-12, w / safe, 0.0)
    return np.einsum("wct,wdt->wcd", z, z)


def max_correlation(corr: np.ndarray) -> np.ndarray:
    """Per window, each channel's largest absolute correlation with another channel."""
    c = np.abs(corr).copy()
    idx = np.arange(c.shape[1])
    c[:, idx, idx] = 0.0
    return c.max(axis=2)  # (n_windows, n_ch)


def neighbor_correlation(corr: np.ndarray, neighbors: list[np.ndarray] | None = None,
                         k: int = 3, top: int = 2) -> np.ndarray:
    """How well each channel agrees with its neighbours.

    For each neighbour the median |r| over windows is taken; the channel's
    value is the mean of its ``top`` best neighbours. Neighbours are the
    spatially nearest electrodes when ``neighbors`` is given, otherwise the
    ``k`` channels it correlates with most. Using the best neighbours means a
    good channel is not penalised because one of its neighbours is broken.
    """
    c = np.abs(corr)
    n_ch = c.shape[1]
    if n_ch < 2:
        return np.ones(n_ch)
    med = np.median(c, axis=0)  # (n_ch, n_ch)
    if neighbors is None:
        tmp = med.copy()
        np.fill_diagonal(tmp, -np.inf)
        neighbors = [np.argsort(tmp[i])[::-1][: min(k, n_ch - 1)] for i in range(n_ch)]
    out = np.empty(n_ch)
    for i, nb in enumerate(neighbors):
        nb = np.asarray(nb, dtype=int)
        if nb.size == 0:
            out[i] = np.nan
            continue
        vals = np.sort(med[i, nb])[::-1]
        out[i] = vals[: min(top, vals.size)].mean()
    return out


def local_kurtosis(x: np.ndarray, neighbors: list[np.ndarray] | None) -> np.ndarray:
    """Excess kurtosis of each channel minus the mean of its neighbours.

    Activity shared with neighbours (eye blinks, true brain transients)
    cancels; transients confined to one electrode (pops, loose contacts)
    remain, so this isolates channel-specific spikes.
    """
    if neighbors is None or x.shape[0] < 3:
        return excess_kurtosis(x)
    resid = np.empty_like(x)
    for i, nb in enumerate(neighbors):
        nb = np.asarray(nb, dtype=int)
        resid[i] = x[i] - (x[nb].mean(axis=0) if nb.size else 0.0)
    return excess_kurtosis(resid)


def top_correlated(corr: np.ndarray, k: int = 4) -> list[np.ndarray]:
    """For each channel, the ``k`` channels it is most correlated with (median over windows)."""
    med = np.median(np.abs(corr), axis=0)
    np.fill_diagonal(med, -np.inf)
    return [np.argsort(med[i])[::-1][: min(k, med.shape[0] - 1)] for i in range(med.shape[0])]


def hf_noise_ratio(x: np.ndarray, sfreq: float, split: float = 50.0) -> np.ndarray:
    """MAD of the content above ``split`` Hz divided by the MAD below it (PREP noisiness)."""
    from ..preprocessing.filters import filter_data

    if sfreq / 2 <= split + 5:
        split = 0.7 * sfreq / 2  # low sampling rate: use the top 30 % of the band
    low = filter_data(x, sfreq, None, split)
    high = x - low
    denom = mad(low, axis=-1)
    return mad(high, axis=-1) / np.where(denom > 1e-12, denom, np.nan)


def psd(x: np.ndarray, sfreq: float, window_sec: float = 4.0) -> tuple[np.ndarray, np.ndarray]:
    return welch_psd(x, sfreq, window_sec=min(window_sec, x.shape[-1] / sfreq), average="median")


def line_noise_db(freqs: np.ndarray, p: np.ndarray, line_freq: float | None) -> np.ndarray:
    """Peak at the mains frequency relative to its neighbourhood (dB)."""
    if not line_freq or line_freq >= freqs[-1] - 3:
        return np.zeros(p.shape[0])
    peak = p[:, np.abs(freqs - line_freq) <= 1.0].max(axis=1)
    flank_mask = (np.abs(freqs - line_freq) > 3) & (np.abs(freqs - line_freq) < 8)
    flank = np.median(p[:, flank_mask], axis=1)
    return 10 * np.log10(np.maximum(peak, 1e-20) / np.maximum(flank, 1e-20))


def band_mean(freqs: np.ndarray, p: np.ndarray, lo: float, hi: float,
              exclude: list[float] | None = None) -> np.ndarray:
    mask = (freqs >= lo) & (freqs <= hi)
    for f0 in exclude or []:
        mask &= np.abs(freqs - f0) > 2.0
    if not mask.any():
        return np.full(p.shape[0], np.nan)
    return p[:, mask].mean(axis=1)


def snr_db(freqs: np.ndarray, p: np.ndarray, sfreq: float, line_freq: float | None = None) -> np.ndarray:
    """Physiological band (1-30 Hz) power density over high-frequency noise floor (dB)."""
    nyq = sfreq / 2
    hi_lo, hi_hi = (45.0, min(95.0, nyq - 2)) if nyq > 55 else (0.7 * nyq, nyq - 1)
    harmonics = [line_freq * m for m in range(1, 5)] if line_freq else []
    sig = band_mean(freqs, p, 1.0, 30.0)
    noise = band_mean(freqs, p, hi_lo, hi_hi, exclude=harmonics)
    return 10 * np.log10(np.maximum(sig, 1e-20) / np.maximum(noise, 1e-20))


def drift_db(x: np.ndarray, x_band: np.ndarray, x_high: np.ndarray) -> np.ndarray:
    """Size of slow (<1 Hz) fluctuations relative to the 1-40 Hz signal, in dB.

    ``x`` is the unfiltered signal, ``x_high`` the same signal high-passed at
    1 Hz and ``x_band`` the 1-40 Hz band. Measured in the time domain because
    drifts slower than the spectral window would be invisible in a PSD.
    """
    slow = x - x_high
    ratio = robust_std(slow) / np.maximum(robust_std(x_band), 1e-9)
    return 20 * np.log10(np.maximum(ratio, 1e-9))


def aperiodic_exponent(freqs: np.ndarray, p: np.ndarray, fmin: float = 2.0, fmax: float = 40.0,
                       line_freq: float | None = None) -> np.ndarray:
    """Slope of the 1/f background in log-log space (robust to oscillatory peaks).

    Fits ``log10 P = b - chi * log10 f``; peaks above the first fit are
    excluded and the line refitted. Returns ``chi`` (typical EEG: 1-3; values
    near 0 mean a white-noise-like spectrum).
    """
    mask = (freqs >= fmin) & (freqs <= min(fmax, freqs[-1]))
    if line_freq:
        mask &= np.abs(freqs - line_freq) > 2
    if mask.sum() < 4:
        return np.full(p.shape[0], np.nan)
    lf = np.log10(freqs[mask])
    out = np.empty(p.shape[0])
    for k, row in enumerate(p[:, mask]):
        lp = np.log10(np.maximum(row, 1e-20))
        slope, icpt = np.polyfit(lf, lp, 1)
        resid = lp - (slope * lf + icpt)
        keep = resid < np.percentile(resid, 75) + 0.05
        if keep.sum() >= 4:
            slope, icpt = np.polyfit(lf[keep], lp[keep], 1)
        out[k] = -slope
    return out


def alpha_peak(freqs: np.ndarray, p: np.ndarray, band=(7.0, 14.0)) -> tuple[np.ndarray, np.ndarray]:
    """Peak frequency in the alpha band and its height above the 1/f fit (dB)."""
    mask = (freqs >= band[0]) & (freqs <= band[1])
    fit_mask = ((freqs >= 2) & (freqs < band[0])) | ((freqs > band[1]) & (freqs <= 30))
    peak_f = np.full(p.shape[0], np.nan)
    peak_db = np.zeros(p.shape[0])
    if mask.sum() < 2 or fit_mask.sum() < 3:
        return peak_f, peak_db
    lf = np.log10(freqs[fit_mask])
    for k, row in enumerate(p):
        slope, icpt = np.polyfit(lf, np.log10(np.maximum(row[fit_mask], 1e-20)), 1)
        base = slope * np.log10(freqs[mask]) + icpt
        excess = np.log10(np.maximum(row[mask], 1e-20)) - base
        j = int(np.argmax(excess))
        peak_f[k] = freqs[mask][j]
        peak_db[k] = 10 * excess[j]
    return peak_f, peak_db


def amplitude_instability(x: np.ndarray, sfreq: float, window: float = 1.0) -> np.ndarray:
    """Spread (MAD) of log10 window std: high for intermittent contact."""
    w = _windows(x, sfreq, window)
    if w.shape[1] < 3:
        return np.zeros(x.shape[0])
    s = np.log10(np.maximum(w.std(axis=-1), 1e-6))
    return mad(s, axis=1)


def bad_window_fraction(x: np.ndarray, sfreq: float, window: float = 1.0,
                        ptp_threshold: float = 200.0, z_threshold: float = 5.0) -> np.ndarray:
    """Fraction of windows where a channel has an artifact of its own.

    A window counts when the channel exceeds ``ptp_threshold`` µV peak-to-peak,
    or its window amplitude is a robust outlier (z > ``z_threshold``) compared
    with the other channels in the same window.
    """
    w = _windows(x, sfreq, window)
    if w.shape[1] == 0:
        return np.zeros(x.shape[0])
    ptp = w.max(axis=-1) - w.min(axis=-1)
    std = np.log10(np.maximum(w.std(axis=-1), 1e-6))
    bad = ptp > ptp_threshold
    if x.shape[0] >= 4:
        med = np.median(std, axis=0, keepdims=True)
        spread = mad(std, axis=0)
        spread = np.where(spread > 1e-6, spread, np.nan)
        z = (std - med) / spread[np.newaxis, :]
        bad |= np.nan_to_num(z) > z_threshold
    return bad.mean(axis=1)


def excess_kurtosis(x: np.ndarray) -> np.ndarray:
    return stats.kurtosis(x, axis=-1, fisher=True, bias=False)


def deviation_z(x: np.ndarray) -> np.ndarray:
    """Robust z-score of each channel's robust amplitude (PREP 'deviation')."""
    return robust_zscore(robust_std(x))
