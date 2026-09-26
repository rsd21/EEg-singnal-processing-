"""Per-channel features for description and machine learning.

Time domain: mean, std, RMS, peak-to-peak, skewness, kurtosis, line length,
zero-crossing rate, Hjorth activity/mobility/complexity.
Complexity: sample entropy, permutation entropy, Higuchi fractal dimension,
Katz fractal dimension.
Spectral: absolute and relative band power, band ratios, peak alpha
frequency, spectral entropy, spectral edge, 1/f exponent.
"""

from __future__ import annotations

import math
from typing import Iterable, Sequence

import numpy as np
from scipy import stats

from ..utils import format_table, write_csv
from .spectral import BANDS, Spectrum, fit_aperiodic
from .._spectral_core import welch_psd


# ------------------------------------------------------------ time domain
def hjorth(x: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Hjorth activity (variance), mobility and complexity along the last axis."""
    x = np.asarray(x, dtype=float)
    dx = np.diff(x, axis=-1)
    ddx = np.diff(dx, axis=-1)
    v0, v1, v2 = x.var(axis=-1), dx.var(axis=-1), ddx.var(axis=-1)
    mobility = np.sqrt(v1 / np.where(v0 > 0, v0, np.nan))
    complexity = np.sqrt(v2 / np.where(v1 > 0, v1, np.nan)) / mobility
    return v0, mobility, complexity


def line_length(x: np.ndarray) -> np.ndarray:
    """Mean absolute sample-to-sample difference (sensitive to spikes/seizures)."""
    return np.abs(np.diff(x, axis=-1)).mean(axis=-1)


def zero_crossing_rate(x: np.ndarray, sfreq: float) -> np.ndarray:
    xc = x - x.mean(axis=-1, keepdims=True)
    crossings = (np.diff(np.signbit(xc), axis=-1) != 0).sum(axis=-1)
    return crossings / (x.shape[-1] / sfreq)


# ------------------------------------------------------------- complexity
def sample_entropy(x: np.ndarray, m: int = 2, r: float = 0.2, max_samples: int = 3000) -> float:
    """Sample entropy (Richman & Moorman, 2000) of a 1D signal.

    ``r`` is a fraction of the signal's standard deviation. Long signals are
    truncated to ``max_samples`` (the cost is quadratic).
    """
    x = np.asarray(x, dtype=float)[:max_samples]
    n = len(x)
    if n < m + 2:
        return float("nan")
    tol = r * x.std()
    if tol == 0:
        return float("nan")

    def count(mm: int) -> int:
        emb = np.lib.stride_tricks.sliding_window_view(x, mm)[: n - m]
        total = 0
        for i in range(len(emb) - 1):
            d = np.max(np.abs(emb[i + 1:] - emb[i]), axis=1)
            total += int((d <= tol).sum())
        return total

    b, a = count(m), count(m + 1)
    if a == 0 or b == 0:
        return float("inf") if b > 0 else float("nan")
    return float(-np.log(a / b))


def permutation_entropy(x: np.ndarray, order: int = 3, delay: int = 1, normalize: bool = True) -> float:
    """Permutation entropy (Bandt & Pompe, 2002) of a 1D signal."""
    x = np.asarray(x, dtype=float)
    n = len(x) - (order - 1) * delay
    if n <= 0:
        return float("nan")
    emb = np.stack([x[i * delay: i * delay + n] for i in range(order)], axis=1)
    patterns = np.argsort(emb, axis=1, kind="stable")
    codes = (patterns * (order ** np.arange(order))).sum(axis=1)
    _, counts = np.unique(codes, return_counts=True)
    p = counts / counts.sum()
    h = -(p * np.log2(p)).sum()
    return float(h / np.log2(math.factorial(order))) if normalize else float(h)


def higuchi_fd(x: np.ndarray, kmax: int = 10) -> float:
    """Higuchi fractal dimension (1 = smooth line, 2 = space-filling noise)."""
    x = np.asarray(x, dtype=float)
    n = len(x)
    lk = []
    ks = np.arange(1, kmax + 1)
    for k in ks:
        lm = []
        for m in range(k):
            idx = np.arange(m, n, k)
            if len(idx) < 2:
                continue
            length = np.abs(np.diff(x[idx])).sum() * (n - 1) / ((len(idx) - 1) * k) / k
            lm.append(length)
        lk.append(np.mean(lm))
    lk = np.asarray(lk)
    valid = lk > 0
    slope, _ = np.polyfit(np.log(1.0 / ks[valid]), np.log(lk[valid]), 1)
    return float(slope)


def katz_fd(x: np.ndarray) -> float:
    x = np.asarray(x, dtype=float)
    dists = np.abs(np.diff(x))
    total = dists.sum()
    if total == 0:
        return float("nan")
    a = dists.mean()
    d = np.max(np.abs(x - x[0]))
    n = total / a
    return float(np.log10(n) / (np.log10(n) + np.log10(d / total)))


# ------------------------------------------------------------- extraction
TIME_FEATURES = ("mean", "std", "rms", "ptp", "skewness", "kurtosis", "line_length", "zero_crossings",
                 "hjorth_activity", "hjorth_mobility", "hjorth_complexity")
COMPLEXITY_FEATURES = ("sample_entropy", "permutation_entropy", "higuchi_fd", "katz_fd")
SPECTRAL_FEATURES = ("abs_power", "rel_power", "ratios", "peak_alpha", "spectral_entropy",
                     "spectral_edge", "aperiodic_exponent")


def channel_features(x: np.ndarray, sfreq: float, bands: dict | None = None,
                     include: Iterable[str] = ("time", "spectral"), window_sec: float = 2.0) -> dict[str, np.ndarray]:
    """Compute features for each row of ``x`` (n_channels, n_samples).

    ``include`` chooses groups: ``'time'``, ``'spectral'``, ``'complexity'``
    (complexity is slower). Returns ``{feature_name: array(n_channels)}``.
    """
    x = np.atleast_2d(np.asarray(x, dtype=float))
    include = set(include)
    bands = bands or BANDS
    out: dict[str, np.ndarray] = {}
    if "time" in include:
        act, mob, comp = hjorth(x)
        out.update({
            "mean": x.mean(axis=1), "std": x.std(axis=1), "rms": np.sqrt((x ** 2).mean(axis=1)),
            "ptp": x.max(axis=1) - x.min(axis=1), "skewness": stats.skew(x, axis=1),
            "kurtosis": stats.kurtosis(x, axis=1), "line_length": line_length(x),
            "zero_crossings": zero_crossing_rate(x, sfreq), "hjorth_activity": act,
            "hjorth_mobility": mob, "hjorth_complexity": comp,
        })
    if "spectral" in include:
        freqs, p = welch_psd(x, sfreq, window_sec=min(window_sec, x.shape[1] / sfreq))
        spec = Spectrum(freqs, p, [f"c{i}" for i in range(len(x))])
        top = min(max(b[1] for b in bands.values()), freqs[-1])
        valid_bands = {k: v for k, v in bands.items() if v[0] < freqs[-1]}
        ab = spec.band_power(valid_bands)
        rel = spec.band_power(valid_bands, relative=True, total_range=(min(b[0] for b in valid_bands.values()), top))
        for b in valid_bands:
            out[f"abs_{b}"] = ab[b]
            out[f"rel_{b}"] = rel[b]
        if {"theta", "beta"} <= set(ab):
            out["theta_beta_ratio"] = ab["theta"] / ab["beta"]
        if {"alpha", "theta"} <= set(ab):
            out["alpha_theta_ratio"] = ab["alpha"] / ab["theta"]
        if freqs[-1] > 14:
            out["peak_alpha_freq"] = spec.peak_frequency(7.0, 14.0)
        out["spectral_entropy"] = spec.spectral_entropy(1.0, min(45.0, freqs[-1]))
        out["spectral_edge_95"] = spec.spectral_edge(0.95, min(45.0, freqs[-1]))
        if freqs[-1] > 20:
            out["aperiodic_exponent"] = fit_aperiodic(freqs, p, 2.0, min(40.0, freqs[-1]))[1]
    if "complexity" in include:
        out["sample_entropy"] = np.array([sample_entropy(r) for r in x])
        out["permutation_entropy"] = np.array([permutation_entropy(r) for r in x])
        out["higuchi_fd"] = np.array([higuchi_fd(r) for r in x])
        out["katz_fd"] = np.array([katz_fd(r) for r in x])
    return out


class FeatureTable:
    """Rows = channels (or epochs x channels), columns = features."""

    def __init__(self, rows: list[dict]):
        self.rows = rows

    @property
    def columns(self) -> list[str]:
        return list(self.rows[0]) if self.rows else []

    def __len__(self) -> int:
        return len(self.rows)

    def __repr__(self) -> str:
        return f"<FeatureTable | {len(self.rows)} rows x {len(self.columns)} columns>"

    def to_csv(self, path) -> None:
        write_csv(self.rows, path)

    def to_dataframe(self):
        from ..utils import to_dataframe

        return to_dataframe(self.rows)

    def to_array(self, exclude: Sequence[str] = ("channel", "epoch", "label")) -> np.ndarray:
        cols = [c for c in self.columns if c not in exclude]
        return np.array([[r[c] for c in cols] for r in self.rows], dtype=float)

    def __str__(self) -> str:
        return format_table(self.rows, max_rows=40)


def extract_features(inst, picks="eeg", include: Iterable[str] = ("time", "spectral"),
                     bands: dict | None = None) -> FeatureTable:
    """Feature table for a :class:`RawEEG` (one row per channel) or
    :class:`Epochs` (one row per epoch and channel, with the epoch label)."""
    from ..core.epochs import Epochs

    if isinstance(inst, Epochs):
        rows = []
        idx = inst._pick(None if picks in ("eeg", "data", None) else picks)
        for e in range(inst.n_epochs):
            feats = channel_features(inst.data[e, idx], inst.sfreq, bands, include)
            for j, i in enumerate(idx):
                row = {"epoch": e, "label": int(inst.labels[e]), "channel": inst.ch_names[i]}
                row.update({k: float(v[j]) for k, v in feats.items()})
                rows.append(row)
        return FeatureTable(rows)
    idx = inst.pick_indices(picks)
    feats = channel_features(inst.data[idx], inst.sfreq, bands, include)
    rows = []
    for j, i in enumerate(idx):
        row = {"channel": inst.ch_names[i]}
        row.update({k: float(v[j]) for k, v in feats.items()})
        rows.append(row)
    return FeatureTable(rows)
