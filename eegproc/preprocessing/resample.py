"""Resampling with anti-aliasing (polyphase FIR) that keeps trigger codes intact."""

from __future__ import annotations

from fractions import Fraction

import numpy as np
from scipy.signal import resample_poly

from ..utils import logger


def resample_data(data: np.ndarray, sfreq: float, new_sfreq: float) -> tuple[np.ndarray, float]:
    """Resample along the last axis. Returns ``(data, exact_new_sfreq)``."""
    ratio = Fraction(new_sfreq / sfreq).limit_denominator(1000)
    up, down = ratio.numerator, ratio.denominator
    exact = sfreq * up / down
    if abs(exact - new_sfreq) > 1e-6 * new_sfreq:
        logger.warning("Requested %g Hz; using the closest rational rate %.6g Hz", new_sfreq, exact)
    if up == down:
        return np.array(data, dtype=float, copy=True), float(sfreq)
    return resample_poly(np.asarray(data, dtype=float), up, down, axis=-1, padtype="line"), float(exact)


def _resample_stim(x: np.ndarray, n_new: int, factor: float) -> np.ndarray:
    """Move every trigger onset to its new sample, keeping its code and (scaled) length."""
    out = np.zeros(n_new)
    change = np.flatnonzero(np.diff(x) != 0) + 1
    starts = np.concatenate([[0], change])
    ends = np.concatenate([change, [len(x)]])
    for a, b in zip(starts, ends):
        v = x[a]
        if v == 0:
            continue
        na = min(n_new - 1, int(round(a * factor)))
        nb = max(na + 1, min(n_new, int(round(b * factor))))
        out[na:nb] = v
    return out


def resample_raw(raw, sfreq: float):
    """Return a resampled copy of a :class:`RawEEG`. Annotations keep their times."""
    data_idx = [i for i, t in enumerate(raw.ch_types) if t != "stim"]
    stim_idx = [i for i, t in enumerate(raw.ch_types) if t == "stim"]
    new_data, new_sfreq = resample_data(raw.data[data_idx], raw.sfreq, sfreq)
    n_new = new_data.shape[1] if data_idx else int(round(raw.n_times * sfreq / raw.sfreq))
    out_data = np.zeros((raw.n_channels, n_new))
    if data_idx:
        out_data[data_idx] = new_data
    factor = new_sfreq / raw.sfreq
    for i in stim_idx:
        out_data[i] = _resample_stim(raw.data[i], n_new, factor)
    out = raw._new(out_data, sfreq=new_sfreq, note=f"resample {raw.sfreq:g} -> {new_sfreq:g} Hz")
    out.annotations = raw.annotations.crop(0.0, n_new / new_sfreq, shift=False)
    lp = out.meta.get("lowpass")
    out.meta["lowpass"] = min(float(lp), new_sfreq / 2) if lp else new_sfreq / 2
    return out
