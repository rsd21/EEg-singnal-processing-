"""Artifact detection that marks time segments (not channels) as bad."""

from __future__ import annotations

import numpy as np
from scipy import signal

from ..core.annotations import Annotations
from ..utils import mad, sliding_windows
from .filters import filter_data

FRONTAL_EOG_PROXIES = ("Fp1", "Fp2", "Fpz", "AF7", "AF8", "AF3", "AF4", "F7", "F8")


def _merge(mask: np.ndarray, sfreq: float, min_gap: float = 0.0) -> list[tuple[float, float]]:
    """Turn a boolean sample mask into (onset, duration) segments."""
    if not mask.any():
        return []
    edges = np.diff(np.concatenate([[0], mask.astype(np.int8), [0]]))
    starts, ends = np.flatnonzero(edges == 1), np.flatnonzero(edges == -1)
    segs = []
    for a, b in zip(starts, ends):
        if segs and (a - segs[-1][1]) / sfreq <= min_gap:
            segs[-1][1] = b
        else:
            segs.append([a, b])
    return [(a / sfreq, (b - a) / sfreq) for a, b in segs]


def annotate_amplitude(raw, picks="eeg", peak_to_peak: float | None = 150.0,
                       flat: float | None = 1.0, window: float = 1.0, step: float = 0.5,
                       min_channels: int = 1, exclude_bads: bool = True, highpass: float | None = 0.5,
                       description: str = "BAD_amplitude") -> Annotations:
    """Mark windows where the peak-to-peak amplitude is too large (or too flat).

    ``peak_to_peak`` and ``flat`` are in µV. A window is marked when at least
    ``min_channels`` channels exceed the limit.
    """
    idx = raw.pick_indices(picks, exclude_bads=exclude_bads)
    x = raw.data[idx]
    if highpass:
        x = filter_data(x, raw.sfreq, highpass, None)
    win, hop = int(round(window * raw.sfreq)), max(1, int(round(step * raw.sfreq)))
    w = sliding_windows(x, win, hop)
    if w.shape[1] == 0:
        return Annotations()
    ptp = w.max(-1) - w.min(-1)
    bad_win = np.zeros(w.shape[1], dtype=bool)
    if peak_to_peak is not None:
        bad_win |= (ptp > peak_to_peak).sum(axis=0) >= min_channels
    flat_win = np.zeros(w.shape[1], dtype=bool)
    if flat is not None:
        flat_win = (ptp < flat).sum(axis=0) >= max(min_channels, 1)
    ann = Annotations()
    for mask_w, desc in ((bad_win, description), (flat_win, "BAD_flat")):
        mask = np.zeros(raw.n_times, dtype=bool)
        for k in np.flatnonzero(mask_w):
            mask[k * hop: k * hop + win] = True
        for onset, dur in _merge(mask, raw.sfreq):
            ann.append(onset, dur, desc)
    return ann


def annotate_muscle(raw, picks="eeg", threshold: float = 4.0, min_duration: float = 0.1,
                    band: tuple[float, float] | None = None, exclude_bads: bool = True,
                    description: str = "BAD_muscle") -> Annotations:
    """Mark bursts of muscle (EMG) activity.

    High-frequency envelope (default 110-140 Hz, or the top of the available
    band at low sampling rates) is z-scored per channel, averaged over
    channels and thresholded.
    """
    nyq = raw.sfreq / 2
    if band is None:
        band = (110.0, 140.0) if nyq > 150 else (max(30.0, 0.6 * nyq), 0.9 * nyq)
    idx = raw.pick_indices(picks, exclude_bads=exclude_bads)
    x = filter_data(raw.data[idx], raw.sfreq, band[0], band[1])
    env = np.abs(signal.hilbert(x, axis=-1))
    env = filter_data(env, raw.sfreq, None, min(4.0, 0.4 * nyq))
    med = np.median(env, axis=1, keepdims=True)
    spread = mad(env, axis=1)[:, None]
    z = ((env - med) / np.where(spread > 0, spread, 1.0)).mean(axis=0)
    mask = z > threshold
    ann = Annotations()
    for onset, dur in _merge(mask, raw.sfreq, min_gap=0.05):
        if dur >= min_duration:
            ann.append(onset, dur, description)
    return ann


def find_blinks(raw, channel: str | None = None, threshold: float | None = None,
                min_interval: float = 0.3, description: str = "blink") -> Annotations:
    """Detect eye blinks on an EOG channel (or a frontal channel if none).

    The channel is band-passed 1-10 Hz; blinks are peaks above ``threshold``
    µV (default: 4 x robust std + median, adaptive to the recording).
    """
    if channel is None:
        eog = raw.pick_names("eog")
        if eog:
            channel = eog[0]
        else:
            channel = next((c for c in FRONTAL_EOG_PROXIES if c in raw.ch_names), None)
    if channel is None:
        raise ValueError("No EOG or frontal channel found; pass channel=...")
    x = filter_data(raw[channel], raw.sfreq, 1.0, 10.0)
    # blinks can appear with either polarity depending on the reference
    if np.abs(np.percentile(x, 0.5)) > np.abs(np.percentile(x, 99.5)):
        x = -x
    thr = threshold if threshold is not None else np.median(x) + 4 * mad(x)
    peaks, props = signal.find_peaks(x, height=thr, distance=max(1, int(min_interval * raw.sfreq)))
    widths = signal.peak_widths(x, peaks, rel_height=0.5)[0] / raw.sfreq if len(peaks) else []
    ann = Annotations()
    for p, w in zip(peaks, widths):
        ann.append(p / raw.sfreq - w, 2 * w, description)
    return ann


def annotate_artifacts(raw, amplitude: bool = True, muscle: bool = True, blinks: bool = False,
                       **kwargs):
    """Run the amplitude / muscle (/ blink) detectors and add the results to a copy."""
    out = raw.copy()
    found = Annotations()
    if amplitude:
        found.extend(annotate_amplitude(raw, **{k: v for k, v in kwargs.items()
                                                 if k in ("peak_to_peak", "flat", "window", "picks")}))
    if muscle:
        found.extend(annotate_muscle(raw))
    if blinks:
        try:
            found.extend(find_blinks(raw))
        except ValueError:
            pass
    out.annotations.extend(found)
    out.annotations = out.annotations.sorted()
    counts = found.count()
    out.history.append(f"annotate artifacts {counts}")
    return out


def bad_time_fraction(raw) -> float:
    """Fraction of the recording covered by BAD annotations."""
    return float(raw.annotations.bad_mask(raw.n_times, raw.sfreq).mean()) if raw.n_times else 0.0
