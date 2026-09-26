"""Digital filters for EEG: band-pass, high-pass, low-pass, band-stop and notch.

Two designs are available:

* ``method='iir'`` (default): Butterworth filter in second-order sections,
  run forwards and backwards (``sosfiltfilt``) for zero phase distortion.
  The effective order is twice the design order.
* ``method='fir'``: windowed-sinc (Hamming) linear-phase FIR whose length is
  set from the transition bandwidth (3.3 / transition), applied with FFT
  convolution and delay compensation, so it is also zero-phase.

Zero-phase filtering keeps ERP latencies and oscillation phases intact, which
matters for EEG analysis.
"""

from __future__ import annotations

import numpy as np
from scipy import signal

from ..utils import logger


def _check_freqs(sfreq: float, l_freq: float | None, h_freq: float | None) -> tuple:
    nyq = sfreq / 2.0
    if l_freq is not None and l_freq <= 0:
        l_freq = None
    if h_freq is not None and h_freq >= nyq:
        logger.warning("h_freq %.4g Hz is at/above Nyquist (%.4g Hz); low-pass skipped", h_freq, nyq)
        h_freq = None
    if l_freq is not None and l_freq >= nyq:
        raise ValueError(f"l_freq {l_freq} Hz must be below Nyquist ({nyq} Hz)")
    return l_freq, h_freq


def design_iir(sfreq: float, l_freq: float | None, h_freq: float | None, order: int = 4,
               ftype: str = "butter", rp: float = 0.5, rs: float = 40.0) -> np.ndarray | None:
    """Return second-order sections for the requested band (None if nothing to do)."""
    l_freq, h_freq = _check_freqs(sfreq, l_freq, h_freq)
    if l_freq is None and h_freq is None:
        return None
    if l_freq is not None and h_freq is not None:
        if l_freq < h_freq:
            wn, btype = [l_freq, h_freq], "bandpass"
        else:
            wn, btype = [h_freq, l_freq], "bandstop"
    elif l_freq is not None:
        wn, btype = l_freq, "highpass"
    else:
        wn, btype = h_freq, "lowpass"
    kwargs = {}
    if ftype in ("cheby1", "ellip"):
        kwargs["rp"] = rp
    if ftype in ("cheby2", "ellip"):
        kwargs["rs"] = rs
    return signal.iirfilter(order, wn, btype=btype, ftype=ftype, fs=sfreq, output="sos", **kwargs)


def _auto_transition(freq: float, sfreq: float, high: bool) -> float:
    nyq = sfreq / 2.0
    if high:  # low-pass edge
        return min(max(0.25 * freq, 2.0), nyq - freq)
    return min(max(0.25 * freq, 2.0), freq)


def design_fir(sfreq: float, l_freq: float | None, h_freq: float | None,
               l_trans: float | str = "auto", h_trans: float | str = "auto",
               window: str = "hamming") -> np.ndarray | None:
    """Windowed-sinc FIR taps with automatic transition bands (odd length)."""
    l_freq, h_freq = _check_freqs(sfreq, l_freq, h_freq)
    if l_freq is None and h_freq is None:
        return None
    trans = []
    if l_freq is not None:
        lt = _auto_transition(l_freq, sfreq, False) if l_trans == "auto" else float(l_trans)
        trans.append(lt)
    if h_freq is not None:
        ht = _auto_transition(h_freq, sfreq, True) if h_trans == "auto" else float(h_trans)
        trans.append(ht)
    width = min(trans)
    numtaps = int(np.ceil(3.3 / width * sfreq))
    numtaps += 1 - numtaps % 2  # odd -> type I, works for every band type
    nyq = sfreq / 2.0
    if l_freq is not None and h_freq is not None:
        if l_freq < h_freq:
            cutoff, pass_zero = [l_freq - lt / 2, h_freq + ht / 2], False
        else:
            cutoff, pass_zero = [h_freq + ht / 2, l_freq - lt / 2], True
    elif l_freq is not None:
        cutoff, pass_zero = l_freq - lt / 2, False
    else:
        cutoff, pass_zero = h_freq + ht / 2, True
    cutoff = np.clip(cutoff, 1e-6, nyq - 1e-6)
    return signal.firwin(numtaps, cutoff, window=window, pass_zero=pass_zero, fs=sfreq)


def _apply_fir(x: np.ndarray, taps: np.ndarray) -> np.ndarray:
    n = len(taps)
    pad = n - 1
    n_t = x.shape[-1]
    mode = "reflect" if n_t > 1 else "edge"
    padded = np.pad(x, [(0, 0)] * (x.ndim - 1) + [(pad, pad)], mode=mode)
    y = signal.fftconvolve(padded, taps.reshape((1,) * (x.ndim - 1) + (-1,)), mode="same", axes=-1)
    return y[..., pad: pad + n_t]


def filter_data(data: np.ndarray, sfreq: float, l_freq: float | None = None,
                h_freq: float | None = None, method: str = "iir", order: int = 4,
                ftype: str = "butter", phase: str = "zero", l_trans="auto", h_trans="auto") -> np.ndarray:
    """Filter an array along its last axis.

    ``l_freq`` only -> high-pass, ``h_freq`` only -> low-pass, both with
    ``l_freq < h_freq`` -> band-pass, ``l_freq > h_freq`` -> band-stop.
    ``phase='causal'`` runs the filter forwards only (for real-time use).
    """
    x = np.asarray(data, dtype=float)
    if method == "iir":
        sos = design_iir(sfreq, l_freq, h_freq, order=order, ftype=ftype)
        if sos is None:
            return x.copy()
        if phase == "causal":
            return signal.sosfilt(sos, x, axis=-1)
        padlen = min(x.shape[-1] - 1, 3 * (2 * len(sos) + 1) * 10)
        return signal.sosfiltfilt(sos, x, axis=-1, padtype="odd", padlen=max(padlen, 0))
    if method == "fir":
        taps = design_fir(sfreq, l_freq, h_freq, l_trans, h_trans)
        if taps is None:
            return x.copy()
        if len(taps) > x.shape[-1]:
            logger.warning("FIR filter (%d taps) is longer than the signal (%d samples); "
                           "consider method='iir' or a wider transition band", len(taps), x.shape[-1])
        if phase == "causal":
            return signal.lfilter(taps, [1.0], x, axis=-1)
        return _apply_fir(x, taps)
    raise ValueError(f"Unknown method {method!r}; use 'iir' or 'fir'")


def notch_filter_data(data: np.ndarray, sfreq: float, freqs=50.0, quality: float = 30.0,
                      harmonics: bool = True) -> np.ndarray:
    """Remove power-line interference with zero-phase IIR notch filters.

    ``freqs`` may be a number or a list; with ``harmonics=True`` every
    multiple of each frequency below Nyquist is also removed.
    """
    x = np.asarray(data, dtype=float).copy()
    nyq = sfreq / 2.0
    targets: list[float] = []
    for f0 in np.atleast_1d(freqs).astype(float):
        mult = np.arange(1, int(nyq // f0) + 1) if harmonics else np.array([1])
        targets += [f0 * m for m in mult if f0 * m < nyq - 1.0]
    for f in sorted(set(targets)):
        b, a = signal.iirnotch(f, quality, fs=sfreq)
        x = signal.filtfilt(b, a, x, axis=-1, padlen=min(x.shape[-1] - 1, 3 * max(len(a), len(b)) * 20))
    return x


def detect_line_noise(data: np.ndarray, sfreq: float) -> float | None:
    """Return 50 or 60 (Hz) if power-line interference is visible, else None."""
    from .._spectral_core import welch_psd

    x = np.atleast_2d(np.asarray(data, dtype=float))
    freqs, psd = welch_psd(x, sfreq, window_sec=min(4.0, x.shape[-1] / sfreq))
    med = np.median(psd, axis=0)
    scores = {}
    for f0 in (50.0, 60.0):
        if f0 >= sfreq / 2 - 1:
            continue
        peak = med[np.abs(freqs - f0) <= 1.0].max(initial=0)
        flank = med[(np.abs(freqs - f0) > 3) & (np.abs(freqs - f0) < 8)]
        if flank.size and peak > 0:
            scores[f0] = 10 * np.log10(peak / np.median(flank))
    if not scores:
        return None
    best = max(scores, key=scores.get)
    return best if scores[best] > 6.0 else None


# ------------------------------------------------------------ RawEEG wrappers
def filter_raw(raw, l_freq: float | None, h_freq: float | None, picks="data", **kwargs):
    """Filter the selected channels of a :class:`RawEEG` (stim channels never)."""
    out = raw.copy()
    idx = [i for i in raw.pick_indices(picks) if raw.ch_types[i] != "stim"]
    if idx:
        out.data[idx] = filter_data(raw.data[idx], raw.sfreq, l_freq, h_freq, **kwargs)
    bandstop = bool(l_freq and h_freq and l_freq > h_freq)
    if l_freq and not bandstop:
        out.meta["highpass"] = max(float(l_freq), float(out.meta.get("highpass") or 0))
    if h_freq and h_freq < raw.sfreq / 2 and not bandstop:
        prev = out.meta.get("lowpass")
        out.meta["lowpass"] = min(float(h_freq), float(prev)) if prev else float(h_freq)
    method = kwargs.get("method", "iir")
    out.history.append(f"filter l_freq={l_freq} h_freq={h_freq} ({method})")
    return out


def notch_raw(raw, freqs=50.0, picks="data", **kwargs):
    if freqs == "auto":
        eeg = raw.pick_indices("eeg") or raw.pick_indices("data")
        freqs = detect_line_noise(raw.data[eeg], raw.sfreq)
        if freqs is None:
            logger.info("No line noise detected; notch filter skipped")
            return raw.copy()
    out = raw.copy()
    idx = [i for i in raw.pick_indices(picks) if raw.ch_types[i] != "stim"]
    if idx:
        out.data[idx] = notch_filter_data(raw.data[idx], raw.sfreq, freqs, **kwargs)
    out.history.append(f"notch {np.atleast_1d(freqs).tolist()} Hz")
    return out
