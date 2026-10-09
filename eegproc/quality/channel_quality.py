"""Channel quality assessment: a 0-100 score, a grade and reasons per channel.

The score starts at 100 and loses points for each problem found. Every
penalty is a linear ramp between a "fine" and a "clearly bad" value of one
metric, capped at a maximum, so the result is easy to explain:

=================  =====================================  ==========  =======
Problem            Metric                                 Ramp        Max
=================  =====================================  ==========  =======
Flat / dead        fraction of 1 s windows with std<0.5µV 0 -> 50 %   100
Clipping           samples in saturation plateaus         0.05->1 %   40
Abnormal amplitude robust z of log amplitude / abs. level z 3 -> 8    40
Poor correlation   median |r| with spatial neighbours     0.6 -> 0.2  40
High-freq. noise   PREP noisiness z / SNR (1-30 Hz vs HF) z 3->8 /    35
                                                          15->3 dB
Line noise         mains peak over neighbourhood          25->45 dB   15
Slow drift         <1 Hz amplitude vs 1-40 Hz amplitude   12->26 dB   20
                   ... relative to the median channel     6->15 dB    25
Local artifacts    windows with own artifacts             2 -> 30 %   30
Spikes / pops      kurtosis of channel minus neighbours   5 -> 20     25
Unstable contact   spread of log window amplitude         0.25->0.6   15
Noise-like PSD     1/f exponent                           0.6 -> 0    10
=================  =====================================  ==========  =======

Grades: good >= 75 > fair >= 50 > poor >= 25 > bad. A channel is suggested
as bad when its grade is ``bad`` or a PREP criterion flags it
(:func:`eegproc.preprocessing.find_bad_channels`).
"""

from __future__ import annotations

import json
from typing import Sequence

import numpy as np

from ..core.montage import find_neighbors
from ..preprocessing.filters import detect_line_noise, filter_data
from ..utils import format_table, logger, robust_zscore, to_dataframe, write_csv
from . import metrics as M

GRADE_LIMITS = (("good", 75.0), ("fair", 50.0), ("poor", 25.0), ("bad", 0.0))
GRADE_COLORS = {"good": "#2a9d4b", "fair": "#e0a526", "poor": "#e0662b", "bad": "#c62828"}


def grade_for(score: float) -> str:
    for name, lim in GRADE_LIMITS:
        if score >= lim:
            return name
    return "bad"


def _ramp(value, start: float, end: float, max_pen: float) -> float:
    """Linear penalty: 0 at ``start``, ``max_pen`` at ``end`` (works for both directions)."""
    if value is None or not np.isfinite(value):
        return 0.0
    frac = (value - start) / (end - start)
    return float(max_pen * np.clip(frac, 0.0, 1.0))


class ChannelQualityReport:
    """Result of :func:`assess_channel_quality`."""

    def __init__(self, rows: list[dict], info: dict, positions: dict | None = None):
        self.rows = rows
        self.info = info
        self.positions = dict(positions or {})

    # ------------------------------------------------------------- access
    @property
    def ch_names(self) -> list[str]:
        return [r["channel"] for r in self.rows]

    @property
    def scores(self) -> np.ndarray:
        return np.array([r["score"] for r in self.rows])

    @property
    def grades(self) -> list[str]:
        return [r["grade"] for r in self.rows]

    def __getitem__(self, channel: str) -> dict:
        for r in self.rows:
            if r["channel"] == channel:
                return r
        raise KeyError(channel)

    def metric(self, name: str) -> np.ndarray:
        return np.array([r.get(name, np.nan) for r in self.rows], dtype=float)

    def ranked(self) -> list[dict]:
        return sorted(self.rows, key=lambda r: (-r["score"], r["channel"]))

    def best(self, n: int = 5) -> list[str]:
        return [r["channel"] for r in self.ranked()[:n]]

    def worst(self, n: int = 5) -> list[str]:
        return [r["channel"] for r in self.ranked()[::-1][:n]]

    @property
    def bad_channels(self) -> list[str]:
        return [r["channel"] for r in self.rows if r["suggest_bad"]]

    def by_grade(self) -> dict[str, list[str]]:
        out = {g: [] for g, _ in GRADE_LIMITS}
        for r in self.ranked():
            out[r["grade"]].append(r["channel"])
        return out

    @property
    def overall_score(self) -> float:
        """Median channel score: one number for the whole recording."""
        return float(np.median(self.scores)) if self.rows else float("nan")

    # ------------------------------------------------------------ display
    def __repr__(self) -> str:
        counts = {g: len(v) for g, v in self.by_grade().items()}
        return (f"<ChannelQualityReport | {len(self.rows)} channels | " +
                ", ".join(f"{v} {k}" for k, v in counts.items()) +
                f" | median score {self.overall_score:.0f}>")

    def summary(self, max_rows: int | None = None) -> str:
        counts = {g: len(v) for g, v in self.by_grade().items()}
        lines = [
            f"Channel quality: {len(self.rows)} channels | " +
            ", ".join(f"{v} {k}" for k, v in counts.items()) +
            f" | median score {self.overall_score:.0f}/100",
            f"Line noise: {self.info.get('line_freq') or 'none detected'} | "
            f"analysed {self.info.get('duration', 0):.1f} s at {self.info.get('sfreq', 0):g} Hz",
            "Best channels : " + ", ".join(f"{r['channel']} ({r['score']:.0f})" for r in self.ranked()[:5]),
            "Worst channels: " + ", ".join(f"{r['channel']} ({r['score']:.0f})" for r in self.ranked()[::-1][:5]),
            "Suggested bad : " + (", ".join(self.bad_channels) or "none"),
            "",
        ]
        table = [{"rank": i + 1, "channel": r["channel"], "score": round(r["score"], 1), "grade": r["grade"],
                  "amp_uV": r["amplitude_uv"], "nbr_corr": r["neighbor_corr"], "snr_dB": r["snr_db"],
                  "issues": "; ".join(r["reasons"]) or "-"}
                 for i, r in enumerate(self.ranked())]
        lines.append(format_table(table, max_rows=max_rows, max_width=90))
        return "\n".join(lines)

    def table(self, columns: Sequence[str] | None = None) -> list[dict]:
        cols = columns or ["rank", "channel", "score", "grade", "suggest_bad", "reasons", "prep_flags",
                           "amplitude_uv", "flat_fraction", "clipping_fraction", "deviation_z",
                           "neighbor_corr", "corr_bad_fraction", "hf_noise_z", "snr_db", "line_noise_db",
                           "drift_db", "bad_window_fraction", "kurtosis", "instability",
                           "aperiodic_exponent", "alpha_peak_hz", "alpha_peak_db"]
        return [{c: r.get(c) for c in cols} for r in self.ranked()]

    def to_csv(self, path) -> None:
        write_csv(self.table(), path)

    def to_dataframe(self):
        return to_dataframe(self.table())

    def to_dict(self) -> dict:
        def clean(v):
            if isinstance(v, (np.floating, float)):
                return None if not np.isfinite(v) else float(v)
            if isinstance(v, (np.integer,)):
                return int(v)
            if isinstance(v, dict):
                return {k: clean(x) for k, x in v.items()}
            if isinstance(v, (list, tuple)):
                return [clean(x) for x in v]
            return v
        return {"info": clean(self.info), "channels": [clean(r) for r in self.ranked()]}

    def to_json(self, path) -> None:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2)

    def plot(self, **kwargs):
        from ..viz.quality import plot_quality_report

        return plot_quality_report(self, **kwargs)

    def plot_bars(self, **kwargs):
        from ..viz.quality import plot_quality_bars

        return plot_quality_bars(self, **kwargs)

    def plot_topomap(self, metric: str = "score", **kwargs):
        from ..viz.quality import plot_quality_topomap

        return plot_quality_topomap(self, metric=metric, **kwargs)


def assess_channel_quality(raw, picks="eeg", line_freq: float | str | None = "auto",
                           window: float = 1.0, flat_threshold: float = 0.5,
                           ptp_threshold: float = 200.0, use_prep: bool = True,
                           max_seconds: float | None = 1800.0) -> ChannelQualityReport:
    """Score every channel's signal quality (0-100) and explain the score.

    Parameters
    ----------
    raw : RawEEG
    picks : channels to assess (default: all EEG channels, including bads).
    line_freq : 50, 60, None or ``'auto'`` (detected from the spectrum).
    window : window length (s) for windowed metrics.
    flat_threshold : µV; windows quieter than this count as flat.
    ptp_threshold : µV; peak-to-peak above this in a window is an artifact.
    use_prep : also run PREP bad-channel criteria and report their flags.
    max_seconds : analyse at most this much data (evenly spread chunks) to
        keep long recordings fast; ``None`` uses everything.
    """
    idx = raw.pick_indices(picks)
    if not idx:
        raise ValueError(f"No channels match picks={picks!r}")
    names = [raw.ch_names[i] for i in idx]
    sfreq = raw.sfreq
    x = np.nan_to_num(raw.data[idx].astype(float))
    if max_seconds is not None and x.shape[1] > max_seconds * sfreq:
        x = _subsample(x, sfreq, max_seconds)
    duration = x.shape[1] / sfreq
    if duration < 10:
        logger.warning("Only %.1f s of data; quality metrics will be rough", duration)
    if isinstance(line_freq, str) and line_freq == "auto":
        line_freq = detect_line_noise(x, sfreq) if duration >= 2 else None

    n_ch = len(names)
    xh = filter_data(x, sfreq, 1.0, None)
    xb = filter_data(x, sfreq, 1.0, min(40.0, 0.45 * sfreq))
    freqs, p = M.psd(x - x.mean(axis=1, keepdims=True), sfreq, window_sec=4.0)

    amplitude = M.robust_std(xb)
    flat = M.flat_fraction(xh, sfreq, window, flat_threshold)
    clip = M.clipping_fraction(x)
    alive = flat < 0.5
    dev_z = np.zeros(n_ch)
    if alive.sum() >= 3:
        dev_z[alive] = robust_zscore(np.log10(np.maximum(amplitude[alive], 1e-6)))

    positions = {n: raw.positions[n] for n in names if n in raw.positions}
    neigh_corr = np.full(n_ch, np.nan)
    corr_bad = np.zeros(n_ch)
    neighbors = None
    if n_ch >= 3:
        corr = M.window_correlations(xb, sfreq, window)
        if len(positions) == n_ch:
            neighbors = find_neighbors(np.array([positions[n] for n in names]), min(4, n_ch - 1))
        else:
            neighbors = M.top_correlated(corr, min(4, n_ch - 1))
        # do not judge a channel by dead neighbours
        neighbors = [nb[alive[nb]] if alive[nb].any() else nb for nb in neighbors]
        neigh_corr = M.neighbor_correlation(corr, neighbors)
        corr_bad = (M.max_correlation(corr) < 0.4).mean(axis=0)

    hf_z = np.zeros(n_ch)
    if sfreq > 40 and alive.sum() >= 3:
        ratio = M.hf_noise_ratio(xh, sfreq)
        hf_z[alive] = robust_zscore(np.nan_to_num(ratio[alive]))
    snr = M.snr_db(freqs, p, sfreq, line_freq)
    line_db = M.line_noise_db(freqs, p, line_freq)
    drift = M.drift_db(x, xb, xh)
    drift_ref = float(np.median(drift[alive])) if alive.any() else 0.0
    badwin = M.bad_window_fraction(xh, sfreq, window, ptp_threshold)
    kurt = M.local_kurtosis(xb, neighbors)
    instab = M.amplitude_instability(xb, sfreq, window)
    expo = M.aperiodic_exponent(freqs, p, line_freq=line_freq)
    alpha_f, alpha_db = M.alpha_peak(freqs, p)

    prep_flags: dict[str, list[str]] = {}
    if use_prep and n_ch >= 3:
        from ..preprocessing.bad_channels import find_bad_channels

        prep = find_bad_channels(raw, picks=idx, random_state=0)
        prep_flags = prep.reasons()

    rows = []
    for k, name in enumerate(names):
        pens: dict[str, float] = {}
        reasons: list[tuple[float, str]] = []

        def add(key: str, value: float, text: str):
            if value > 0:
                pens[key] = value
                if value >= 3:
                    reasons.append((value, text))

        add("flat", _ramp(flat[k], 0.0, 0.5, 100),
            f"flat / no signal in {100 * flat[k]:.0f}% of the recording")
        add("clipping", _ramp(clip[k], 0.0005, 0.01, 40),
            f"amplifier saturation (clipping) in {100 * clip[k]:.1f}% of samples")
        if alive[k]:
            amp_pen = max(_ramp(abs(dev_z[k]), 3, 8, 40), _ramp(amplitude[k], 60, 150, 40),
                          _ramp(-amplitude[k], -2.0, -0.5, 25))
            direction = "high" if dev_z[k] > 0 or amplitude[k] > 60 else "low"
            add("amplitude", amp_pen,
                f"abnormally {direction} amplitude ({amplitude[k]:.1f} µV, z={dev_z[k]:+.1f})")
            corr_pen = max(_ramp(-neigh_corr[k], -0.6, -0.2, 40), _ramp(corr_bad[k], 0.05, 0.5, 30))
            add("correlation", corr_pen,
                f"poorly correlated with neighbouring channels (r={neigh_corr[k]:.2f})")
            hf_pen = max(_ramp(hf_z[k], 3, 8, 35), _ramp(-snr[k], -15, -3, 35))
            add("hf_noise", hf_pen, f"high-frequency noise (SNR {snr[k]:.1f} dB, z={hf_z[k]:+.1f})")
            add("line_noise", _ramp(line_db[k], 25, 45, 15),
                f"strong {line_freq:g} Hz line noise (+{line_db[k]:.0f} dB; a notch filter removes it)"
                if line_freq else "line noise")
            add("drift", max(_ramp(drift[k], 12, 26, 20), _ramp(drift[k] - drift_ref, 6, 15, 25)),
                f"large slow drifts / baseline wander ({drift[k] - drift_ref:+.0f} dB vs other channels)")
            add("artifacts", _ramp(badwin[k], 0.02, 0.3, 30),
                f"channel-specific artifacts in {100 * badwin[k]:.0f}% of windows")
            add("spikes", _ramp(kurt[k], 5, 20, 25),
                f"spikes / electrode pops (local kurtosis {kurt[k]:.1f})")
            add("instability", _ramp(instab[k], 0.25, 0.6, 15),
                f"unstable amplitude over time (intermittent contact, {instab[k]:.2f})")
            add("spectrum", _ramp(-expo[k], -0.6, 0.0, 10),
                f"noise-like flat spectrum (1/f exponent {expo[k]:.2f})")
        score = 0.0 if flat[k] >= 0.5 else float(np.clip(100.0 - sum(pens.values()), 0.0, 100.0))
        grade = grade_for(score)
        flags = prep_flags.get(name, [])
        rows.append({
            "channel": name, "score": round(score, 1), "grade": grade,
            "suggest_bad": grade == "bad" or bool(flags),
            "reasons": [t for _, t in sorted(reasons, key=lambda r: -r[0])],
            "prep_flags": flags, "penalties": {k2: round(v, 1) for k2, v in pens.items()},
            "amplitude_uv": float(amplitude[k]), "flat_fraction": float(flat[k]),
            "clipping_fraction": float(clip[k]), "deviation_z": float(dev_z[k]),
            "neighbor_corr": float(neigh_corr[k]), "corr_bad_fraction": float(corr_bad[k]),
            "hf_noise_z": float(hf_z[k]), "snr_db": float(snr[k]), "line_noise_db": float(line_db[k]),
            "drift_db": float(drift[k]), "bad_window_fraction": float(badwin[k]),
            "kurtosis": float(kurt[k]), "instability": float(instab[k]),
            "aperiodic_exponent": float(expo[k]), "alpha_peak_hz": float(alpha_f[k]),
            "alpha_peak_db": float(alpha_db[k]),
        })
    for rank, r in enumerate(sorted(rows, key=lambda r: (-r["score"], r["channel"])), start=1):
        r["rank"] = rank
    info = {"sfreq": sfreq, "duration": duration, "line_freq": line_freq, "window": window,
            "n_channels": n_ch, "filename": raw.meta.get("filename")}
    return ChannelQualityReport(rows, info, positions)


def _subsample(x: np.ndarray, sfreq: float, max_seconds: float, n_chunks: int = 30) -> np.ndarray:
    """Evenly spaced chunks totalling ``max_seconds`` (keeps start, middle and end)."""
    chunk = int(max_seconds * sfreq / n_chunks)
    starts = np.linspace(0, x.shape[1] - chunk, n_chunks).astype(int)
    return np.concatenate([x[:, s: s + chunk] for s in starts], axis=1)
