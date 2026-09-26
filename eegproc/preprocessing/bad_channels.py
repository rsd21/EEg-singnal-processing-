"""Automatic bad-channel detection following the PREP pipeline.

Bigdely-Shamlo et al. (2015), "The PREP pipeline: standardized preprocessing
for large-scale EEG analysis", Front. Neuroinform. 9:16.

Criteria (each can be switched on or off):

``nan``          channel contains NaN/Inf
``flat``         robust amplitude below ``flat_threshold`` µV
``deviation``    robust amplitude is an outlier (|robust z| > ``deviation_z``)
``correlation``  max |r| with any other channel < ``correlation_threshold`` in
                 more than ``correlation_fraction`` of 1 s windows (PREP uses 1 %;
                 the default here, 5 %, avoids false alarms on low-density caps)
``hf_noise``     high-frequency noisiness robust z > ``hf_noise_z``
``dropout``      flat in more than ``dropout_fraction`` of windows
``ransac``       poorly predicted from random subsets of other channels
                 (spherical splines, needs electrode positions; only run with
                 at least ``ransac_min_channels`` channels because sparse caps
                 cannot predict their channels well)
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .._spline import interpolation_matrix
from ..quality import metrics as M
from ..utils import check_random_state, format_table, robust_zscore
from .filters import filter_data

ALL_METHODS = ("nan", "flat", "deviation", "correlation", "hf_noise", "dropout", "ransac")


@dataclass
class BadChannelResult:
    """Outcome of :func:`find_bad_channels`."""

    ch_names: list[str]
    by_method: dict[str, list[str]] = field(default_factory=dict)
    scores: dict[str, np.ndarray] = field(default_factory=dict)

    @property
    def bads(self) -> list[str]:
        found = set().union(*self.by_method.values()) if self.by_method else set()
        return [c for c in self.ch_names if c in found]

    def reasons(self) -> dict[str, list[str]]:
        out: dict[str, list[str]] = {}
        for method, names in self.by_method.items():
            for n in names:
                out.setdefault(n, []).append(method)
        return out

    def table(self) -> list[dict]:
        rows = []
        reasons = self.reasons()
        for i, c in enumerate(self.ch_names):
            row = {"channel": c, "bad": c in reasons, "reasons": reasons.get(c, [])}
            for k, v in self.scores.items():
                row[k] = float(v[i])
            rows.append(row)
        return rows

    def __repr__(self) -> str:
        return f"<BadChannelResult | {len(self.bads)} bad: {self.bads}>"

    def summary(self) -> str:
        return format_table(self.table())


def find_bad_channels(raw, picks="eeg", methods=ALL_METHODS, flat_threshold: float = 0.5,
                      deviation_z: float = 5.0, correlation_threshold: float = 0.4,
                      correlation_fraction: float = 0.05, hf_noise_z: float = 5.0,
                      dropout_fraction: float = 0.01, ransac_threshold: float = 0.75,
                      ransac_fraction: float = 0.4, ransac_samples: int = 50,
                      ransac_subset: float = 0.25, ransac_min_channels: int = 32, window: float = 1.0,
                      random_state=None) -> BadChannelResult:
    """Detect bad channels. Returns a :class:`BadChannelResult`.

    Use ``raw.set_bads(result.bads)`` (or :func:`mark_bad_channels`) to
    apply the result.
    """
    idx = raw.pick_indices(picks)
    names = [raw.ch_names[i] for i in idx]
    x = raw.data[idx].astype(float)
    sfreq = raw.sfreq
    result = BadChannelResult(names)
    methods = set(methods)

    finite = np.isfinite(x).all(axis=1)
    if "nan" in methods:
        result.by_method["nan"] = [n for n, ok in zip(names, finite) if not ok]
    x = np.where(np.isfinite(x), x, 0.0)
    h_freq = min(50.0, 0.45 * sfreq)
    xb = filter_data(x, sfreq, 1.0, h_freq) if x.shape[1] > 3 * sfreq else x - x.mean(axis=1, keepdims=True)

    amp = M.robust_std(xb)
    result.scores["robust_std_uv"] = amp
    flat = amp < flat_threshold
    if "flat" in methods:
        result.by_method["flat"] = [n for n, f in zip(names, flat) if f]

    usable = ~flat & finite
    if "deviation" in methods:
        z = np.zeros(len(names))
        if usable.sum() >= 3:
            z[usable] = robust_zscore(amp[usable])
        result.scores["deviation_z"] = z
        result.by_method["deviation"] = [n for n, v in zip(names, z) if abs(v) > deviation_z]

    if "dropout" in methods:
        ff = M.flat_fraction(xb, sfreq, window, threshold=flat_threshold * 0.2)
        result.scores["dropout_fraction"] = ff
        result.by_method["dropout"] = [n for n, v, f in zip(names, ff, flat) if v > dropout_fraction and not f]

    if "correlation" in methods and usable.sum() >= 3:
        corr = M.window_correlations(xb[usable], sfreq, window)
        maxc = M.max_correlation(corr)
        frac = np.zeros(len(names))
        frac[usable] = (maxc < correlation_threshold).mean(axis=0)
        med = np.zeros(len(names))
        med[usable] = np.median(maxc, axis=0)
        result.scores["correlation_bad_fraction"] = frac
        result.scores["max_correlation_median"] = med
        result.by_method["correlation"] = [n for n, v, u in zip(names, frac, usable)
                                           if u and v > correlation_fraction]

    if "hf_noise" in methods and usable.sum() >= 3 and sfreq > 40:
        xh = filter_data(x, sfreq, 1.0, None)
        ratio = M.hf_noise_ratio(xh, sfreq)
        z = np.zeros(len(names))
        z[usable] = robust_zscore(np.nan_to_num(ratio[usable]))
        result.scores["hf_noise_z"] = z
        result.by_method["hf_noise"] = [n for n, v in zip(names, z) if v > hf_noise_z]

    if "ransac" in methods:
        pos_ok = all(n in raw.positions for n in names)
        already = set().union(*result.by_method.values()) if result.by_method else set()
        pool = [k for k, n in enumerate(names) if n not in already]
        if pos_ok and len(names) >= ransac_min_channels and len(pool) >= 8:
            pos = np.array([raw.positions[n] for n in names])
            corr_frac = _ransac(xb, sfreq, pos, pool, ransac_samples, ransac_subset,
                                ransac_threshold, random_state)
            result.scores["ransac_bad_fraction"] = corr_frac
            result.by_method["ransac"] = [n for n, v in zip(names, corr_frac)
                                          if v > ransac_fraction and n not in already]
    return result


def _ransac(x: np.ndarray, sfreq: float, pos: np.ndarray, pool: list[int], n_samples: int,
            subset: float, threshold: float, random_state, window: float = 5.0,
            max_windows: int = 120) -> np.ndarray:
    """Fraction of windows in which each channel is poorly predicted by its peers."""
    rng = check_random_state(random_state)
    n_ch = x.shape[0]
    n_sub = max(4, int(np.ceil(subset * len(pool))))
    win = int(round(window * sfreq))
    n_win = x.shape[1] // win
    if n_win < 1:
        return np.zeros(n_ch)
    models = []
    for _ in range(n_samples):
        chosen = rng.choice(pool, size=n_sub, replace=False)
        models.append((chosen, interpolation_matrix(pos[chosen], pos)))
    windows = np.arange(n_win)
    if n_win > max_windows:
        windows = np.linspace(0, n_win - 1, max_windows).astype(int)
    r = np.empty((n_ch, len(windows)))
    for j, w in enumerate(windows):
        seg = x[:, w * win: (w + 1) * win]
        pred = np.median(np.stack([mat @ seg[chosen] for chosen, mat in models]), axis=0)
        a = seg - seg.mean(axis=1, keepdims=True)
        b = pred - pred.mean(axis=1, keepdims=True)
        den = np.sqrt((a ** 2).sum(1) * (b ** 2).sum(1))
        r[:, j] = np.where(den > 0, (a * b).sum(1) / np.where(den > 0, den, 1.0), 0.0)
    return (r < threshold).mean(axis=1)


def mark_bad_channels(raw, **kwargs):
    """Detect bad channels and add them to a copy's ``bads`` list."""
    result = find_bad_channels(raw, **kwargs)
    out = raw.copy()
    out.add_bads(result.bads)
    out.history.append(f"marked bad channels {result.bads}")
    out.meta["bad_channel_reasons"] = result.reasons()
    return out, result
