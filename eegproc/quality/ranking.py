"""Which channel performs best? Rank channels by many criteria.

Signal quality is only one meaning of "performing". Depending on the
question, a channel performs well when it

* has clean signal                    -> ``by='quality'``
* has a high signal-to-noise ratio    -> ``by='snr'``
* carries the most power in a band    -> ``by='alpha'`` (or any band name, or
  ``'band:8-12'``)
* shows the clearest event-related potential -> ``by='erp'`` (needs events)
* shows the strongest event-related (de)synchronisation -> ``by='erd'``
* separates experimental conditions best -> ``by='discriminability'``
* decodes the condition best on its own (cross-validated classifier)
  -> ``by='decoding'``
* is most connected to other channels (hub) -> ``by='connectivity'``

All rankings return a :class:`ChannelRanking` that can be printed, saved,
plotted as a bar chart or as a topographic map.
"""

from __future__ import annotations

import re
from typing import Sequence

import numpy as np
from scipy import stats

from ..analysis.spectral import BANDS, compute_psd
from ..preprocessing.filters import filter_data
from ..utils import check_random_state, format_table, write_csv


class ChannelRanking:
    """Channels ordered by a performance metric."""

    def __init__(self, ch_names: Sequence[str], values: np.ndarray, metric: str,
                 higher_is_better: bool = True, unit: str = "", description: str = "",
                 details: dict | None = None, positions: dict | None = None):
        self.ch_names = list(ch_names)
        self.values = np.asarray(values, dtype=float)
        self.metric = metric
        self.higher_is_better = higher_is_better
        self.unit = unit
        self.description = description
        self.details = dict(details or {})
        self.positions = dict(positions or {})

    def _order(self) -> np.ndarray:
        v = np.where(np.isfinite(self.values), self.values, -np.inf if self.higher_is_better else np.inf)
        order = np.argsort(v, kind="stable")
        return order[::-1] if self.higher_is_better else order

    def ranked(self) -> list[dict]:
        rows = []
        for rank, i in enumerate(self._order(), start=1):
            row = {"rank": rank, "channel": self.ch_names[i], self.metric: float(self.values[i])}
            for k, v in self.details.items():
                if isinstance(v, (list, np.ndarray)) and len(v) == len(self.ch_names):
                    row[k] = v[i] if not isinstance(v[i], np.floating) else float(v[i])
            rows.append(row)
        return rows

    def top(self, n: int = 5) -> list[str]:
        return [self.ch_names[i] for i in self._order()[:n]]

    def bottom(self, n: int = 5) -> list[str]:
        return [self.ch_names[i] for i in self._order()[::-1][:n]]

    @property
    def best(self) -> str:
        return self.top(1)[0]

    def value(self, ch_name: str) -> float:
        return float(self.values[self.ch_names.index(ch_name)])

    def __repr__(self) -> str:
        return (f"<ChannelRanking '{self.metric}' | best: " +
                ", ".join(f"{c} ({self.value(c):.3g}{self.unit})" for c in self.top(3)) + ">")

    def summary(self, max_rows: int | None = 20) -> str:
        head = f"Ranking by {self.metric}" + (f" [{self.unit}]" if self.unit else "")
        if self.description:
            head += f"\n{self.description}"
        scalars = {k: v for k, v in self.details.items() if np.isscalar(v) or isinstance(v, str)}
        if scalars:
            head += "\n" + ", ".join(f"{k}: {v:.4g}" if isinstance(v, float) else f"{k}: {v}"
                                     for k, v in scalars.items())
        return head + "\n\n" + format_table(self.ranked(), max_rows=max_rows)

    def to_csv(self, path) -> None:
        write_csv(self.ranked(), path)

    def plot(self, kind: str = "both", **kwargs):
        from ..viz.quality import plot_ranking

        return plot_ranking(self, kind=kind, **kwargs)


# =================================================================== raw data
def band_activity(raw, bands: dict | None = None, relative: bool = True, picks="eeg",
                  exclude_bads: bool = True) -> dict[str, ChannelRanking]:
    """Rank channels by (relative) power in every frequency band."""
    bands = bands or BANDS
    spec = compute_psd(raw, picks=picks if not exclude_bads else [
        c for c in raw.pick_names(picks) if c not in raw.bads], window_sec=2.0, average="median")
    nyq_bands = {k: v for k, v in bands.items() if v[0] < spec.freqs[-1]}
    power = spec.band_power(nyq_bands, relative=relative)
    absolute = spec.band_power(nyq_bands, relative=False)
    unit = "fraction" if relative else "µV²"
    out = {}
    for b, vals in power.items():
        out[b] = ChannelRanking(spec.ch_names, vals, f"{b}_{'relative' if relative else 'absolute'}_power",
                                True, unit if relative else " µV²",
                                f"{b} band {nyq_bands[b][0]:g}-{nyq_bands[b][1]:g} Hz",
                                {"absolute_uV2": absolute[b]}, spec.positions)
    return out


def dominant_bands(raw, bands: dict | None = None, picks="eeg") -> list[dict]:
    """For each channel: its dominant band and peak frequency."""
    bands = bands or BANDS
    spec = compute_psd(raw, picks=picks, window_sec=2.0, average="median")
    rel = spec.band_power({k: v for k, v in bands.items() if v[0] < spec.freqs[-1]}, relative=True)
    names = list(rel)
    mat = np.column_stack([rel[b] for b in names])
    peaks = spec.peak_frequency(1.0, min(45.0, spec.freqs[-1]))
    return [{"channel": c, "dominant_band": names[int(np.nanargmax(mat[i]))],
             "dominant_fraction": float(np.nanmax(mat[i])), "peak_hz": float(peaks[i])}
            for i, c in enumerate(spec.ch_names)]


def rank_by_quality(raw, **kwargs) -> ChannelRanking:
    from .channel_quality import assess_channel_quality

    rep = assess_channel_quality(raw, **kwargs)
    return ChannelRanking(rep.ch_names, rep.scores, "quality_score", True, "/100",
                          "0-100 signal quality score (see assess_channel_quality)",
                          {"grade": rep.grades, "snr_db": rep.metric("snr_db")}, rep.positions)


def rank_by_snr(raw, picks="eeg", **kwargs) -> ChannelRanking:
    from .channel_quality import assess_channel_quality

    rep = assess_channel_quality(raw, picks=picks, use_prep=False, **kwargs)
    return ChannelRanking(rep.ch_names, rep.metric("snr_db"), "snr_db", True, " dB",
                          "Power density 1-30 Hz over the high-frequency noise floor", {}, rep.positions)


def rank_by_connectivity(raw, method: str = "wpli", band=(8.0, 13.0)) -> ChannelRanking:
    from ..analysis.connectivity import compute_connectivity

    con = compute_connectivity(raw, method=method, band=band)
    return ChannelRanking(con.ch_names, con.node_strength(), f"{method}_strength", True, "",
                          f"Mean {method} with all other channels, {band[0]:g}-{band[1]:g} Hz",
                          {}, con.positions)


# ==================================================================== epochs
def _epochs_positions(epochs, idx):
    return {epochs.ch_names[i]: epochs.positions[epochs.ch_names[i]] for i in idx
            if epochs.ch_names[i] in epochs.positions}


def _eeg_idx(epochs, picks=None):
    if picks is not None:
        return epochs._pick(picks)
    idx = [i for i, t in enumerate(epochs.ch_types) if t == "eeg" and epochs.ch_names[i] not in epochs.bads]
    return idx or list(range(len(epochs.ch_names)))


def erp_snr(epochs, window: tuple[float, float] = (0.1, 0.6), picks=None,
            condition: str | None = None, n_permutations: int = 200,
            random_state=0) -> ChannelRanking:
    """Signal-to-noise ratio of the averaged response per channel (dB).

    Noise is estimated with the plus/minus-average idea (Schimmel, 1967):
    inverting half of the trials cancels the time-locked response and leaves
    noise at the level it has in the average. Here the noise power is the mean
    over ``n_permutations`` random balanced sign patterns, which is far more
    stable than a single alternating pattern. SNR = 10 log10(P_evoked /
    P_noise), so 0 dB means "no response above noise".
    """
    ep = epochs[condition] if condition else epochs
    if ep.n_epochs < 4:
        raise ValueError("ERP SNR needs at least 4 epochs")
    rng = check_random_state(random_state)
    idx = _eeg_idx(ep, picks)
    mask = (ep.times >= window[0]) & (ep.times <= window[1])
    data = ep.data[:, idx][:, :, mask]
    n = data.shape[0] - data.shape[0] % 2
    data = data[:n]
    evoked = data.mean(axis=0)
    base_signs = np.repeat([1.0, -1.0], n // 2)
    noise_pow = np.zeros(len(idx))
    for _ in range(n_permutations):
        signs = rng.permutation(base_signs)
        pm = np.tensordot(signs, data, axes=(0, 0)) / n
        noise_pow += (pm ** 2).mean(axis=-1)
    noise_pow /= n_permutations
    ev_pow = (evoked ** 2).mean(axis=-1)
    snr = 10 * np.log10(np.maximum(ev_pow, 1e-24) / np.maximum(noise_pow, 1e-24))
    peak_idx = np.argmax(np.abs(evoked), axis=1)
    names = [ep.ch_names[i] for i in idx]
    return ChannelRanking(names, snr, "erp_snr_db", True, " dB",
                          f"Evoked power / sign-flip noise power, {window[0]:g}-{window[1]:g} s"
                          + (f", condition '{condition}'" if condition else ""),
                          {"peak_latency_s": ep.times[mask][peak_idx],
                           "peak_amplitude_uv": evoked[np.arange(len(idx)), peak_idx],
                           "n_epochs": n}, _epochs_positions(ep, idx))


def erd_ers(epochs, band: tuple[float, float] = (8.0, 13.0), baseline: tuple[float, float] | None = None,
            window: tuple[float, float] | None = None, picks=None,
            condition: str | None = None) -> ChannelRanking:
    """Event-related (de)synchronisation in % of baseline band power (Pfurtscheller).

    Negative values are desynchronisation (ERD). Channels are ranked by the
    size of the change, so the most responsive channel comes first.
    """
    ep = epochs[condition] if condition else epochs
    idx = _eeg_idx(ep, picks)
    times = ep.times
    baseline = baseline or (times[0], min(0.0, times[-1]))
    window = window or (max(0.0, times[0]), times[-1])
    x = filter_data(ep.data[:, idx], ep.sfreq, band[0], band[1])
    power = (x ** 2).mean(axis=0)  # average band power over trials: (ch, time)
    b = (times >= baseline[0]) & (times <= baseline[1])
    w = (times >= window[0]) & (times <= window[1])
    if not b.any() or not w.any():
        raise ValueError("baseline/window outside the epoch time range")
    base = power[:, b].mean(axis=1)
    act = power[:, w].mean(axis=1)
    pct = 100.0 * (act - base) / np.where(base > 0, base, np.nan)
    names = [ep.ch_names[i] for i in idx]
    r = ChannelRanking(names, np.abs(pct), "abs_erd_ers_pct", True, " %",
                       f"|band-power change| {band[0]:g}-{band[1]:g} Hz, window {window[0]:g}-{window[1]:g} s "
                       f"vs baseline {baseline[0]:g}-{baseline[1]:g} s"
                       + (f", condition '{condition}'" if condition else ""),
                       {"erd_ers_pct": pct}, _epochs_positions(ep, idx))
    return r


def _labels(epochs, labels):
    y = np.asarray(epochs.labels if labels is None else labels)
    classes = np.unique(y)
    if len(classes) < 2:
        raise ValueError("Need at least two conditions (event types) to compare")
    return y, classes


def epoch_features(epochs, idx, feature: str = "log_bandpower", bands: dict | None = None,
                   window: tuple[float, float] | None = None, n_bins: int = 5) -> tuple[np.ndarray, list[str]]:
    """Per-channel features: ``(n_epochs, n_channels, n_features)`` and feature names.

    ``'log_bandpower'``: log10 band power in each band inside ``window``.
    ``'erp'``: mean amplitude in ``n_bins`` equal time bins inside ``window``.
    """
    times = epochs.times
    window = window or (max(0.0, times[0]), times[-1])
    mask = (times >= window[0]) & (times <= window[1])
    if feature == "log_bandpower":
        from .._spectral_core import band_power, welch_psd

        bands = bands or {"mu": (8.0, 13.0), "beta": (13.0, 30.0)}
        seg = epochs.data[:, idx][:, :, mask]
        freqs, p = welch_psd(seg, epochs.sfreq, window_sec=min(1.0, seg.shape[-1] / epochs.sfreq))
        feats = np.stack([np.log10(np.maximum(band_power(freqs, p, rng), 1e-20)) for rng in bands.values()],
                         axis=-1)
        return feats, list(bands)
    if feature == "erp":
        seg = epochs.data[:, idx][:, :, mask]
        edges = np.linspace(0, seg.shape[-1], n_bins + 1).astype(int)
        feats = np.stack([seg[:, :, a:b].mean(axis=-1) for a, b in zip(edges[:-1], edges[1:])], axis=-1)
        t = times[mask]
        return feats, [f"{t[a]:.2f}-{t[b - 1]:.2f}s" for a, b in zip(edges[:-1], edges[1:])]
    raise ValueError("feature must be 'log_bandpower' or 'erp'")


def fisher_score(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Between-class over within-class variance, per column of ``x`` (n_samples, n_features)."""
    classes = np.unique(y)
    mu = x.mean(axis=0)
    between = np.zeros(x.shape[1:])
    within = np.zeros(x.shape[1:])
    for c in classes:
        xc = x[y == c]
        between += len(xc) * (xc.mean(axis=0) - mu) ** 2
        within += len(xc) * xc.var(axis=0)
    return between / np.where(within > 0, within, np.nan)


def discriminability(epochs, labels=None, feature: str = "log_bandpower", bands: dict | None = None,
                     window: tuple[float, float] | None = None, picks=None) -> ChannelRanking:
    """How well each channel separates the conditions (Fisher score, AUC, p-value).

    Each channel gets the score of its best feature (band or time bin);
    ``details`` holds that feature's name, the ROC AUC and a t-test / ANOVA
    p-value.
    """
    y, classes = _labels(epochs, labels)
    idx = _eeg_idx(epochs, picks)
    feats, feat_names = epoch_features(epochs, idx, feature, bands, window)
    n_ep, n_ch, n_f = feats.shape
    fs = fisher_score(feats.reshape(n_ep, -1), y).reshape(n_ch, n_f)
    best = np.nanargmax(np.nan_to_num(fs, nan=-1), axis=1)
    auc = np.full(n_ch, np.nan)
    pval = np.full(n_ch, np.nan)
    for c in range(n_ch):
        v = feats[:, c, best[c]]
        groups = [v[y == k] for k in classes]
        if len(classes) == 2:
            u = stats.mannwhitneyu(groups[0], groups[1], alternative="two-sided").statistic
            a = u / (len(groups[0]) * len(groups[1]))
            auc[c] = max(a, 1 - a)
            pval[c] = stats.ttest_ind(groups[0], groups[1], equal_var=False).pvalue
        else:
            pval[c] = stats.f_oneway(*groups).pvalue
    names = [epochs.ch_names[i] for i in idx]
    return ChannelRanking(names, fs[np.arange(n_ch), best], "fisher_score", True, "",
                          f"Class separability of {feature} features ({', '.join(map(str, classes))})",
                          {"best_feature": [feat_names[b] for b in best], "auc": auc, "p_value": pval},
                          _epochs_positions(epochs, idx))


# ------------------------------------------------------------- decoding
class ShrinkageLDA:
    """Linear discriminant analysis with Ledoit-Wolf covariance shrinkage."""

    def fit(self, x: np.ndarray, y: np.ndarray) -> "ShrinkageLDA":
        self.classes_ = np.unique(y)
        means = np.array([x[y == c].mean(axis=0) for c in self.classes_])
        centered = np.concatenate([x[y == c] - means[k] for k, c in enumerate(self.classes_)])
        cov = ledoit_wolf(centered)
        priors = np.array([(y == c).mean() for c in self.classes_])
        self.coef_ = np.linalg.solve(cov, means.T).T
        self.intercept_ = -0.5 * np.einsum("kf,kf->k", means, self.coef_) + np.log(priors)
        return self

    def decision_function(self, x: np.ndarray) -> np.ndarray:
        return x @ self.coef_.T + self.intercept_

    def predict(self, x: np.ndarray) -> np.ndarray:
        return self.classes_[np.argmax(self.decision_function(x), axis=1)]


def ledoit_wolf(x: np.ndarray) -> np.ndarray:
    """Ledoit-Wolf shrunk covariance of centred data ``x`` (n_samples, n_features)."""
    n, p = x.shape
    emp = x.T @ x / n
    mu = np.trace(emp) / p
    x2 = x ** 2
    beta_ = (x2.T @ x2).sum() / n
    delta_ = (emp ** 2).sum()
    beta = (beta_ - delta_) / (n * p) if n > 0 else 0.0
    delta = (delta_ - 2 * mu * np.trace(emp) + p * mu ** 2) / p
    shrink = 0.0 if delta <= 0 else min(max(beta / delta, 0.0), 1.0)
    return (1 - shrink) * emp + shrink * mu * np.eye(p)


def stratified_folds(y: np.ndarray, n_folds: int, rng) -> list[np.ndarray]:
    folds = [[] for _ in range(n_folds)]
    for c in np.unique(y):
        members = rng.permutation(np.flatnonzero(y == c))
        for k, i in enumerate(members):
            folds[k % n_folds].append(i)
    return [np.array(sorted(f), dtype=int) for f in folds if f]


def cross_val_accuracy(x: np.ndarray, y: np.ndarray, n_folds: int = 5, n_repeats: int = 3,
                       random_state=0) -> float:
    """Balanced accuracy of shrinkage LDA with repeated stratified k-fold CV."""
    rng = check_random_state(random_state)
    classes = np.unique(y)
    n_folds = max(2, min(n_folds, min((y == c).sum() for c in classes)))
    accs = []
    for _ in range(n_repeats):
        pred = np.empty_like(y)
        for test in stratified_folds(y, n_folds, rng):
            train = np.setdiff1d(np.arange(len(y)), test)
            mu, sd = x[train].mean(axis=0), x[train].std(axis=0) + 1e-12
            model = ShrinkageLDA().fit((x[train] - mu) / sd, y[train])
            pred[test] = model.predict((x[test] - mu) / sd)
        accs.append(np.mean([(pred[y == c] == c).mean() for c in classes]))
    return float(np.mean(accs))


def decoding_accuracy(epochs, labels=None, feature: str = "log_bandpower", bands: dict | None = None,
                      window: tuple[float, float] | None = None, n_folds: int = 5, n_repeats: int = 3,
                      picks=None, random_state=0) -> ChannelRanking:
    """Cross-validated accuracy of a classifier that sees one channel only.

    A shrinkage-LDA is trained on each channel's features and evaluated
    with repeated stratified k-fold cross-validation (balanced accuracy).
    ``details`` holds the chance level, a binomial p-value per channel and
    the accuracy obtained with all channels together.
    """
    y, classes = _labels(epochs, labels)
    idx = _eeg_idx(epochs, picks)
    feats, _ = epoch_features(epochs, idx, feature, bands, window)
    acc = np.array([cross_val_accuracy(feats[:, c], y, n_folds, n_repeats, random_state)
                    for c in range(len(idx))])
    chance = 1.0 / len(classes)
    n = len(y)
    pvals = np.array([stats.binomtest(int(round(a * n)), n, chance, alternative="greater").pvalue for a in acc])
    all_acc = cross_val_accuracy(feats.reshape(n, -1), y, n_folds, n_repeats, random_state)
    names = [epochs.ch_names[i] for i in idx]
    return ChannelRanking(names, acc, "decoding_accuracy", True, "",
                          f"Balanced accuracy of shrinkage-LDA on single-channel {feature} features "
                          f"({n_folds}-fold CV x {n_repeats})",
                          {"p_value": pvals, "chance_level": chance, "all_channels_accuracy": all_acc,
                           "n_epochs": n}, _epochs_positions(epochs, idx))


# ================================================================ dispatch
def rank_channels(inst, by: str = "quality", **kwargs) -> ChannelRanking:
    """Rank channels of a :class:`RawEEG` or :class:`Epochs` by ``by``.

    ``by``: ``'quality'``, ``'snr'``, a band name (``'delta'``, ``'theta'``,
    ``'alpha'``, ``'beta'``, ``'gamma'``) or ``'band:LOW-HIGH'``,
    ``'connectivity'`` (raw data); ``'erp'``, ``'erd'``, ``'discriminability'``,
    ``'decoding'`` (epochs).
    """
    from ..core.epochs import Epochs

    key = by.lower()
    if isinstance(inst, Epochs):
        funcs = {"erp": erp_snr, "erp_snr": erp_snr, "erd": erd_ers, "ers": erd_ers, "erd_ers": erd_ers,
                 "discriminability": discriminability, "fisher": discriminability,
                 "decoding": decoding_accuracy, "accuracy": decoding_accuracy}
        if key not in funcs:
            raise ValueError(f"For epochs, 'by' must be one of {sorted(funcs)}")
        return funcs[key](inst, **kwargs)
    if key == "quality":
        return rank_by_quality(inst, **kwargs)
    if key == "snr":
        return rank_by_snr(inst, **kwargs)
    if key == "connectivity":
        return rank_by_connectivity(inst, **kwargs)
    m = re.match(r"^band:([\d.]+)-([\d.]+)$", key)
    if m:
        lo, hi = float(m.group(1)), float(m.group(2))
        return band_activity(inst, {key: (lo, hi)}, **kwargs)[key]
    if key in BANDS:
        return band_activity(inst, **kwargs)[key]
    raise ValueError(f"Unknown ranking {by!r}; see rank_channels docstring")
