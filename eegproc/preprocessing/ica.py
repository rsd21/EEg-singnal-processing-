"""Independent Component Analysis (FastICA) for artifact removal.

Implementation of symmetric FastICA (Hyvärinen & Oja, 2000) after PCA
whitening, plus helpers that flag ocular, cardiac and muscle components so
they can be removed from the data.

Typical use::

    ica = ICA(n_components=0.99, random_state=0).fit(raw)
    eye, scores = ica.find_bads_eog(raw)
    muscle, _ = ica.find_bads_muscle(raw)
    clean = ica.apply(raw, exclude=eye + muscle)
"""

from __future__ import annotations

import numpy as np
from scipy import stats

from .._spectral_core import welch_psd
from ..utils import check_random_state, format_table, logger, robust_zscore
from .artifacts import FRONTAL_EOG_PROXIES
from .filters import filter_data


def _sym_decorrelation(w: np.ndarray) -> np.ndarray:
    s, u = np.linalg.eigh(w @ w.T)
    s = np.clip(s, np.finfo(float).tiny, None)
    return (u * (1.0 / np.sqrt(s))) @ u.T @ w


def _nonlinearity(fun: str):
    if fun == "logcosh":
        def g(u):
            t = np.tanh(u)
            return t, 1.0 - t ** 2
    elif fun == "exp":
        def g(u):
            e = np.exp(-(u ** 2) / 2)
            return u * e, (1 - u ** 2) * e
    elif fun == "cube":
        def g(u):
            return u ** 3, 3 * u ** 2
    else:
        raise ValueError("fun must be 'logcosh', 'exp' or 'cube'")
    return g


def fastica(z: np.ndarray, fun: str = "logcosh", max_iter: int = 500, tol: float = 1e-5,
            random_state=None) -> tuple[np.ndarray, int, bool]:
    """Symmetric FastICA on whitened data ``z`` (k, n). Returns ``(W, n_iter, converged)``."""
    rng = check_random_state(random_state)
    k, n = z.shape
    w = _sym_decorrelation(rng.standard_normal((k, k)))
    g = _nonlinearity(fun)
    for it in range(1, max_iter + 1):
        wx = w @ z
        gwx, g_wx = g(wx)
        w_new = _sym_decorrelation((gwx @ z.T) / n - g_wx.mean(axis=1)[:, None] * w)
        lim = np.max(np.abs(np.abs(np.einsum("ij,ij->i", w_new, w)) - 1.0))
        w = w_new
        if lim < tol:
            return w, it, True
    return w, max_iter, False


class ICA:
    """Independent component decomposition of EEG channels.

    Parameters
    ----------
    n_components : int, float or None
        Number of components (int), fraction of variance to keep (float in
        (0, 1)) or ``None`` for the data rank.
    fun : ``'logcosh'`` | ``'exp'`` | ``'cube'``
        FastICA contrast function.
    fit_highpass : float or None
        High-pass (Hz) applied to the copy used for fitting; 1 Hz makes ICA
        much more stable. The decomposition is then applied to the data you
        pass to :meth:`apply`.
    """

    def __init__(self, n_components: int | float | None = None, fun: str = "logcosh",
                 max_iter: int = 1000, tol: float = 1e-4, random_state=None,
                 fit_highpass: float | None = 1.0):
        self.n_components = n_components
        self.fun = fun
        self.max_iter = max_iter
        self.tol = tol
        self.random_state = random_state
        self.fit_highpass = fit_highpass
        self.exclude: list[int] = []
        self.labels_: dict[int, str] = {}

    # ---------------------------------------------------------------- fit
    def fit(self, raw, picks="eeg", exclude_bads: bool = True, reject_by_annotation: bool = True,
            decim: int | None = None) -> "ICA":
        idx = raw.pick_indices(picks, exclude_bads=exclude_bads)
        if len(idx) < 2:
            raise ValueError("ICA needs at least 2 channels")
        self.ch_names = [raw.ch_names[i] for i in idx]
        self.positions = {n: raw.positions[n] for n in self.ch_names if n in raw.positions}
        self.sfreq = raw.sfreq
        x = raw.data[idx]
        if self.fit_highpass:
            x = filter_data(x, raw.sfreq, self.fit_highpass, None)
        if reject_by_annotation and len(raw.annotations):
            good = ~raw.annotations.bad_mask(raw.n_times, raw.sfreq)
            if good.sum() > 10 * len(idx):
                x = x[:, good]
        if decim and decim > 1:
            x = x[:, ::decim]
        self.mean_ = x.mean(axis=1)
        xc = x - self.mean_[:, None]
        cov = xc @ xc.T / xc.shape[1]
        evals, evecs = np.linalg.eigh(cov)
        order = np.argsort(evals)[::-1]
        evals, evecs = np.clip(evals[order], 0, None), evecs[:, order]
        rank = int((evals > evals[0] * 1e-8).sum())
        nc = self.n_components
        if nc is None:
            k = rank
        elif isinstance(nc, float) and 0 < nc < 1:
            k = int(np.searchsorted(np.cumsum(evals) / evals.sum(), nc) + 1)
        else:
            k = int(nc)
        k = max(1, min(k, rank))
        if xc.shape[1] < 20 * k * k:
            logger.warning("Only %d samples for %d components; ICA may be unreliable "
                           "(rule of thumb: >= 20 x n_components^2)", xc.shape[1], k)
        whiten = evecs[:, :k].T / np.sqrt(evals[:k])[:, None]
        z = whiten @ xc
        w, n_iter, ok = fastica(z, self.fun, self.max_iter, self.tol, self.random_state)
        if not ok:
            logger.warning("FastICA did not converge in %d iterations; try more max_iter", n_iter)
        unmixing = w @ whiten
        mixing = np.linalg.pinv(unmixing)
        # order by variance explained in channel space (sources have unit variance)
        var = (mixing ** 2).sum(axis=0)
        order = np.argsort(var)[::-1]
        self.unmixing_ = unmixing[order]
        self.mixing_ = mixing[:, order]
        self.explained_variance_ = var[order] / np.trace(cov)
        self.n_components_ = k
        self.n_iter_ = n_iter
        self.pca_explained_variance_ = evals[:k] / evals.sum()
        return self

    def _check(self):
        if not hasattr(self, "unmixing_"):
            raise RuntimeError("Call ICA.fit(raw) first")

    def _data(self, raw) -> np.ndarray:
        return raw.data[[raw.channel_index(c) for c in self.ch_names]]

    def get_sources(self, raw) -> np.ndarray:
        """Component time courses ``(n_components, n_samples)``."""
        self._check()
        return self.unmixing_ @ (self._data(raw) - self.mean_[:, None])

    def get_components(self) -> np.ndarray:
        """Scalp maps (mixing matrix columns): ``(n_channels, n_components)``."""
        self._check()
        return self.mixing_.copy()

    # --------------------------------------------------------------- apply
    def apply(self, raw, exclude=None):
        """Return a copy of ``raw`` with the excluded components removed."""
        self._check()
        exclude = sorted(set(self.exclude if exclude is None else exclude))
        out = raw.copy()
        if not exclude:
            return out
        idx = [raw.channel_index(c) for c in self.ch_names]
        src = self.get_sources(raw)[exclude]
        out.data[idx] = raw.data[idx] - self.mixing_[:, exclude] @ src
        labels = [self.labels_.get(i, "?") for i in exclude]
        out.history.append(f"ICA removed components {exclude} ({', '.join(labels)})")
        return out

    # ------------------------------------------------- component labelling
    def find_bads_eog(self, raw, ch_name: str | list[str] | None = None, threshold: float = 3.0,
                      abs_threshold: float = 0.6, min_r: float = 0.3) -> tuple[list[int], np.ndarray]:
        """Components correlated with EOG (or frontal proxy) channels.

        A component is flagged if |r| >= ``abs_threshold``, or if |r| >= ``min_r``
        and its robust z-score across components exceeds ``threshold``.
        Correlations are computed on 1-10 Hz band-passed signals.
        """
        self._check()
        if ch_name is None:
            names = raw.pick_names("eog") or [c for c in FRONTAL_EOG_PROXIES if c in raw.ch_names][:2]
        else:
            names = [ch_name] if isinstance(ch_name, str) else list(ch_name)
        if not names:
            raise ValueError("No EOG or frontal channels available; pass ch_name=...")
        return self._flag_by_reference(raw, names, (1.0, 10.0), threshold, abs_threshold, min_r, "eye")

    def find_bads_ecg(self, raw, ch_name: str | None = None, threshold: float = 3.0,
                      abs_threshold: float = 0.5, min_r: float = 0.25) -> tuple[list[int], np.ndarray]:
        """Components correlated with an ECG channel (QRS band, 8-16 Hz)."""
        self._check()
        names = [ch_name] if ch_name else raw.pick_names("ecg")
        if not names:
            raise ValueError("No ECG channel found; pass ch_name=...")
        return self._flag_by_reference(raw, names, (8.0, 16.0), threshold, abs_threshold, min_r, "heart")

    def _flag_by_reference(self, raw, names, band, threshold, abs_threshold, min_r, label):
        src = filter_data(self.get_sources(raw), raw.sfreq, *band)
        ref = filter_data(raw.data[[raw.channel_index(n) for n in names]], raw.sfreq, *band)
        src = (src - src.mean(1, keepdims=True)) / (src.std(1, keepdims=True) + 1e-12)
        ref = (ref - ref.mean(1, keepdims=True)) / (ref.std(1, keepdims=True) + 1e-12)
        r = np.abs(src @ ref.T / src.shape[1]).max(axis=1)
        z = robust_zscore(r)
        bad = [int(i) for i in np.flatnonzero((r >= abs_threshold) | ((z > threshold) & (r >= min_r)))]
        for i in bad:
            self.labels_[i] = label
        return bad, r

    def find_bads_muscle(self, raw, slope_threshold: float = -0.3,
                         fmin: float = 7.0, fmax: float = 45.0) -> tuple[list[int], np.ndarray]:
        """Components whose spectrum does not fall off with frequency (EMG-like).

        Brain components have a clearly negative log-log spectral slope;
        muscle components are flat or rising between ``fmin`` and ``fmax``.
        """
        self._check()
        fmax = min(fmax, 0.9 * raw.sfreq / 2)
        freqs, p = welch_psd(self.get_sources(raw), raw.sfreq, window_sec=2.0, fmin=fmin, fmax=fmax)
        lf = np.log10(freqs)
        slopes = np.array([np.polyfit(lf, np.log10(np.maximum(row, 1e-20)), 1)[0] for row in p])
        bad = [int(i) for i in np.flatnonzero(slopes > slope_threshold)]
        for i in bad:
            self.labels_.setdefault(i, "muscle")
        return bad, slopes

    def component_table(self, raw) -> list[dict]:
        """Per-component summary: variance explained, kurtosis, EOG r, spectral slope, label."""
        self._check()
        src = self.get_sources(raw)
        kurt = stats.kurtosis(src, axis=1)
        try:
            _, eog_r = self.find_bads_eog(raw)
        except ValueError:
            eog_r = np.full(self.n_components_, np.nan)
        _, slopes = self.find_bads_muscle(raw)
        rows = []
        for i in range(self.n_components_):
            rows.append({"component": i, "variance_pct": 100 * self.explained_variance_[i],
                         "kurtosis": float(kurt[i]), "eog_r": float(eog_r[i]),
                         "spectral_slope": float(slopes[i]), "label": self.labels_.get(i, "brain/other"),
                         "excluded": i in self.exclude})
        return rows

    def summary(self, raw) -> str:
        return format_table(self.component_table(raw))

    def plot_components(self, **kwargs):
        from ..viz.topomap import plot_ica_components

        return plot_ica_components(self, **kwargs)

    def __repr__(self) -> str:
        if not hasattr(self, "unmixing_"):
            return "<ICA | not fitted>"
        return (f"<ICA | {self.n_components_} components from {len(self.ch_names)} channels | "
                f"excluded {self.exclude}>")


def remove_artifacts_ica(raw, n_components=0.99, eog: bool = True, muscle: bool = False,
                         ecg: bool = True, random_state=0):
    """One-call ICA cleaning: fit, flag eye (and optionally muscle/heart) components, remove them.

    Returns ``(clean_raw, ica)``.
    """
    ica = ICA(n_components=n_components, random_state=random_state).fit(raw)
    exclude: list[int] = []
    if eog:
        try:
            exclude += ica.find_bads_eog(raw)[0]
        except ValueError:
            logger.info("No EOG/frontal channels; ocular components not searched")
    if ecg and raw.pick_names("ecg"):
        exclude += ica.find_bads_ecg(raw)[0]
    if muscle:
        exclude += ica.find_bads_muscle(raw)[0]
    ica.exclude = sorted(set(exclude))
    return ica.apply(raw), ica
