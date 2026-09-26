"""Events, epochs (trials cut around events) and evoked responses (ERPs)."""

from __future__ import annotations

from typing import Iterable, Sequence

import numpy as np

from ..utils import format_table
from .annotations import Annotations, auto_event_id


# ------------------------------------------------------------------ events
def find_stim_events(signal: np.ndarray, mask: int | None = None,
                     min_samples: int = 1) -> np.ndarray:
    """Events from a trigger channel: every change to a non-zero value.

    Returns an ``(n, 3)`` int array ``[sample, previous_value, new_value]``.
    ``mask`` keeps only these bits (BioSemi uses ``0xFFFF`` for triggers).
    """
    x = np.rint(np.nan_to_num(np.asarray(signal, dtype=float))).astype(np.int64)
    if mask is not None:
        x = x & int(mask)
    # A constant offset on the trigger line (e.g. BioSemi status bits) is not an event.
    if x.size:
        values, counts = np.unique(x, return_counts=True)
        baseline = values[np.argmax(counts)]
        if baseline != 0:
            x = np.where(x == baseline, 0, x)
    change = np.flatnonzero(np.diff(x) != 0) + 1
    if x.size and x[0] != 0:
        change = np.concatenate([[0], change])
    rows = []
    for k, s in enumerate(change):
        val = x[s]
        if val == 0:
            continue
        nxt = change[k + 1] if k + 1 < len(change) else x.size
        if nxt - s < min_samples:
            continue
        rows.append((s, x[s - 1] if s > 0 else 0, val))
    return np.array(rows, dtype=int).reshape(-1, 3)


def find_events(raw, stim_channel: str | None = None, event_id: dict[str, int] | None = None,
                mask: int | None = None, min_duration: float = 0.0,
                source: str = "auto") -> tuple[np.ndarray, dict[str, int]]:
    """Find events in a recording.

    ``source='auto'`` uses the annotations when they contain events (their
    names are more descriptive) and otherwise the stim/trigger channel;
    ``'annotations'`` or ``'stim'`` force one of them. Returns
    ``(events, event_id)`` where ``events`` is ``(n, 3)`` ``[sample, previous,
    code]`` and ``event_id`` maps names to codes.
    """
    has_ann = len(raw.annotations.drop_bad()) > 0
    stim_idx = [raw.channel_index(stim_channel)] if stim_channel is not None else \
        [i for i, t in enumerate(raw.ch_types) if t == "stim"]
    use_ann = source == "annotations" or (source == "auto" and has_ann and stim_channel is None)
    if use_ann:
        return raw.annotations.to_events(raw.sfreq, event_id)
    events = np.zeros((0, 3), dtype=int)
    if stim_idx:
        if mask is None and raw.meta.get("format") == "bdf":
            mask = 0xFFFF
        min_samples = max(1, int(round(min_duration * raw.sfreq)))
        events = find_stim_events(raw.data[stim_idx[0]], mask=mask, min_samples=min_samples)
        if len(events):
            codes = sorted(set(events[:, 2].tolist()))
            ids = event_id or {str(c): c for c in codes}
            keep = np.isin(events[:, 2], list(ids.values()))
            return events[keep], {k: v for k, v in ids.items() if v in set(events[keep, 2].tolist())}
    if source != "stim" and has_ann:
        return raw.annotations.to_events(raw.sfreq, event_id)
    return events, {}


# ------------------------------------------------------------------ epochs
class Epochs:
    """Fixed-length segments cut around events.

    Parameters
    ----------
    raw : RawEEG
    events : array (n, 3), optional
        Defaults to :func:`find_events` on ``raw``.
    event_id : dict, optional
        ``{name: code}``; only these events are epoched.
    tmin, tmax : float
        Window around each event in seconds.
    baseline : (float|None, float|None) or None
        Interval subtracted from each epoch (``(None, 0)`` = pre-stimulus).
    picks : picks, optional
        Channels to keep (default: all data channels).
    reject : float or dict, optional
        Peak-to-peak threshold in µV. A number applies to good EEG channels
        only (EOG is expected to be large); a dict maps channel types to
        thresholds, e.g. ``{'eeg': 150, 'eog': 250}``. Channels in
        ``raw.bads`` never cause rejection.
    flat : float, optional
        Drop epochs where any good EEG channel's peak-to-peak is below this value.
    reject_by_annotation : bool
        Drop epochs overlapping ``BAD*`` annotations.
    """

    def __init__(self, raw, events: np.ndarray | None = None, event_id: dict[str, int] | None = None,
                 tmin: float = -0.2, tmax: float = 0.8,
                 baseline: tuple[float | None, float | None] | None = (None, 0.0),
                 picks="data", reject: float | dict | None = None, flat: float | None = None,
                 reject_by_annotation: bool = True, detrend: bool = False):
        if events is None:
            events, found_id = find_events(raw)
            event_id = event_id or found_id
        events = np.asarray(events, dtype=int).reshape(-1, 3)
        if event_id is None:
            event_id = {str(c): int(c) for c in sorted(set(events[:, 2].tolist()))}
        elif isinstance(event_id, (list, tuple)):
            event_id = {str(c): int(c) for c in event_id}
        codes = set(event_id.values())
        events = events[np.isin(events[:, 2], list(codes))]
        if tmax <= tmin:
            raise ValueError("tmax must be greater than tmin")
        idx = raw.pick_indices(picks)
        self.sfreq = raw.sfreq
        self.ch_names = [raw.ch_names[i] for i in idx]
        self.ch_types = [raw.ch_types[i] for i in idx]
        self.positions = {n: raw.positions[n] for n in self.ch_names if n in raw.positions}
        self.bads = [b for b in raw.bads if b in self.ch_names]
        self.event_id = dict(event_id)
        self.tmin = tmin
        start_off = int(round(tmin * raw.sfreq))
        stop_off = int(round(tmax * raw.sfreq)) + 1
        self.times = np.arange(start_off, stop_off) / raw.sfreq
        bad_mask = raw.annotations.bad_mask(raw.n_times, raw.sfreq) if reject_by_annotation else None

        data, kept, drop_log = [], [], []
        for ev in events:
            a, b = ev[0] + start_off, ev[0] + stop_off
            if a < 0 or b > raw.n_times:
                drop_log.append("OUT_OF_BOUNDS")
                continue
            if bad_mask is not None and bad_mask[a:b].any():
                drop_log.append("BAD_ANNOTATION")
                continue
            seg = raw.data[idx, a:b].astype(float, copy=True)
            if detrend:
                from scipy.signal import detrend as _detrend

                seg = _detrend(seg, axis=-1, type="linear")
            if baseline is not None:
                seg = _apply_baseline(seg, self.times, baseline)
            reason = self._reject_reason(seg, reject, flat)
            if reason:
                drop_log.append(reason)
                continue
            drop_log.append("")
            data.append(seg)
            kept.append(ev)
        self._data = np.array(data).reshape(len(data), len(idx), len(self.times))
        self.events = np.array(kept, dtype=int).reshape(-1, 3)
        self.drop_log = drop_log
        self.baseline = baseline

    def _reject_reason(self, seg: np.ndarray, reject, flat) -> str:
        ptp = seg.max(axis=1) - seg.min(axis=1)
        good = np.array([n not in self.bads for n in self.ch_names])
        if reject is not None:
            if isinstance(reject, dict):
                thr = np.array([reject.get(t, np.inf) for t in self.ch_types], dtype=float)
            else:
                thr = np.array([reject if t == "eeg" else np.inf for t in self.ch_types], dtype=float)
            over = (ptp > thr) & good
            if over.any():
                return "REJECT " + ",".join(np.array(self.ch_names)[over][:3])
        if flat is not None:
            under = (ptp < flat) & good & np.array([t == "eeg" for t in self.ch_types])
            if under.any():
                return "FLAT " + ",".join(np.array(self.ch_names)[under][:3])
        return ""

    # ------------------------------------------------------------ accessors
    @classmethod
    def from_array(cls, data: np.ndarray, sfreq: float, ch_names: Sequence[str],
                   labels: Sequence[int] | None = None, tmin: float = 0.0,
                   event_id: dict[str, int] | None = None, ch_types: Sequence[str] | None = None,
                   positions: dict | None = None) -> "Epochs":
        """Wrap an existing ``(n_epochs, n_channels, n_times)`` array."""
        data = np.asarray(data, dtype=float)
        if data.ndim != 3:
            raise ValueError("data must be (n_epochs, n_channels, n_times)")
        n_ep, n_ch, n_t = data.shape
        labels = np.ones(n_ep, dtype=int) if labels is None else np.asarray(labels, dtype=int)
        obj = cls.__new__(cls)
        obj.sfreq = float(sfreq)
        obj.ch_names = list(ch_names)
        obj.ch_types = list(ch_types) if ch_types is not None else ["eeg"] * n_ch
        obj.positions = dict(positions or {})
        obj.bads = []
        obj.event_id = event_id or {str(c): int(c) for c in sorted(set(labels.tolist()))}
        obj.tmin = tmin
        obj.times = tmin + np.arange(n_t) / sfreq
        obj._data = data
        obj.events = np.column_stack([np.arange(n_ep) * n_t, np.zeros(n_ep, int), labels]).astype(int)
        obj.drop_log = [""] * n_ep
        obj.baseline = None
        return obj

    @property
    def data(self) -> np.ndarray:
        return self._data

    @property
    def labels(self) -> np.ndarray:
        return self.events[:, 2].copy()

    @property
    def n_epochs(self) -> int:
        return self._data.shape[0]

    def __len__(self) -> int:
        return self.n_epochs

    def __repr__(self) -> str:
        counts = ", ".join(f"{k}: {int((self.labels == v).sum())}" for k, v in self.event_id.items())
        return (f"<Epochs | {self.n_epochs} epochs ({counts}) | {len(self.ch_names)} channels | "
                f"{self.times[0]:.3f}-{self.times[-1]:.3f} s | dropped {self.n_dropped}>")

    @property
    def n_dropped(self) -> int:
        return sum(1 for d in self.drop_log if d)

    def get_data(self, picks=None) -> np.ndarray:
        idx = self._pick(picks)
        return self._data[:, idx].copy()

    def _pick(self, picks) -> list[int]:
        if picks is None:
            return list(range(len(self.ch_names)))
        if isinstance(picks, str):
            picks = [picks]
        out = []
        for p in picks:
            if isinstance(p, (int, np.integer)):
                out.append(int(p))
            elif p in self.ch_names:
                out.append(self.ch_names.index(p))
            elif p in ("eeg", "eog", "ecg", "emg", "misc", "resp"):
                out += [i for i, t in enumerate(self.ch_types) if t == p]
            else:
                raise KeyError(f"Channel {p!r} not in epochs")
        return out

    def __getitem__(self, key) -> "Epochs":
        """Select epochs by condition name(s), indices, slice or boolean mask."""
        if isinstance(key, str) or (isinstance(key, (list, tuple)) and key and isinstance(key[0], str)):
            names = [key] if isinstance(key, str) else list(key)
            codes = [self.event_id[n] for n in names]
            sel = np.flatnonzero(np.isin(self.labels, codes))
            new_id = {n: self.event_id[n] for n in names}
        else:
            sel = np.arange(self.n_epochs)[key]
            sel = np.atleast_1d(sel)
            new_id = dict(self.event_id)
        out = self.copy()
        out._data = self._data[sel]
        out.events = self.events[sel]
        out.event_id = new_id
        return out

    def copy(self) -> "Epochs":
        import copy as _copy

        return _copy.deepcopy(self)

    def pick(self, picks) -> "Epochs":
        idx = self._pick(picks)
        out = self.copy()
        out._data = self._data[:, idx]
        out.ch_names = [self.ch_names[i] for i in idx]
        out.ch_types = [self.ch_types[i] for i in idx]
        out.positions = {n: p for n, p in self.positions.items() if n in out.ch_names}
        out.bads = [b for b in self.bads if b in out.ch_names]
        return out

    def crop(self, tmin: float | None = None, tmax: float | None = None) -> "Epochs":
        mask = np.ones(len(self.times), dtype=bool)
        if tmin is not None:
            mask &= self.times >= tmin - 1e-12
        if tmax is not None:
            mask &= self.times <= tmax + 1e-12
        out = self.copy()
        out._data = self._data[:, :, mask]
        out.times = self.times[mask]
        out.tmin = float(out.times[0])
        return out

    def average(self, by_condition: bool = False):
        """Mean over epochs -> :class:`Evoked` (or a dict of them per condition)."""
        if by_condition:
            return {name: self[name].average() for name in self.event_id
                    if np.any(self.labels == self.event_id[name])}
        if self.n_epochs == 0:
            raise ValueError("No epochs to average")
        return Evoked(self._data.mean(axis=0), self.times, self.ch_names, self.sfreq,
                      nave=self.n_epochs, comment=" + ".join(self.event_id), ch_types=self.ch_types,
                      positions=self.positions)

    def summary_table(self) -> str:
        rows = [{"condition": k, "code": v, "n_epochs": int((self.labels == v).sum())}
                for k, v in self.event_id.items()]
        return format_table(rows)


def _apply_baseline(seg: np.ndarray, times: np.ndarray, baseline) -> np.ndarray:
    b0, b1 = baseline
    b0 = times[0] if b0 is None else b0
    b1 = times[-1] if b1 is None else b1
    mask = (times >= b0 - 1e-12) & (times <= b1 + 1e-12)
    if mask.any():
        seg -= seg[..., mask].mean(axis=-1, keepdims=True)
    return seg


# ------------------------------------------------------------------ evoked
class Evoked:
    """An averaged event-related response (ERP)."""

    def __init__(self, data: np.ndarray, times: np.ndarray, ch_names: Sequence[str], sfreq: float,
                 nave: int = 1, comment: str = "", ch_types: Sequence[str] | None = None,
                 positions: dict | None = None):
        self.data = np.asarray(data, dtype=float)
        self.times = np.asarray(times, dtype=float)
        self.ch_names = list(ch_names)
        self.sfreq = float(sfreq)
        self.nave = int(nave)
        self.comment = comment
        self.ch_types = list(ch_types) if ch_types is not None else ["eeg"] * len(self.ch_names)
        self.positions = dict(positions or {})

    def __repr__(self) -> str:
        return (f"<Evoked '{self.comment}' | nave={self.nave} | {len(self.ch_names)} channels | "
                f"{self.times[0]:.3f}-{self.times[-1]:.3f} s>")

    def gfp(self) -> np.ndarray:
        """Global field power: spatial standard deviation at each time point."""
        eeg = [i for i, t in enumerate(self.ch_types) if t == "eeg"] or list(range(len(self.ch_names)))
        return self.data[eeg].std(axis=0)

    def get_peak(self, ch_name: str | None = None, tmin: float | None = None,
                 tmax: float | None = None, mode: str = "abs") -> tuple[str, float, float]:
        """Return ``(channel, latency_s, amplitude)`` of the largest peak.

        ``mode`` is ``'abs'``, ``'pos'`` or ``'neg'``.
        """
        mask = np.ones(len(self.times), dtype=bool)
        if tmin is not None:
            mask &= self.times >= tmin
        if tmax is not None:
            mask &= self.times <= tmax
        chans = [self.ch_names.index(ch_name)] if ch_name else \
            [i for i, t in enumerate(self.ch_types) if t == "eeg"] or list(range(len(self.ch_names)))
        seg = self.data[np.ix_(chans, np.flatnonzero(mask))]
        score = {"abs": np.abs(seg), "pos": seg, "neg": -seg}[mode]
        ci, ti = np.unravel_index(np.argmax(score), score.shape)
        return self.ch_names[chans[ci]], float(self.times[mask][ti]), float(seg[ci, ti])

    def plot(self, **kwargs):
        from ..viz.erp import plot_evoked

        return plot_evoked(self, **kwargs)

    def plot_topomap(self, times: float | Iterable[float] = "peaks", **kwargs):
        from ..viz.erp import plot_evoked_topomaps

        return plot_evoked_topomaps(self, times=times, **kwargs)


def make_fixed_length_events(raw, duration: float = 2.0, overlap: float = 0.0,
                             code: int = 1) -> np.ndarray:
    """Evenly spaced events, e.g. to cut resting-state data into epochs."""
    step = duration - overlap
    if step <= 0:
        raise ValueError("overlap must be smaller than duration")
    starts = np.arange(0, raw.duration - duration + 1e-9, step)
    samples = np.round(starts * raw.sfreq).astype(int)
    return np.column_stack([samples, np.zeros_like(samples), np.full_like(samples, code)])


def make_fixed_length_epochs(raw, duration: float = 2.0, overlap: float = 0.0, picks="data",
                             reject: float | None = None, reject_by_annotation: bool = True) -> Epochs:
    events = make_fixed_length_events(raw, duration, overlap)
    return Epochs(raw, events, {"segment": 1}, tmin=0.0, tmax=duration - 1.0 / raw.sfreq,
                  baseline=None, picks=picks, reject=reject, reject_by_annotation=reject_by_annotation)


__all__ = ["find_events", "find_stim_events", "Epochs", "Evoked", "Annotations",
           "make_fixed_length_events", "make_fixed_length_epochs", "auto_event_id"]
