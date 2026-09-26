"""The :class:`RawEEG` container for continuous multichannel recordings."""

from __future__ import annotations

import copy as _copy
import os
from typing import Any, Iterable, Sequence

import numpy as np

from ..utils import format_table
from .annotations import Annotations
from .channels import (CHANNEL_TYPES, DATA_TYPES, VOLTAGE_TYPES, clean_channel_name,
                       default_unit, infer_channel_type)
from .montage import Montage, get_montage

TYPE_KEYWORDS = set(CHANNEL_TYPES) | {"all", "data", "good"}


class RawEEG:
    """Continuous EEG recording held in memory.

    Parameters
    ----------
    data : array, shape (n_channels, n_samples)
        Signals. Voltage channels (EEG/EOG/ECG/EMG) are stored in microvolts.
    sfreq : float
        Sampling frequency in Hz.
    ch_names, ch_types, units : list of str, optional
        Channel labels, types (``eeg``, ``eog``, ``ecg``, ``emg``, ``stim``,
        ``resp``, ``misc``) and units. Types are guessed from the labels when
        omitted.
    annotations : Annotations, optional
        Events and marked segments.
    positions : dict, optional
        ``{channel: (x, y, z)}`` electrode positions on the unit sphere.
    bads : list of str, optional
        Channels marked as bad.
    meta : dict, optional
        Free-form metadata (source file, format, subject, recording date ...).

    Methods that change the signal (``filter``, ``crop``, ``pick`` ...) return a
    new object and leave the original untouched. Metadata setters such as
    :meth:`set_bads` work in place and return ``self`` for chaining.
    """

    def __init__(self, data, sfreq: float, ch_names: Sequence[str] | None = None,
                 ch_types: Sequence[str] | None = None, units: Sequence[str] | None = None,
                 annotations: Annotations | None = None, positions: dict | None = None,
                 bads: Iterable[str] | None = None, meta: dict | None = None,
                 history: Iterable[str] | None = None):
        data = np.asarray(data, dtype=float)
        if data.ndim == 1:
            data = data[np.newaxis, :]
        if data.ndim != 2:
            raise ValueError(f"data must be 2D (n_channels, n_samples), got shape {data.shape}")
        sfreq = float(sfreq)
        if not np.isfinite(sfreq) or sfreq <= 0:
            raise ValueError(f"sfreq must be a positive number, got {sfreq}")
        n_ch = data.shape[0]
        ch_names = [f"Ch{i + 1}" for i in range(n_ch)] if ch_names is None else [str(c) for c in ch_names]
        if len(ch_names) != n_ch:
            raise ValueError(f"{len(ch_names)} channel names for {n_ch} channels")
        if len(set(ch_names)) != len(ch_names):
            dup = sorted({c for c in ch_names if ch_names.count(c) > 1})
            raise ValueError(f"Duplicate channel names: {dup}")
        if ch_types is None:
            ch_types = [infer_channel_type(c, u) for c, u in
                        zip(ch_names, units if units is not None else [None] * n_ch)]
        ch_types = [str(t).lower() for t in ch_types]
        bad_types = sorted(set(ch_types) - set(CHANNEL_TYPES))
        if bad_types:
            raise ValueError(f"Unknown channel types {bad_types}; use {CHANNEL_TYPES}")
        if len(ch_types) != n_ch:
            raise ValueError("ch_types must have one entry per channel")
        units = [default_unit(t) for t in ch_types] if units is None else [str(u) for u in units]
        if len(units) != n_ch:
            raise ValueError("units must have one entry per channel")

        self._data = data
        self.sfreq = sfreq
        self.ch_names = ch_names
        self.ch_types = ch_types
        self.units = units
        self.annotations = annotations.copy() if annotations is not None else Annotations()
        self.positions: dict[str, np.ndarray] = {}
        if positions:
            for name, xyz in positions.items():
                if name in ch_names:
                    xyz = np.asarray(xyz, dtype=float)
                    n = np.linalg.norm(xyz)
                    if n > 0 and np.all(np.isfinite(xyz)):
                        self.positions[name] = xyz / n
        self.bads = [b for b in (bads or []) if b in ch_names]
        self.meta: dict[str, Any] = dict(meta or {})
        self.history: list[str] = list(history or [])

    # ------------------------------------------------------------ properties
    @property
    def data(self) -> np.ndarray:
        """The signal array ``(n_channels, n_samples)`` (not a copy)."""
        return self._data

    @data.setter
    def data(self, value) -> None:
        value = np.asarray(value, dtype=float)
        if value.shape[0] != self.n_channels:
            raise ValueError("New data must keep the number of channels")
        self._data = value

    @property
    def n_channels(self) -> int:
        return self._data.shape[0]

    @property
    def n_times(self) -> int:
        return self._data.shape[1]

    n_samples = n_times

    @property
    def duration(self) -> float:
        return self.n_times / self.sfreq

    @property
    def times(self) -> np.ndarray:
        return np.arange(self.n_times) / self.sfreq

    @property
    def filename(self) -> str | None:
        return self.meta.get("filename")

    def __len__(self) -> int:
        return self.n_times

    def __repr__(self) -> str:
        types = ", ".join(f"{n} {t.upper()}" for t, n in self.type_counts().items())
        name = os.path.basename(self.filename) if self.filename else "in-memory"
        return (f"<RawEEG '{name}' | {types} | {self.sfreq:g} Hz | {self.duration:.2f} s"
                f" | {len(self.annotations)} annotations | bads: {self.bads or 'none'}>")

    def type_counts(self) -> dict[str, int]:
        out: dict[str, int] = {}
        for t in self.ch_types:
            out[t] = out.get(t, 0) + 1
        return out

    # --------------------------------------------------------------- picking
    def pick_indices(self, picks=None, exclude_bads: bool = False) -> list[int]:
        """Resolve ``picks`` into channel indices.

        ``picks`` may be ``None``/``'all'`` (every channel), ``'data'`` (all but
        stim), ``'good'`` (EEG channels not marked bad), a channel type
        (``'eeg'``, ``'eog'`` ...), a channel name, an index, a slice, a boolean
        mask or a list mixing names, types and indices.
        """
        n = self.n_channels
        if picks is None:
            idx = list(range(n))
        elif isinstance(picks, slice):
            idx = list(range(n))[picks]
        elif isinstance(picks, (str, int, np.integer)):
            idx = self._resolve_one(picks)
        else:
            arr = np.asarray(picks)
            if arr.dtype == bool:
                if arr.size != n:
                    raise ValueError("Boolean picks must have one entry per channel")
                idx = list(np.flatnonzero(arr))
            else:
                idx = []
                for p in picks:
                    for i in self._resolve_one(p):
                        if i not in idx:
                            idx.append(i)
        if exclude_bads:
            idx = [i for i in idx if self.ch_names[i] not in self.bads]
        return idx

    def _resolve_one(self, p) -> list[int]:
        if isinstance(p, (int, np.integer)):
            i = int(p)
            if not -self.n_channels <= i < self.n_channels:
                raise IndexError(f"Channel index {i} out of range")
            return [i % self.n_channels]
        p = str(p)
        if p in self.ch_names:
            return [self.ch_names.index(p)]
        low = p.lower()
        if low == "all":
            return list(range(self.n_channels))
        if low == "data":
            return [i for i, t in enumerate(self.ch_types) if t in DATA_TYPES]
        if low == "good":
            return [i for i, t in enumerate(self.ch_types)
                    if t == "eeg" and self.ch_names[i] not in self.bads]
        if low in CHANNEL_TYPES:
            return [i for i, t in enumerate(self.ch_types) if t == low]
        lower_names = [c.lower() for c in self.ch_names]
        if low in lower_names:
            return [lower_names.index(low)]
        cleaned = clean_channel_name(p)
        cleaned_names = [clean_channel_name(c) for c in self.ch_names]
        if cleaned in cleaned_names:
            return [cleaned_names.index(cleaned)]
        raise KeyError(f"Channel {p!r} not found. Available: {', '.join(self.ch_names)}")

    def pick_names(self, picks=None, exclude_bads: bool = False) -> list[str]:
        return [self.ch_names[i] for i in self.pick_indices(picks, exclude_bads)]

    # ----------------------------------------------------------- data access
    def time_to_index(self, t: float) -> int:
        return int(np.clip(round(t * self.sfreq), 0, self.n_times))

    def get_data(self, picks=None, tmin: float | None = None, tmax: float | None = None,
                 exclude_bads: bool = False, copy: bool = True) -> np.ndarray:
        """Signals for the selected channels between ``tmin`` and ``tmax`` seconds."""
        idx = self.pick_indices(picks, exclude_bads)
        start = 0 if tmin is None else self.time_to_index(tmin)
        stop = self.n_times if tmax is None else self.time_to_index(tmax)
        out = self._data[idx, start:stop]
        return out.copy() if copy else out

    def __getitem__(self, key):
        """``raw['Cz']`` -> 1D signal, ``raw[['Fz', 'Cz'], 0:256]`` -> 2D array."""
        if isinstance(key, tuple):
            picks, samples = key
        else:
            picks, samples = key, slice(None)
        idx = self.pick_indices(picks)
        out = self._data[idx][:, samples]
        if isinstance(picks, (str, int, np.integer)) and len(idx) == 1 and picks not in TYPE_KEYWORDS:
            return out[0]
        return out

    def channel_index(self, name: str) -> int:
        return self._resolve_one(name)[0]

    def get_channel_type(self, name: str) -> str:
        return self.ch_types[self.channel_index(name)]

    # ------------------------------------------------------ copies & subsets
    def copy(self) -> "RawEEG":
        return RawEEG(self._data.copy(), self.sfreq, list(self.ch_names), list(self.ch_types),
                      list(self.units), self.annotations.copy(),
                      {k: v.copy() for k, v in self.positions.items()}, list(self.bads),
                      _copy.deepcopy(self.meta), list(self.history))

    def _new(self, data: np.ndarray, idx: Sequence[int] | None = None, note: str | None = None,
             annotations: Annotations | None = None, sfreq: float | None = None) -> "RawEEG":
        """Build a new RawEEG sharing metadata with this one."""
        idx = list(range(self.n_channels)) if idx is None else list(idx)
        names = [self.ch_names[i] for i in idx]
        out = RawEEG(data, self.sfreq if sfreq is None else sfreq, names,
                     [self.ch_types[i] for i in idx], [self.units[i] for i in idx],
                     self.annotations if annotations is None else annotations,
                     {n: self.positions[n] for n in names if n in self.positions},
                     [b for b in self.bads if b in names], _copy.deepcopy(self.meta),
                     list(self.history))
        if note:
            out.history.append(note)
        return out

    def pick(self, picks, exclude_bads: bool = False) -> "RawEEG":
        """New recording with only the selected channels (in the given order)."""
        idx = self.pick_indices(picks, exclude_bads)
        if not idx:
            raise ValueError(f"No channels match picks={picks!r}")
        return self._new(self._data[idx].copy(), idx, f"pick {len(idx)} channels")

    def pick_types(self, eeg=True, eog=False, ecg=False, emg=False, stim=False, resp=False,
                   misc=False, exclude_bads: bool = False) -> "RawEEG":
        wanted = [t for t, flag in [("eeg", eeg), ("eog", eog), ("ecg", ecg), ("emg", emg),
                                     ("stim", stim), ("resp", resp), ("misc", misc)] if flag]
        return self.pick([i for i, t in enumerate(self.ch_types) if t in wanted], exclude_bads)

    def drop_channels(self, names: str | Iterable[str]) -> "RawEEG":
        drop = set(self.pick_indices([names] if isinstance(names, str) else list(names)))
        keep = [i for i in range(self.n_channels) if i not in drop]
        if not keep:
            raise ValueError("Cannot drop every channel")
        return self._new(self._data[keep].copy(), keep, f"drop channels {sorted(self.ch_names[i] for i in drop)}")

    def reorder_channels(self, names: Sequence[str]) -> "RawEEG":
        return self.pick(list(names))

    def crop(self, tmin: float = 0.0, tmax: float | None = None) -> "RawEEG":
        """New recording restricted to ``[tmin, tmax)`` seconds."""
        tmax = self.duration if tmax is None else tmax
        if tmin < 0 or tmax > self.duration + 1e-9 or tmin >= tmax:
            raise ValueError(f"Invalid crop window [{tmin}, {tmax}] for a {self.duration:.3f} s recording")
        a, b = self.time_to_index(tmin), self.time_to_index(tmax)
        ann = self.annotations.crop(a / self.sfreq, b / self.sfreq, shift=True)
        return self._new(self._data[:, a:b].copy(), None, f"crop {tmin:g}-{tmax:g} s", ann)

    def append(self, other: "RawEEG") -> "RawEEG":
        """Concatenate another recording with the same channels in time."""
        if other.ch_names != self.ch_names or other.sfreq != self.sfreq:
            raise ValueError("Recordings must have the same channels and sampling rate")
        ann = self.annotations.copy().extend(other.annotations.shift(self.duration))
        ann.append(self.duration, 0.0, "BAD boundary")
        data = np.concatenate([self._data, other._data], axis=1)
        return self._new(data, None, "append recording", ann)

    def add_channels(self, data, ch_names: Sequence[str], ch_types=None, units=None) -> "RawEEG":
        data = np.atleast_2d(np.asarray(data, dtype=float))
        if data.shape[1] != self.n_times:
            raise ValueError("New channels must have the same number of samples")
        tmp = RawEEG(data, self.sfreq, ch_names, ch_types, units)
        out = RawEEG(np.vstack([self._data, data]), self.sfreq, self.ch_names + tmp.ch_names,
                     self.ch_types + tmp.ch_types, self.units + tmp.units, self.annotations,
                     self.positions, self.bads, self.meta, self.history)
        out.history.append(f"add channels {list(tmp.ch_names)}")
        return out

    # ---------------------------------------------------- metadata (in place)
    def rename_channels(self, mapping: dict[str, str] | Any) -> "RawEEG":
        """Rename channels in place. ``mapping`` is a dict or a function."""
        new = [mapping(c) if callable(mapping) else mapping.get(c, c) for c in self.ch_names]
        if len(set(new)) != len(new):
            raise ValueError("Renaming would create duplicate channel names")
        rename = dict(zip(self.ch_names, new))
        self.ch_names = new
        self.positions = {rename[k]: v for k, v in self.positions.items()}
        self.bads = [rename[b] for b in self.bads]
        return self

    def standardize_channel_names(self) -> "RawEEG":
        """Clean labels in place (``'EEG Fp1-REF'`` -> ``'Fp1'``, ``'FP1'`` -> ``'Fp1'``)."""
        from .channels import clean_channel_names

        new = clean_channel_names(self.ch_names)
        self.meta.setdefault("original_ch_names", list(self.ch_names))
        return self.rename_channels(dict(zip(self.ch_names, new)))

    def set_channel_types(self, mapping: dict[str, str]) -> "RawEEG":
        for name, t in mapping.items():
            t = t.lower()
            if t not in CHANNEL_TYPES:
                raise ValueError(f"Unknown channel type {t!r}")
            i = self.channel_index(name)
            old = self.ch_types[i]
            self.ch_types[i] = t
            if (old in VOLTAGE_TYPES) != (t in VOLTAGE_TYPES) and self.units[i] in ("uV", "a.u.", ""):
                self.units[i] = default_unit(t)
        return self

    def set_bads(self, bads: Iterable[str]) -> "RawEEG":
        self.bads = [self.ch_names[self.channel_index(b)] for b in bads]
        return self

    def add_bads(self, bads: Iterable[str]) -> "RawEEG":
        for b in bads:
            name = self.ch_names[self.channel_index(b)]
            if name not in self.bads:
                self.bads.append(name)
        return self

    def set_annotations(self, annotations: Annotations) -> "RawEEG":
        self.annotations = annotations.copy()
        return self

    def set_montage(self, montage="standard_1010", on_missing: str = "ignore") -> "RawEEG":
        """Assign electrode positions (in place) from a montage.

        ``montage`` may be ``'standard_1020'``, ``'standard_1010'``, a file path
        (``.elc``, ``.sfp``, ``.loc``, ``.ced``, ``.csv``/``.tsv``), a dict or a
        :class:`Montage`. Channels not found keep no position; with
        ``on_missing='raise'`` a missing EEG channel raises an error.
        """
        mont = get_montage(montage)
        missing = []
        for name, t in zip(self.ch_names, self.ch_types):
            xyz = mont.get(name)
            if xyz is not None:
                self.positions[name] = xyz
            elif t == "eeg":
                missing.append(name)
        if missing and on_missing == "raise":
            raise ValueError(f"No position for EEG channels {missing} in montage {mont.name}")
        self.meta["montage"] = mont.name
        return self

    def has_positions(self, picks="eeg") -> bool:
        names = self.pick_names(picks)
        return bool(names) and all(n in self.positions for n in names)

    def get_positions(self, picks="eeg") -> np.ndarray:
        """(n, 3) positions for the selected channels (NaN where unknown)."""
        return np.array([self.positions.get(n, np.full(3, np.nan)) for n in self.pick_names(picks)]).reshape(-1, 3)

    def get_montage(self) -> Montage:
        return Montage(dict(self.positions), name=self.meta.get("montage", "custom"), normalize=False)

    # -------------------------------------------------------------- summaries
    def channel_table(self) -> list[dict]:
        """One row per channel: type, unit, position, bad flag and amplitude stats."""
        rows = []
        for i, name in enumerate(self.ch_names):
            x = self._data[i]
            finite = x[np.isfinite(x)]
            rows.append({
                "index": i, "name": name, "type": self.ch_types[i], "unit": self.units[i],
                "position": name in self.positions, "bad": name in self.bads,
                "mean": float(finite.mean()) if finite.size else np.nan,
                "std": float(finite.std()) if finite.size else np.nan,
                "min": float(finite.min()) if finite.size else np.nan,
                "max": float(finite.max()) if finite.size else np.nan,
                "nan_samples": int(x.size - finite.size),
            })
        return rows

    def summary(self) -> dict:
        return {
            "filename": self.filename, "format": self.meta.get("format"),
            "sfreq": self.sfreq, "n_channels": self.n_channels, "n_samples": self.n_times,
            "duration_s": self.duration, "channel_types": self.type_counts(),
            "ch_names": list(self.ch_names), "bads": list(self.bads),
            "has_positions": len(self.positions), "annotations": self.annotations.count(),
            "meas_date": str(self.meta.get("meas_date")) if self.meta.get("meas_date") else None,
            "subject": self.meta.get("subject"), "history": list(self.history),
        }

    def describe(self, channels: bool = True) -> str:
        """Human-readable overview of the recording."""
        s = self.summary()
        mins, secs = divmod(self.duration, 60)
        lines = [
            f"File        : {s['filename'] or '(in memory)'}",
            f"Format      : {s['format'] or 'n/a'}",
            f"Sampling    : {self.sfreq:g} Hz ({self.n_times} samples, {int(mins)} min {secs:.2f} s)",
            f"Channels    : {self.n_channels} (" + ", ".join(f"{v} {k}" for k, v in s['channel_types'].items()) + ")",
            f"Positions   : {s['has_positions']} channels with electrode positions",
            f"Bad channels: {', '.join(self.bads) if self.bads else 'none'}",
        ]
        if s["meas_date"]:
            lines.append(f"Recorded    : {s['meas_date']}")
        if s["subject"]:
            subj = s["subject"]
            lines.append("Subject     : " + (", ".join(f"{k}={v}" for k, v in subj.items())
                                            if isinstance(subj, dict) else str(subj)))
        if len(self.annotations):
            ann = ", ".join(f"{k} (x{v})" for k, v in list(s["annotations"].items())[:12])
            lines.append(f"Annotations : {len(self.annotations)} -> {ann}")
        if self.history:
            lines.append("History     : " + " | ".join(self.history))
        if channels:
            lines += ["", format_table(self.channel_table(),
                                       ["index", "name", "type", "unit", "position", "bad",
                                        "mean", "std", "min", "max"])]
        return "\n".join(lines)

    def to_dataframe(self, picks=None, time_column: bool = True):
        """Samples x channels pandas DataFrame (requires pandas)."""
        try:
            import pandas as pd
        except ImportError as err:  # pragma: no cover - depends on environment
            raise ImportError("pandas is required for to_dataframe(); pip install pandas") from err
        idx = self.pick_indices(picks)
        df = pd.DataFrame(self._data[idx].T, columns=[self.ch_names[i] for i in idx])
        if time_column:
            df.insert(0, "time", self.times)
        return df

    # ------------------------------------------------ processing conveniences
    def filter(self, l_freq: float | None, h_freq: float | None, picks="data", **kwargs) -> "RawEEG":
        """Band-pass / high-pass / low-pass filter (see :func:`eegproc.preprocessing.filter_raw`)."""
        from ..preprocessing.filters import filter_raw

        return filter_raw(self, l_freq, h_freq, picks=picks, **kwargs)

    def notch_filter(self, freqs=50.0, picks="data", **kwargs) -> "RawEEG":
        from ..preprocessing.filters import notch_raw

        return notch_raw(self, freqs, picks=picks, **kwargs)

    def resample(self, sfreq: float) -> "RawEEG":
        from ..preprocessing.resample import resample_raw

        return resample_raw(self, sfreq)

    def set_reference(self, ref="average", **kwargs) -> "RawEEG":
        from ..preprocessing.reference import set_reference

        return set_reference(self, ref, **kwargs)

    def interpolate_bads(self, reset_bads: bool = True) -> "RawEEG":
        from ..preprocessing.interpolation import interpolate_bads

        return interpolate_bads(self, reset_bads=reset_bads)

    def compute_psd(self, picks="data", **kwargs):
        from ..analysis.spectral import compute_psd

        return compute_psd(self, picks=picks, **kwargs)

    def assess_quality(self, **kwargs):
        from ..quality.channel_quality import assess_channel_quality

        return assess_channel_quality(self, **kwargs)

    def find_events(self, **kwargs):
        """Events from annotations or the trigger channel -> ``(events, event_id)``."""
        from .epochs import find_events

        return find_events(self, **kwargs)

    def plot(self, **kwargs):
        from ..viz.timeseries import plot_raw

        return plot_raw(self, **kwargs)

    def plot_psd(self, **kwargs):
        from ..viz.spectra import plot_psd

        return plot_psd(self, **kwargs)

    def plot_sensors(self, **kwargs):
        from ..viz.topomap import plot_sensors

        return plot_sensors(self, **kwargs)

    def save(self, path, fmt: str | None = None, **kwargs) -> str:
        """Write to disk; the format follows the file extension (see :func:`eegproc.write_raw`)."""
        from ..io import write_raw

        return write_raw(self, path, fmt=fmt, **kwargs)

    def to_mne(self):
        from ..io.mne_bridge import to_mne

        return to_mne(self)
