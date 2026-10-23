"""NumPy (.npy/.npz), MATLAB (.mat, EEGLAB .set) and HDF5 (.h5/.hdf5) files.

Generic array files rarely follow a fixed layout, so the readers search for
the signal matrix and for common names of the sampling rate and channel
labels (``fs``, ``srate``, ``sfreq``, ``labels``, ``ch_names`` ...).
"""

from __future__ import annotations

import os
from typing import Any

import numpy as np

from ..core.annotations import Annotations
from ..core.channels import clean_channel_names, infer_channel_type
from ..core.raw import RawEEG
from ..utils import logger

SFREQ_KEYS = ("sfreq", "fs", "srate", "sampling_rate", "samplingrate", "sample_rate", "samplerate",
              "sampling_frequency", "samplingfrequency", "freq", "rate", "hz")
DATA_KEYS = ("data", "eeg", "signals", "signal", "x", "raw", "samples", "y")
NAME_KEYS = ("ch_names", "channels", "labels", "channel_names", "chan_names", "chanlabels",
             "electrodes", "names", "clab", "channel_labels")


def _norm(key: str) -> str:
    return key.lower().replace(" ", "").replace("-", "_")


def _find(mapping: dict, keys) -> Any:
    lowered = {_norm(k): k for k in mapping}
    for k in keys:
        if _norm(k) in lowered:
            return mapping[lowered[_norm(k)]]
    return None


def _as_str_list(x) -> list[str] | None:
    if x is None:
        return None
    arr = np.asarray(x, dtype=object).ravel()
    out = []
    for v in arr:
        if isinstance(v, bytes):
            v = v.decode("utf-8", errors="replace")
        elif isinstance(v, np.ndarray):
            v = "".join(chr(int(c)) for c in v.ravel()) if v.dtype.kind in "iu" else str(v.squeeze())
        out.append(str(v).strip())
    return out


def _orient(data: np.ndarray, n_names: int | None) -> np.ndarray:
    """Return (n_channels, n_samples); channels are the smaller dimension unless names say otherwise."""
    if data.ndim == 1:
        return data[np.newaxis]
    if n_names is not None:
        if data.shape[0] == n_names:
            return data
        if data.shape[1] == n_names:
            return data.T
    return data if data.shape[0] <= data.shape[1] else data.T


def _build(data, sfreq, names, source: str, fmt: str, ann: Annotations | None = None,
           clean_names: bool = True, types=None, units=None, positions=None, meta=None) -> RawEEG:
    data = np.asarray(data, dtype=float)
    if data.ndim == 3:
        data, ann = _flatten_epochs(data, sfreq, names, ann)
    data = _orient(data, len(names) if names else None)
    if sfreq is None:
        raise ValueError(f"Could not find the sampling rate in {source}; pass sfreq=...")
    if names is None or len(names) != data.shape[0]:
        names = [f"Ch{i + 1}" for i in range(data.shape[0])]
    labels = list(names)
    names = clean_channel_names(labels) if clean_names else labels
    types = types or [infer_channel_type(n) for n in names]
    m = {"filename": source, "format": fmt, "original_ch_names": labels}
    m.update(meta or {})
    return RawEEG(data, float(sfreq), names, types, units, annotations=ann, positions=positions, meta=m)


def _flatten_epochs(data: np.ndarray, sfreq, names, ann):
    """(ch, time, trials) or (trials, ch, time) -> continuous data with boundaries."""
    n_names = len(names) if names else None
    if n_names is not None and data.shape[1] == n_names:  # (trials, ch, time)
        data = np.transpose(data, (1, 2, 0))
    elif n_names is None and data.shape[0] > data.shape[1] and data.shape[2] > data.shape[1]:
        data = np.transpose(data, (1, 2, 0))
    n_ch, n_t, n_tr = data.shape
    logger.warning("3D array found: %d trials of %d samples concatenated into continuous data", n_tr, n_t)
    cont = data.transpose(0, 2, 1).reshape(n_ch, n_tr * n_t)
    ann = ann or Annotations()
    if sfreq:
        for k in range(1, n_tr):
            ann.append(k * n_t / float(sfreq), 0.0, "BAD boundary")
        for k in range(n_tr):
            ann.append(k * n_t / float(sfreq), 0.0, "trial")
    return cont, ann


# ======================================================================= NumPy
def read_numpy(path: str | os.PathLike, sfreq: float | None = None, ch_names=None,
               clean_names: bool = True) -> RawEEG:
    """Read ``.npy`` (array only, ``sfreq`` required) or ``.npz`` archives."""
    path = os.fspath(path)
    if path.lower().endswith(".npy"):
        data = np.load(path, allow_pickle=False)
        return _build(data, sfreq, list(ch_names) if ch_names is not None else None, path, "npy",
                      clean_names=clean_names)
    with np.load(path, allow_pickle=False) as z:
        content = {k: z[k] for k in z.files}
    data = _find(content, DATA_KEYS)
    if data is None:
        arrays = [v for v in content.values() if v.ndim >= 2]
        if not arrays:
            raise ValueError(f"No 2D array found in {path}")
        data = max(arrays, key=lambda a: a.size)
    fs = sfreq if sfreq is not None else _find(content, SFREQ_KEYS)
    names = list(ch_names) if ch_names is not None else _as_str_list(_find(content, NAME_KEYS))
    types = _as_str_list(content.get("ch_types"))
    units = _as_str_list(content.get("units"))
    ann = None
    if "annot_onset" in content:
        ann = Annotations(content["annot_onset"], content.get("annot_duration", 0.0),
                          _as_str_list(content.get("annot_description")) or [])
    positions = None
    if "positions" in content and names is not None:
        pos = np.asarray(content["positions"], dtype=float)
        positions = {n: p for n, p in zip(names, pos) if np.all(np.isfinite(p))}
    bads = _as_str_list(content.get("bads"))
    raw = _build(data, float(np.asarray(fs).ravel()[0]) if fs is not None else None, names, path, "npz",
                 ann, clean_names=clean_names and types is None, types=types, units=units,
                 positions=positions)
    if bads:
        raw.set_bads([b for b in bads if b in raw.ch_names])
    return raw


def write_numpy(raw: RawEEG, path: str | os.PathLike, picks=None) -> str:
    """Write a self-describing ``.npz`` (data, sfreq, names, types, units, annotations, positions)."""
    path = os.fspath(path)
    idx = raw.pick_indices(picks)
    names = [raw.ch_names[i] for i in idx]
    pos = np.array([raw.positions.get(n, np.full(3, np.nan)) for n in names])
    np.savez_compressed(
        path, data=raw.data[idx], sfreq=np.array(raw.sfreq), ch_names=np.array(names),
        ch_types=np.array([raw.ch_types[i] for i in idx]), units=np.array([raw.units[i] for i in idx]),
        annot_onset=raw.annotations.onset, annot_duration=raw.annotations.duration,
        annot_description=np.array(raw.annotations.description, dtype=str), positions=pos,
        bads=np.array([b for b in raw.bads if b in names], dtype=str))
    return path


# ====================================================================== MATLAB
def _loadmat(path: str) -> dict:
    from scipy.io import loadmat

    try:
        return loadmat(path, squeeze_me=True, struct_as_record=False)
    except NotImplementedError:  # v7.3 files are HDF5
        return {"__hdf5__": True}


def _struct_to_dict(obj) -> dict:
    if isinstance(obj, dict):
        return obj
    if hasattr(obj, "_fieldnames"):
        return {f: getattr(obj, f) for f in obj._fieldnames}
    if isinstance(obj, np.ndarray) and obj.dtype.names:
        return {n: obj[n].squeeze() for n in obj.dtype.names}
    return {}


def read_matlab(path: str | os.PathLike, sfreq: float | None = None, ch_names=None,
                clean_names: bool = True) -> RawEEG:
    """Read a MATLAB ``.mat`` file or an EEGLAB ``.set`` dataset."""
    path = os.fspath(path)
    content = _loadmat(path)
    if content.get("__hdf5__"):
        return read_hdf5(path, sfreq=sfreq, ch_names=ch_names, clean_names=clean_names)
    content = {k: v for k, v in content.items() if not k.startswith("__")}
    eeg = content.get("EEG")
    if eeg is not None or ("srate" in content and "nbchan" in content):
        return _read_eeglab(path, _struct_to_dict(eeg) if eeg is not None else content, clean_names)

    # Search top level and one level into structs.
    flat = dict(content)
    for k, v in content.items():
        sub = _struct_to_dict(v)
        for sk, sv in sub.items():
            flat.setdefault(sk, sv)
    data = _find(flat, DATA_KEYS)
    if data is None or np.asarray(data).ndim < 2 or np.asarray(data).dtype.kind not in "fiu":
        arrays = [np.asarray(v) for v in flat.values()
                  if isinstance(v, np.ndarray) and v.dtype.kind in "fiu" and v.ndim >= 2]
        if not arrays:
            raise ValueError(f"No numeric 2D array found in {path}")
        data = max(arrays, key=lambda a: a.size)
    fs = sfreq if sfreq is not None else _find(flat, SFREQ_KEYS)
    names = list(ch_names) if ch_names is not None else _as_str_list(_find(flat, NAME_KEYS))
    ann = None
    if "annot_onset" in content:
        ann = Annotations(np.atleast_1d(content["annot_onset"]), np.atleast_1d(content.get("annot_duration", 0.0)),
                          _as_str_list(content.get("annot_description")) or [])
    types = _as_str_list(content.get("ch_types"))
    units = _as_str_list(content.get("units"))
    return _build(data, float(np.asarray(fs).ravel()[0]) if fs is not None else None, names, path, "mat", ann,
                  clean_names=clean_names and types is None, types=types, units=units)


def _read_eeglab(path: str, eeg: dict, clean_names: bool) -> RawEEG:
    sfreq = float(eeg["srate"])
    nbchan = int(eeg["nbchan"])
    data = eeg["data"]
    if isinstance(data, str):  # separate .fdt file (float32, column-major)
        fdt = os.path.join(os.path.dirname(path), data)
        pnts = int(eeg["pnts"])
        trials = int(eeg.get("trials", 1) or 1)
        data = np.fromfile(fdt, dtype="<f4").reshape((nbchan, pnts * trials), order="F")
    data = np.asarray(data, dtype=float)
    trials = int(eeg.get("trials", 1) or 1)
    if data.ndim == 3:
        trials = data.shape[2]
        data = data.reshape(nbchan, -1, order="F")
    labels, positions = [], {}
    chanlocs = np.atleast_1d(eeg.get("chanlocs", []))
    for k in range(nbchan):
        loc = _struct_to_dict(chanlocs[k]) if k < len(chanlocs) else {}
        label = str(loc.get("labels", f"Ch{k + 1}")).strip() or f"Ch{k + 1}"
        labels.append(label)
        try:
            xyz = np.array([float(loc["X"]), float(loc["Y"]), float(loc["Z"])])
            if np.all(np.isfinite(xyz)) and np.any(xyz != 0):
                positions[label] = np.array([-xyz[1], xyz[0], xyz[2]])  # EEGLAB: +X nose, +Y left
        except (KeyError, TypeError, ValueError):
            pass
    ann = Annotations()
    for ev in np.atleast_1d(eeg.get("event", [])):
        d = _struct_to_dict(ev)
        if "latency" not in d:
            continue
        try:
            onset = (float(d["latency"]) - 1) / sfreq
        except (TypeError, ValueError):
            continue
        dur = float(d.get("duration", 0) or 0) / sfreq if np.size(d.get("duration", 0)) == 1 else 0.0
        ann.append(onset, dur, str(d.get("type", "event")))
    if trials > 1:  # epoched dataset: mark where one trial ends and the next begins
        pnts = data.shape[1] // trials
        ann.append(np.arange(1, trials) * pnts / sfreq, 0.0, "BAD boundary")
        ann = ann.sorted()
    names = clean_channel_names(labels) if clean_names else labels
    pos = {n: positions[l] for n, l in zip(names, labels) if l in positions}
    types = [infer_channel_type(n) for n in names]
    return RawEEG(data, sfreq, names, types, annotations=ann, positions=pos,
                  meta={"filename": path, "format": "eeglab", "original_ch_names": labels})


def write_matlab(raw: RawEEG, path: str | os.PathLike, picks=None) -> str:
    """Write a ``.mat`` file with ``data``, ``sfreq``, ``ch_names`` and annotations."""
    from scipy.io import savemat

    path = os.fspath(path)
    idx = raw.pick_indices(picks)
    savemat(path, {
        "data": raw.data[idx], "sfreq": raw.sfreq,
        "ch_names": np.array([raw.ch_names[i] for i in idx], dtype=object),
        "ch_types": np.array([raw.ch_types[i] for i in idx], dtype=object),
        "units": np.array([raw.units[i] for i in idx], dtype=object),
        "annot_onset": raw.annotations.onset, "annot_duration": raw.annotations.duration,
        "annot_description": np.array(raw.annotations.description, dtype=object),
    }, do_compression=True)
    return path


# ======================================================================== HDF5
def read_hdf5(path: str | os.PathLike, sfreq: float | None = None, ch_names=None,
              clean_names: bool = True) -> RawEEG:
    """Read an HDF5 file (including MATLAB v7.3) by searching for the signal matrix."""
    try:
        import h5py
    except ImportError as err:
        raise ImportError("Reading HDF5 / MATLAB v7.3 files needs h5py: pip install h5py") from err
    path = os.fspath(path)
    datasets: dict[str, np.ndarray] = {}
    attrs: dict[str, Any] = {}
    with h5py.File(path, "r") as f:
        attrs.update({k: v for k, v in f.attrs.items()})

        def visit(name, obj):
            if isinstance(obj, h5py.Dataset):
                leaf = name.split("/")[-1]
                if h5py.check_dtype(ref=obj.dtype) is not None:
                    return  # MATLAB cell arrays are object references; skip them
                try:
                    value = obj[()]
                except (TypeError, ValueError):
                    return
                if obj.attrs.get("MATLAB_class", b"") in (b"char", "char"):
                    value = "".join(chr(int(c)) for c in np.asarray(value).ravel())
                datasets.setdefault(leaf, value)
                for k, v in obj.attrs.items():
                    attrs.setdefault(k, v)

        f.visititems(visit)
    content = {**datasets, **{k: v for k, v in attrs.items() if k not in datasets}}
    data = _find(datasets, DATA_KEYS)
    if data is None or np.asarray(data).ndim < 2:
        arrays = [np.asarray(v) for v in datasets.values()
                  if isinstance(v, np.ndarray) and v.dtype.kind in "fiu" and v.ndim >= 2]
        if not arrays:
            raise ValueError(f"No numeric 2D dataset found in {path}")
        data = max(arrays, key=lambda a: a.size)
    fs = sfreq if sfreq is not None else _find(content, SFREQ_KEYS)
    names = list(ch_names) if ch_names is not None else _as_str_list(_find(content, NAME_KEYS))
    ann = None
    if "annot_onset" in datasets:
        ann = Annotations(datasets["annot_onset"], datasets.get("annot_duration", 0.0),
                          _as_str_list(datasets.get("annot_description")) or [])
    types = _as_str_list(datasets.get("ch_types"))
    fs_val = float(np.asarray(fs).ravel()[0]) if fs is not None else None
    return _build(np.asarray(data), fs_val, names, path, "hdf5", ann,
                  clean_names=clean_names and types is None, types=types)


def write_hdf5(raw: RawEEG, path: str | os.PathLike, picks=None) -> str:
    try:
        import h5py
    except ImportError as err:
        raise ImportError("Writing HDF5 needs h5py: pip install h5py") from err
    path = os.fspath(path)
    idx = raw.pick_indices(picks)
    str_dt = h5py.string_dtype()
    with h5py.File(path, "w") as f:
        f.create_dataset("data", data=raw.data[idx], compression="gzip")
        f.attrs["sfreq"] = raw.sfreq
        f.create_dataset("ch_names", data=np.array([raw.ch_names[i] for i in idx], dtype=object), dtype=str_dt)
        f.create_dataset("ch_types", data=np.array([raw.ch_types[i] for i in idx], dtype=object), dtype=str_dt)
        f.create_dataset("annot_onset", data=raw.annotations.onset)
        f.create_dataset("annot_duration", data=raw.annotations.duration)
        f.create_dataset("annot_description", data=np.array(raw.annotations.description, dtype=object),
                         dtype=str_dt)
    return path
