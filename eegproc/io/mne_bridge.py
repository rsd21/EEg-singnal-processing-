"""Optional bridge to MNE-Python.

MNE is not required by eegproc. When it is installed, it lets eegproc read the
formats it has no native reader for (FIF, GDF, Neuroscan CNT, EGI/MFF, Nihon
Kohden, Persyst, Nicolet, eXimia, ...) and hand data back and forth.
"""

from __future__ import annotations

import os

import numpy as np

from ..core.annotations import Annotations
from ..core.channels import VOLTAGE_TYPES, clean_channel_names
from ..core.montage import fit_sphere
from ..core.raw import RawEEG

_MNE_TO_OURS = {"eeg": "eeg", "eog": "eog", "ecg": "ecg", "emg": "emg", "stim": "stim",
                "resp": "resp", "misc": "misc", "seeg": "eeg", "ecog": "eeg", "dbs": "eeg",
                "bio": "misc", "syst": "misc", "temperature": "misc", "gsr": "misc"}


def _require_mne():
    try:
        import mne
    except ImportError as err:
        raise ImportError("This file format is read through MNE-Python: pip install mne") from err
    return mne


def from_mne(mne_raw, clean_names: bool = False) -> RawEEG:
    """Convert an ``mne.io.Raw`` object to :class:`RawEEG` (volts -> microvolts)."""
    types = mne_raw.get_channel_types()
    ours = [_MNE_TO_OURS.get(t, "misc") for t in types]
    data = mne_raw.get_data()
    units = []
    for k, t in enumerate(ours):
        if t in VOLTAGE_TYPES:
            data[k] *= 1e6
            units.append("uV")
        else:
            units.append("" if t == "stim" else "a.u.")
    names = list(mne_raw.ch_names)
    new_names = clean_channel_names(names) if clean_names else names
    positions = {}
    locs = np.array([ch["loc"][:3] for ch in mne_raw.info["chs"]])
    ok = np.all(np.isfinite(locs), axis=1) & (np.linalg.norm(locs, axis=1) > 0)
    if ok.sum() >= 4:
        center, _ = fit_sphere(locs[ok])
        for k in np.flatnonzero(ok):
            v = locs[k] - center
            positions[new_names[k]] = v / np.linalg.norm(v)
    ann = Annotations()
    first = mne_raw.first_samp / mne_raw.info["sfreq"]
    for a in mne_raw.annotations:
        ann.append(a["onset"] - first, a["duration"], a["description"])
    meta = {"filename": mne_raw.filenames[0] if getattr(mne_raw, "filenames", None) else None,
            "format": "mne", "meas_date": mne_raw.info.get("meas_date"),
            "highpass": mne_raw.info.get("highpass"), "lowpass": mne_raw.info.get("lowpass")}
    bads = [new_names[names.index(b)] for b in mne_raw.info["bads"] if b in names]
    return RawEEG(data, mne_raw.info["sfreq"], new_names, ours, units, ann, positions, bads, meta)


def to_mne(raw: RawEEG):
    """Convert :class:`RawEEG` to ``mne.io.RawArray`` (microvolts -> volts)."""
    mne = _require_mne()
    data = raw.data.copy()
    for k, t in enumerate(raw.ch_types):
        if t in VOLTAGE_TYPES and raw.units[k] == "uV":
            data[k] *= 1e-6
    info = mne.create_info(raw.ch_names, raw.sfreq, raw.ch_types)
    mne_raw = mne.io.RawArray(data, info, verbose=False)
    if raw.positions:
        # Unit sphere -> 95 mm head radius in MNE head coordinates.
        pos = {n: raw.positions[n] * 0.095 for n in raw.ch_names if n in raw.positions}
        montage = mne.channels.make_dig_montage(ch_pos=pos, coord_frame="head")
        mne_raw.set_montage(montage, on_missing="ignore")
    if len(raw.annotations):
        mne_raw.set_annotations(mne.Annotations(raw.annotations.onset, raw.annotations.duration,
                                                raw.annotations.description))
    mne_raw.info["bads"] = list(raw.bads)
    return mne_raw


def read_with_mne(path: str | os.PathLike, clean_names: bool = True, **kwargs) -> RawEEG:
    """Read any format MNE understands and convert it."""
    mne = _require_mne()
    mne_raw = mne.io.read_raw(os.fspath(path), preload=True, verbose="error", **kwargs)
    raw = from_mne(mne_raw, clean_names=clean_names)
    raw.meta["filename"] = os.fspath(path)
    raw.meta["format"] = "mne:" + os.path.splitext(os.fspath(path))[1].lstrip(".").lower()
    return raw


def write_fif(raw: RawEEG, path: str | os.PathLike) -> str:
    path = os.fspath(path)
    to_mne(raw).save(path, overwrite=True, verbose="error")
    return path
