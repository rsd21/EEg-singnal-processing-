"""Lab Streaming Layer XDF files (optional, needs ``pyxdf``)."""

from __future__ import annotations

import os

import numpy as np

from ..core.annotations import Annotations
from ..core.channels import clean_channel_names, infer_channel_type
from ..core.raw import RawEEG
from ..utils import logger


def _first(x, default=None):
    if isinstance(x, list):
        return x[0] if x else default
    return x if x is not None else default


def _channel_labels(info: dict, n: int) -> list[str]:
    try:
        chans = info["desc"][0]["channels"][0]["channel"]
        labels = [_first(c.get("label"), f"Ch{i + 1}") for i, c in enumerate(chans)]
        if len(labels) == n:
            return labels
    except (KeyError, IndexError, TypeError):
        pass
    return [f"Ch{i + 1}" for i in range(n)]


def read_xdf(path: str | os.PathLike, stream: str | int | None = None, clean_names: bool = True) -> RawEEG:
    """Read the EEG stream of an XDF recording; marker streams become annotations.

    ``stream`` selects a stream by name or index; by default the EEG-typed
    stream with the most channels is used.
    """
    try:
        import pyxdf
    except ImportError as err:
        raise ImportError("Reading XDF files needs pyxdf: pip install pyxdf") from err
    path = os.fspath(path)
    streams, _ = pyxdf.load_xdf(path)
    numeric = [s for s in streams if np.asarray(s["time_series"]).ndim == 2
               and np.asarray(s["time_series"]).dtype.kind in "fiu" and len(s["time_stamps"]) > 1]
    if not numeric:
        raise ValueError(f"No numeric streams in {path}")
    if isinstance(stream, int):
        chosen = streams[stream]
    elif isinstance(stream, str):
        chosen = next(s for s in streams if _first(s["info"]["name"]) == stream)
    else:
        eeg = [s for s in numeric if str(_first(s["info"].get("type"), "")).upper() == "EEG"]
        chosen = max(eeg or numeric, key=lambda s: np.asarray(s["time_series"]).shape[1])
    ts = np.asarray(chosen["time_series"], dtype=float)
    stamps = np.asarray(chosen["time_stamps"], dtype=float)
    nominal = float(_first(chosen["info"].get("nominal_srate"), 0) or 0)
    effective = (len(stamps) - 1) / (stamps[-1] - stamps[0])
    sfreq = nominal if nominal > 0 else effective
    if nominal > 0 and abs(effective - nominal) / nominal > 0.01:
        logger.warning("XDF effective rate %.3f Hz differs from nominal %.3f Hz", effective, nominal)
    labels = _channel_labels(chosen["info"], ts.shape[1])
    names = clean_channel_names(labels) if clean_names else labels
    ann = Annotations()
    t0 = stamps[0]
    for s in streams:
        series = s["time_series"]
        if s is chosen or len(s["time_stamps"]) == 0:
            continue
        is_marker = isinstance(series, list) or np.asarray(series).dtype.kind in "OUS"
        if not is_marker:
            continue
        for t, v in zip(s["time_stamps"], series):
            label = v[0] if isinstance(v, (list, tuple, np.ndarray)) else v
            ann.append(float(t) - t0, 0.0, str(label))
    ann = ann.crop(0.0, ts.shape[0] / sfreq, shift=False) if len(ann) else ann
    types = [infer_channel_type(n) for n in names]
    return RawEEG(ts.T, sfreq, names, types, annotations=ann,
                  meta={"filename": path, "format": "xdf", "original_ch_names": labels,
                        "stream": _first(chosen["info"].get("name"))})
