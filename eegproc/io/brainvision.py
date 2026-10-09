"""BrainVision (.vhdr / .vmrk / .eeg) reader and writer.

Supports binary INT_16, INT_32, UINT_16 and IEEE_FLOAT_32 data in multiplexed
or vectorized orientation, ASCII data files, per-channel resolutions and
units, and marker files.
"""

from __future__ import annotations

import datetime as _dt
import os
import re

import numpy as np

from ..core.annotations import Annotations
from ..core.channels import clean_channel_names, infer_channel_type, voltage_scale_to_uv
from ..core.raw import RawEEG

_DTYPES = {"INT_16": "<i2", "INT_32": "<i4", "UINT_16": "<u2", "IEEE_FLOAT_32": "<f4",
           "IEEE_FLOAT_64": "<f8"}


def _read_text(path: str) -> str:
    with open(path, "rb") as f:
        raw = f.read()
    for enc in ("utf-8", "latin-1"):
        try:
            return raw.decode(enc)
        except UnicodeDecodeError:
            continue
    return raw.decode("latin-1", errors="replace")


def _parse_ini(text: str) -> dict[str, dict[str, str]]:
    sections: dict[str, dict[str, str]] = {}
    current = None
    for line in text.splitlines():
        s = line.strip()
        if not s or s.startswith(";"):
            continue
        m = re.match(r"^\[(.+)\]$", s)
        if m:
            current = m.group(1).strip()
            sections.setdefault(current, {})
            continue
        if current is not None and "=" in s:
            key, val = s.split("=", 1)
            sections[current][key.strip()] = val.strip()
    return sections


def _split_escaped(value: str) -> list[str]:
    parts = value.split(",")
    return [p.replace(r"\1", ",") for p in parts]


def _resolve(base_dir: str, name: str, fallback_ext: str, vhdr_path: str) -> str:
    candidate = os.path.join(base_dir, name)
    if os.path.exists(candidate):
        return candidate
    alt = os.path.splitext(vhdr_path)[0] + fallback_ext
    if os.path.exists(alt):
        return alt
    return candidate


def read_brainvision(path: str | os.PathLike, clean_names: bool = True) -> RawEEG:
    """Read a BrainVision recording given its ``.vhdr`` (or ``.eeg``/``.vmrk``) path."""
    path = os.fspath(path)
    stem, ext = os.path.splitext(path)
    if ext.lower() != ".vhdr":
        path = stem + ".vhdr"
    if not os.path.exists(path):
        raise FileNotFoundError(f"BrainVision header not found: {path}")
    base = os.path.dirname(path)
    ini = _parse_ini(_read_text(path))
    common = ini.get("Common Infos", {})
    n_ch = int(common["NumberOfChannels"])
    sfreq = 1e6 / float(common["SamplingInterval"])
    data_file = _resolve(base, common.get("DataFile", ""), ".eeg", path)
    marker_file = common.get("MarkerFile")
    fmt = common.get("DataFormat", "BINARY").upper()
    orient = common.get("DataOrientation", "MULTIPLEXED").upper()

    labels, resolutions, units = [], [], []
    ch_info = ini.get("Channel Infos", {})
    for i in range(1, n_ch + 1):
        fields = _split_escaped(ch_info.get(f"Ch{i}", f"Ch{i},,1,µV"))
        labels.append(fields[0] or f"Ch{i}")
        res = fields[2].strip() if len(fields) > 2 else ""
        resolutions.append(float(res) if res else 1.0)
        unit = fields[3].strip() if len(fields) > 3 and fields[3].strip() else "µV"
        units.append(unit)

    if fmt == "BINARY":
        bin_fmt = ini.get("Binary Infos", {}).get("BinaryFormat", "INT_16").upper()
        dtype = np.dtype(_DTYPES[bin_fmt])
        raw_vals = np.fromfile(data_file, dtype=dtype)
        n_samples = raw_vals.size // n_ch
        raw_vals = raw_vals[: n_samples * n_ch]
        if orient.startswith("MULTIPLEX"):
            data = raw_vals.reshape(n_samples, n_ch).T.astype(float)
        else:
            data = raw_vals.reshape(n_ch, n_samples).astype(float)
    elif fmt == "ASCII":
        data = _read_ascii(data_file, ini.get("ASCII Infos", {}), n_ch, orient)
    else:
        raise ValueError(f"Unsupported BrainVision DataFormat {fmt}")

    data *= np.asarray(resolutions)[:, None]
    names = clean_channel_names(labels) if clean_names else labels
    ch_types, out_units = [], []
    for j, (name, unit) in enumerate(zip(names, units)):
        scale = voltage_scale_to_uv(unit)
        t = infer_channel_type(name, unit)
        if scale is not None and t != "stim":
            data[j] *= scale
            out_units.append("uV")
        else:
            out_units.append(unit)
        ch_types.append(t)

    ann, meas_date = Annotations(), None
    if marker_file:
        mpath = _resolve(base, marker_file, ".vmrk", path)
        if os.path.exists(mpath):
            ann, meas_date = _read_markers(mpath, sfreq)

    meta = {"filename": path, "format": "brainvision", "original_ch_names": labels,
            "meas_date": meas_date}
    return RawEEG(data, sfreq, names, ch_types, out_units, annotations=ann, meta=meta)


def _read_ascii(path: str, info: dict, n_ch: int, orient: str) -> np.ndarray:
    skip_lines = int(info.get("SkipLines", 0) or 0)
    skip_cols = int(info.get("SkipColumns", 0) or 0)
    dec = info.get("DecimalSymbol", ".")
    rows = []
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        for k, line in enumerate(f):
            if k < skip_lines or not line.strip():
                continue
            if dec != ".":
                line = line.replace(dec, ".")
            parts = line.replace(",", " ").split() if dec == "." else line.split()
            rows.append([float(v) for v in parts[skip_cols:]])
    arr = np.array(rows, dtype=float)
    if orient.startswith("MULTIPLEX"):
        return arr.T.copy()
    return arr[:n_ch].copy()


def _read_markers(path: str, sfreq: float) -> tuple[Annotations, _dt.datetime | None]:
    ini = _parse_ini(_read_text(path))
    onsets, durs, descs = [], [], []
    meas_date = None
    for key, val in ini.get("Marker Infos", {}).items():
        if not key.lower().startswith("mk"):
            continue
        fields = _split_escaped(val)
        mtype = fields[0].strip()
        desc = fields[1].strip() if len(fields) > 1 else ""
        pos = int(fields[2]) if len(fields) > 2 and fields[2].strip() else 1
        size = int(fields[3]) if len(fields) > 3 and fields[3].strip() else 1
        if mtype.lower() == "new segment" and len(fields) > 5 and fields[5].strip() and meas_date is None:
            try:
                meas_date = _dt.datetime.strptime(fields[5].strip()[:20], "%Y%m%d%H%M%S%f")
            except ValueError:
                pass
        if mtype.lower() == "new segment":
            if pos > 1:  # a later segment start marks a discontinuity in the data
                onsets.append((pos - 1) / sfreq)
                durs.append(0.0)
                descs.append("BAD boundary")
            continue
        onsets.append((pos - 1) / sfreq)
        durs.append(size / sfreq if size > 1 else 0.0)
        # Free-text comments keep just their text; other markers keep "Type/Description".
        if mtype.lower() == "comment" and desc:
            descs.append(desc)
        else:
            descs.append(f"{mtype}/{desc}" if desc else mtype)
    order = np.argsort(onsets, kind="stable") if onsets else []
    ann = Annotations([onsets[i] for i in order], [durs[i] for i in order], [descs[i] for i in order])
    return ann, meas_date


def write_brainvision(raw: RawEEG, path: str | os.PathLike, picks=None) -> str:
    """Write ``.vhdr``, ``.vmrk`` and ``.eeg`` (IEEE_FLOAT_32, multiplexed)."""
    path = os.fspath(path)
    stem = os.path.splitext(path)[0]
    vhdr, vmrk, eeg = stem + ".vhdr", stem + ".vmrk", stem + ".eeg"
    base = os.path.basename(stem)
    idx = raw.pick_indices(picks)
    data = raw.data[idx].astype("<f4")
    data.T.tofile(eeg)

    lines = ["Brain Vision Data Exchange Header File Version 1.0",
             "; Data written by eegproc", "", "[Common Infos]", "Codepage=UTF-8",
             f"DataFile={base}.eeg", f"MarkerFile={base}.vmrk", "DataFormat=BINARY",
             "DataOrientation=MULTIPLEXED", f"NumberOfChannels={len(idx)}",
             f"SamplingInterval={1e6 / raw.sfreq:.10g}", "", "[Binary Infos]",
             "BinaryFormat=IEEE_FLOAT_32", "", "[Channel Infos]"]
    for k, i in enumerate(idx, start=1):
        name = raw.ch_names[i].replace(",", r"\1")
        unit = raw.units[i]
        unit = "µV" if unit == "uV" else (unit or "")
        lines.append(f"Ch{k}={name},,1,{unit}")
    with open(vhdr, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")

    date = raw.meta.get("meas_date")
    stamp = date.strftime("%Y%m%d%H%M%S%f") if isinstance(date, _dt.datetime) else ""
    mk = ["Brain Vision Data Exchange Marker File, Version 1.0", "", "[Common Infos]",
          "Codepage=UTF-8", f"DataFile={base}.eeg", "", "[Marker Infos]",
          f"Mk1=New Segment,,1,1,0{',' + stamp if stamp else ''}"]
    ann = raw.annotations.sorted()
    k = 1
    for o, d, s in zip(ann.onset, ann.duration, ann.description):
        if s.lower().startswith("new segment"):
            continue
        mtype, desc = s.split("/", 1) if "/" in s else ("Comment", s)
        pos = int(round(o * raw.sfreq)) + 1
        size = max(1, int(round(d * raw.sfreq)))
        k += 1
        mk.append(f"Mk{k}={mtype},{desc.replace(',', chr(92) + '1')},{pos},{size},0")
    with open(vmrk, "w", encoding="utf-8") as f:
        f.write("\n".join(mk) + "\n")
    return vhdr
