"""Delimited text files: CSV, TSV, TXT (including OpenBCI exports).

The reader works out the delimiter, header row, time/index columns, the
sampling rate (from a time column or ``Sample Rate = 250 Hz`` style comment
lines) and whether channels are stored in columns or rows.
"""

from __future__ import annotations

import csv
import io as _io
import os
import re

import numpy as np

from ..core.annotations import Annotations
from ..core.channels import clean_channel_names, infer_channel_type, voltage_scale_to_uv
from ..core.raw import RawEEG
from ..utils import logger

_TIME_NAMES = re.compile(r"^(time|times|t|timestamp|time_?stamp|time ?\(s\)|time ?\[s\]|time_s|"
                         r"seconds?|secs?|elapsed.*|latency)$", re.IGNORECASE)
_TIME_MS_NAMES = re.compile(r"^(time ?\(ms\)|time ?\[ms\]|time_?ms|ms|milliseconds?)$", re.IGNORECASE)
_INDEX_NAMES = re.compile(r"^(sample|samples|sample ?index|sample_?(no|num|number)|index|idx|n|"
                          r"unnamed: ?0|#|frame|counter)$", re.IGNORECASE)
_EVENT_NAMES = re.compile(r"^(marker|markers|event|events|trigger|triggers|label|labels|stim|annotation)s?$",
                          re.IGNORECASE)
_SFREQ_COMMENT = re.compile(r"(sampl\w*[\s_-]*(rate|frequency|freq)|sfreq|srate|\bfs\b)\s*[:=]?\s*"
                            r"([0-9]+(?:\.[0-9]+)?)", re.IGNORECASE)


def _to_float(tok: str) -> float:
    try:
        return float(tok)
    except ValueError:
        return np.nan


def read_text(path: str | os.PathLike, sfreq: float | None = None, delimiter: str | None = None,
              ch_names: list[str] | None = None, orientation: str = "auto",
              unit: str = "uV", time_column: str | int | None = "auto",
              clean_names: bool = True) -> RawEEG:
    """Read EEG from a delimited text file.

    Parameters
    ----------
    sfreq : float, optional
        Sampling rate. Needed only when the file has neither a time column nor
        a sampling-rate comment.
    delimiter : str, optional
        Column separator; detected automatically by default.
    ch_names : list of str, optional
        Override the channel names.
    orientation : ``'auto'`` | ``'samples_x_channels'`` | ``'channels_x_samples'``
    unit : str
        Unit of the voltage values in the file (``'uV'``, ``'mV'``, ``'V'``).
    time_column : ``'auto'``, column name/index or ``None``
    """
    path = os.fspath(path)
    with open(path, "r", encoding="utf-8-sig", errors="replace") as f:
        text = f.read()
    lines = text.splitlines()
    comments = [ln for ln in lines if ln.lstrip().startswith(("#", "%", "//"))]
    body = [ln for ln in lines if ln.strip() and not ln.lstrip().startswith(("#", "%", "//"))]
    if not body:
        raise ValueError(f"{path} contains no data")

    if sfreq is None:
        for c in comments:
            m = _SFREQ_COMMENT.search(c)
            if m:
                sfreq = float(m.group(3))
                break

    if delimiter is None:
        delimiter = _sniff_delimiter(body[:50])
    rows = _split_rows(body, delimiter)

    width = max(len(r) for r in rows[:200])
    first = rows[0]
    first_numeric = sum(np.isfinite(_to_float(t)) for t in first)
    has_header = first_numeric < max(1, len(first) // 2)
    header = [h.strip() for h in first] if has_header else None
    data_rows = rows[1:] if has_header else rows

    # Channels stored as rows with the label in the first column?
    label_rows = (orientation == "channels_x_samples" or
                  (orientation == "auto" and not has_header and len(data_rows) > 1 and
                   all(not np.isfinite(_to_float(r[0])) for r in data_rows[: min(20, len(data_rows))])))
    if label_rows:
        labels = [r[0].strip() for r in data_rows]
        values = np.array([[_to_float(t) for t in r[1:]] for r in data_rows], dtype=float)
        return _finish(values, labels, sfreq, unit, path, ch_names, clean_names, Annotations())

    table, str_cols = _to_table(data_rows, width)
    header = header or [f"Ch{j + 1}" for j in range(width)]
    header = (header + [f"Ch{j + 1}" for j in range(len(header), width)])[:width]

    # Drop columns that are mostly text (formatted timestamps etc.), but keep
    # an event/marker column as annotations.
    valid = np.isfinite(table).mean(axis=0)
    keep = [j for j in range(width) if valid[j] > 0.5]
    event_cols = [j for j in range(width) if _EVENT_NAMES.match(header[j] or "")]

    time_idx = None
    if time_column == "auto":
        for j in keep:
            if _TIME_NAMES.match(header[j]) or _TIME_MS_NAMES.match(header[j]):
                time_idx = j
                break
    elif time_column is not None:
        time_idx = header.index(time_column) if isinstance(time_column, str) else int(time_column)
    index_cols = [j for j in keep if _INDEX_NAMES.match(header[j])]

    if time_idx is not None and sfreq is None:
        t = table[:, time_idx]
        t = t[np.isfinite(t)]
        dt = np.median(np.diff(t)) if t.size > 1 else np.nan
        if np.isfinite(dt) and dt > 0:
            if _TIME_MS_NAMES.match(header[time_idx]):
                dt /= 1000.0
            elif "timestamp" in header[time_idx].lower() and dt > 1:
                dt /= 1000.0  # millisecond epoch timestamps
            sfreq = 1.0 / dt
            logger.info("Sampling rate %.4g Hz estimated from column %r", sfreq, header[time_idx])
    if sfreq is None:
        raise ValueError(f"Cannot determine the sampling rate of {path}; pass sfreq=...")

    ch_cols = [j for j in keep if j != time_idx and j not in index_cols and j not in event_cols]
    if not ch_cols:
        raise ValueError(f"No numeric channel columns found in {path}")
    values = table[:, ch_cols].T
    labels = [header[j] for j in ch_cols]

    ann = Annotations()
    for j in event_cols:
        col = table[:, j]
        texts = str_cols.get(j)
        for i in range(len(data_rows)):
            label = texts[i] if texts and texts[i] else None
            if label is None and np.isfinite(col[i]) and col[i] != 0:
                label = str(int(col[i])) if float(col[i]).is_integer() else str(col[i])
            if label:
                for part in label.split("|"):
                    ann.append(i / sfreq, 0.0, part)

    return _finish(values, labels, sfreq, unit, path, ch_names, clean_names, ann)


def _finish(values, labels, sfreq, unit, path, ch_names, clean_names, ann) -> RawEEG:
    if ch_names is not None:
        if len(ch_names) != values.shape[0]:
            raise ValueError(f"{len(ch_names)} names given for {values.shape[0]} channels")
        labels = list(ch_names)
    names = clean_channel_names(labels) if clean_names else list(labels)
    types = [infer_channel_type(n) for n in names]
    scale = voltage_scale_to_uv(unit) or 1.0
    units = []
    for k, t in enumerate(types):
        if t in ("eeg", "eog", "ecg", "emg"):
            values[k] = values[k] * scale
            units.append("uV")
        else:
            units.append("" if t == "stim" else "a.u.")
    if np.isnan(values).any():
        n_nan = int(np.isnan(values).sum())
        logger.warning("%s: %d missing values replaced by linear interpolation", os.path.basename(path), n_nan)
        values = _fill_nan(values)
    meta = {"filename": path, "format": "text", "original_ch_names": list(labels)}
    return RawEEG(values, sfreq, names, types, units, annotations=ann, meta=meta)


def _to_table(rows: list[list[str]], width: int) -> tuple[np.ndarray, dict[int, list[str]]]:
    """Numeric table (NaN for non-numbers) plus the text of non-numeric cells per column."""
    try:
        import pandas as pd
    except ImportError:
        pd = None
    str_cols: dict[int, list[str]] = {}
    if pd is not None:
        frame = pd.DataFrame([r[:width] + [""] * (width - len(r)) for r in rows])
        numeric = frame.apply(pd.to_numeric, errors="coerce")
        table = numeric.to_numpy(dtype=float)
        for j in range(width):
            miss = np.isnan(table[:, j])
            if miss.any():
                col = frame.iloc[:, j].astype(str).str.strip()
                if (col[miss] != "").any():
                    str_cols[j] = [c if m else "" for c, m in zip(col.tolist(), miss)]
        return table, str_cols
    table = np.full((len(rows), width), np.nan)
    for i, r in enumerate(rows):
        for j, tok in enumerate(r[:width]):
            v = _to_float(tok)
            table[i, j] = v
            if not np.isfinite(v) and tok.strip():
                str_cols.setdefault(j, [""] * len(rows))[i] = tok.strip()
    return table, str_cols


def _fill_nan(values: np.ndarray) -> np.ndarray:
    out = values.copy()
    idx = np.arange(values.shape[1])
    for k in range(values.shape[0]):
        bad = ~np.isfinite(out[k])
        if bad.all():
            out[k] = 0.0
        elif bad.any():
            out[k, bad] = np.interp(idx[bad], idx[~bad], out[k, ~bad])
    return out


def _sniff_delimiter(lines: list[str]) -> str | None:
    sample = "\n".join(lines)
    try:
        return csv.Sniffer().sniff(sample, delimiters=",;\t| ").delimiter
    except csv.Error:
        for d in ("\t", ",", ";", "|"):
            if all(d in ln for ln in lines[:10]):
                return d
        return None  # whitespace


def _split_rows(lines: list[str], delimiter: str | None) -> list[list[str]]:
    if delimiter is None or delimiter == " ":
        return [ln.split() for ln in lines]
    reader = csv.reader(_io.StringIO("\n".join(lines)), delimiter=delimiter)
    return [[c.strip() for c in row] for row in reader if row]


def write_text(raw: RawEEG, path: str | os.PathLike, picks=None, delimiter: str | None = None,
               time_column: bool = True) -> str:
    """Write samples x channels text with a header row and a ``time`` column."""
    path = os.fspath(path)
    if delimiter is None:
        delimiter = "\t" if path.lower().endswith((".tsv", ".tab")) else ","
    idx = raw.pick_indices(picks)
    cols = [raw.ch_names[i] for i in idx]
    data = raw.data[idx].T
    markers = [""] * raw.n_times
    for o, d in zip(raw.annotations.onset, raw.annotations.description):
        k = int(round(o * raw.sfreq))
        if 0 <= k < raw.n_times:
            text = d.replace(delimiter, " ").replace('"', "'")
            markers[k] = f"{markers[k]}|{text}" if markers[k] else text
    with_markers = len(raw.annotations) > 0
    header = (["time"] if time_column else []) + cols + (["marker"] if with_markers else [])
    with open(path, "w", encoding="utf-8", newline="") as f:
        f.write(f"# sfreq = {raw.sfreq:.10g}\n")
        f.write(delimiter.join(header) + "\n")
        buf = _io.StringIO()
        arr = np.column_stack([raw.times, data]) if time_column else data
        np.savetxt(buf, arr, delimiter=delimiter, fmt="%.7g")
        lines = buf.getvalue().splitlines()
        if with_markers:
            lines = [f"{ln}{delimiter}{m}" for ln, m in zip(lines, markers)]
        f.write("\n".join(lines) + "\n")
    return path
