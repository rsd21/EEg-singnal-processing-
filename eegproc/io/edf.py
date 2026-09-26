"""EDF, EDF+, BDF and BDF+ reader/writer implemented from the specifications.

* EDF / EDF+: https://www.edfplus.info/specs/  (16-bit little-endian samples)
* BDF / BDF+: BioSemi 24-bit variant (header byte 0 = 0xFF, "BIOSEMI")

The reader supports mixed sampling rates, EDF+ annotations (TALs), BDF status
channels and partial reads (a subset of channels and/or a time window) so
large files do not have to be loaded completely.
"""

from __future__ import annotations

import datetime as _dt
import os
import re
from fractions import Fraction
from typing import Iterable

import numpy as np

from ..core.annotations import Annotations
from ..core.channels import clean_channel_names, infer_channel_type, voltage_scale_to_uv
from ..core.raw import RawEEG
from ..utils import logger

_ANNOT_LABELS = ("edf annotations", "bdf annotations")
_MONTHS = ["JAN", "FEB", "MAR", "APR", "MAY", "JUN", "JUL", "AUG", "SEP", "OCT", "NOV", "DEC"]
_TAL = re.compile(r"([+-]\d+(?:\.\d*)?)(?:\x15(\d+(?:\.\d*)?))?(\x14[^\x00]*)\x14?\x00")


# ====================================================================== header
def _num(s: str, default=None, kind=float):
    s = s.strip().replace(",", ".")
    if not s:
        return default
    try:
        return kind(float(s)) if kind is int else kind(s)
    except ValueError:
        return default


def _parse_start(date_s: str, time_s: str, recording: str) -> _dt.datetime | None:
    try:
        d, m, y = (int(p) for p in re.split(r"[.\-/:]", date_s.strip())[:3])
        hh, mm, ss = (int(p) for p in re.split(r"[.:\-]", time_s.strip())[:3])
        year = 1900 + y if y >= 85 else 2000 + y
        # EDF+ stores a 4-digit year in the recording field ("Startdate 02-MAR-2002").
        match = re.search(r"Startdate\s+(\d{2})-([A-Za-z]{3})-(\d{4})", recording)
        if match:
            year = int(match.group(3))
        stamp = _dt.datetime(year, m, d, hh, mm, ss)
        # 01.01.85 00.00.00 is the conventional "unknown/anonymised" start date
        return None if stamp == _dt.datetime(1985, 1, 1) else stamp
    except (ValueError, TypeError):
        return None


def read_edf_header(path: str | os.PathLike) -> dict:
    """Parse the header of an EDF/BDF file without reading the signals."""
    path = os.fspath(path)
    with open(path, "rb") as f:
        head = f.read(256)
        if len(head) < 256:
            raise ValueError(f"{path} is too short to be an EDF/BDF file")
        is_bdf = head[0] == 0xFF
        text = head.decode("latin-1")

        def field(a: int, b: int) -> str:
            return text[a:b].strip()

        n_signals = _num(field(252, 256), kind=int)
        if n_signals is None or n_signals <= 0:
            raise ValueError(f"{path}: invalid number of signals in header")
        sig_raw = f.read(256 * n_signals).decode("latin-1")
    if len(sig_raw) < 256 * n_signals:
        raise ValueError(f"{path}: truncated signal header")

    widths = [("label", 16), ("transducer", 80), ("unit", 8), ("physical_min", 8),
              ("physical_max", 8), ("digital_min", 8), ("digital_max", 8), ("prefilter", 80),
              ("n_samples", 8), ("reserved", 32)]
    columns: dict[str, list[str]] = {}
    pos = 0
    for name, w in widths:
        columns[name] = [sig_raw[pos + i * w: pos + (i + 1) * w].strip() for i in range(n_signals)]
        pos += w * n_signals

    signals = []
    for i in range(n_signals):
        sig = {
            "label": columns["label"][i],
            "transducer": columns["transducer"][i],
            "unit": columns["unit"][i],
            "physical_min": _num(columns["physical_min"][i], -1.0),
            "physical_max": _num(columns["physical_max"][i], 1.0),
            "digital_min": _num(columns["digital_min"][i], -32768, int),
            "digital_max": _num(columns["digital_max"][i], 32767, int),
            "prefilter": columns["prefilter"][i],
            "n_samples": _num(columns["n_samples"][i], 0, int),
        }
        signals.append(sig)

    header_bytes = _num(field(184, 192), 256 * (n_signals + 1), int)
    record_duration = _num(field(244, 252), 1.0)
    n_records = _num(field(236, 244), -1, int)
    bps = 3 if is_bdf else 2
    record_bytes = sum(s["n_samples"] for s in signals) * bps
    file_size = os.path.getsize(path)
    available = (file_size - header_bytes) // record_bytes if record_bytes else 0
    if n_records is None or n_records < 0 or n_records > available:
        if n_records not in (None, -1) and n_records > available:
            logger.warning("%s: header says %d records but only %d are present; reading %d",
                           os.path.basename(path), n_records, available, available)
        n_records = int(available)
    if record_duration == 0:  # EDF+ allows 0 for annotation-only files
        record_duration = 1.0

    reserved = field(192, 236)
    patient = field(8, 88)
    recording = field(88, 168)
    for s in signals:
        s["sfreq"] = s["n_samples"] / record_duration
    return {
        "path": path,
        "format": "bdf" if is_bdf else "edf",
        "subtype": reserved[:5] if reserved[:4] in ("EDF+", "BDF+") else ("24BIT" if is_bdf else "EDF"),
        "discontinuous": reserved[:5] in ("EDF+D", "BDF+D"),
        "patient": patient,
        "recording": recording,
        "meas_date": _parse_start(field(168, 176), field(176, 184), recording),
        "header_bytes": header_bytes,
        "n_records": n_records,
        "record_duration": record_duration,
        "bytes_per_sample": bps,
        "record_bytes": record_bytes,
        "n_signals": n_signals,
        "signals": signals,
        "duration": n_records * record_duration,
    }


def _patient_info(patient: str) -> dict:
    """EDF+ patient field "code sex birthdate name"; 'X' means unknown."""
    parts = patient.split(" ")
    if len(parts) >= 4:
        info = {"id": parts[0], "sex": parts[1], "birthdate": parts[2], "name": parts[3].replace("_", " ")}
        return {k: v for k, v in info.items() if v and v != "X"}
    return {"id": patient} if patient and patient != "X" else {}


# ====================================================================== read
def _decode_samples(block: np.ndarray, byte_off: int, n: int, bps: int) -> np.ndarray:
    """Digital values of one signal from a (n_records, record_bytes) uint8 block."""
    cols = block[:, byte_off: byte_off + n * bps]
    if bps == 2:
        return np.ascontiguousarray(cols).view("<i2").reshape(-1).astype(np.int32)
    c = cols.reshape(block.shape[0], n, 3).astype(np.int32)
    val = c[..., 0] | (c[..., 1] << 8) | (c[..., 2] << 16)
    val = np.where(val >= (1 << 23), val - (1 << 24), val)
    return val.reshape(-1)


def _parse_tals(raw_bytes: bytes) -> list[tuple[float, float, list[str]]]:
    text = raw_bytes.decode("utf-8", errors="replace")
    out = []
    for m in _TAL.finditer(text):
        onset = float(m.group(1))
        duration = float(m.group(2)) if m.group(2) else 0.0
        texts = [t for t in m.group(3).split("\x14") if t]
        out.append((onset, duration, texts))
    return out


def read_edf(path: str | os.PathLike, include: Iterable[str] | None = None,
             exclude: Iterable[str] | None = None, tmin: float | None = None,
             tmax: float | None = None, clean_names: bool = True,
             infer_types: bool = True, chunk_records: int | None = None) -> RawEEG:
    """Read an EDF/EDF+/BDF/BDF+ file.

    Parameters
    ----------
    include, exclude : list of str, optional
        Channel labels to keep / drop (original or cleaned labels).
    tmin, tmax : float, optional
        Only read this time window (seconds from the start).
    clean_names : bool
        Normalise labels (``'EEG Fp1-REF'`` -> ``'Fp1'``). The original labels
        are kept in ``raw.meta['original_ch_names']``.
    """
    hdr = read_edf_header(path)
    signals = hdr["signals"]
    bps = hdr["bytes_per_sample"]
    ann_idx = [i for i, s in enumerate(signals) if s["label"].lower() in _ANNOT_LABELS]
    data_idx = [i for i in range(len(signals)) if i not in ann_idx and signals[i]["n_samples"] > 0]

    labels = [signals[i]["label"] for i in data_idx]
    names = clean_channel_names(labels) if clean_names else list(labels)
    selected = list(range(len(data_idx)))
    if include is not None:
        inc = {str(c).lower() for c in include}
        selected = [k for k in selected if labels[k].lower() in inc or names[k].lower() in inc]
    if exclude is not None:
        exc = {str(c).lower() for c in exclude}
        selected = [k for k in selected if labels[k].lower() not in exc and names[k].lower() not in exc]
    if not selected:
        raise ValueError(f"No channels selected from {path}")

    rates = np.array([signals[data_idx[k]]["sfreq"] for k in selected])
    sfreq = float(rates.max())
    if not np.allclose(rates, sfreq):
        logger.warning("%s has mixed sampling rates %s Hz; resampling all channels to %g Hz",
                       os.path.basename(path), sorted(set(np.round(rates, 4))), sfreq)

    rec_dur = hdr["record_duration"]
    n_rec = hdr["n_records"]
    t0 = 0.0 if tmin is None else max(0.0, float(tmin))
    t1 = n_rec * rec_dur if tmax is None else min(float(tmax), n_rec * rec_dur)
    if t1 <= t0:
        raise ValueError(f"Empty time window [{t0}, {t1}] for a {n_rec * rec_dur:.2f} s file")
    rec_a = int(np.floor(t0 / rec_dur + 1e-9))
    rec_b = min(n_rec, int(np.ceil(t1 / rec_dur - 1e-9)))

    offsets = np.concatenate([[0], np.cumsum([s["n_samples"] * bps for s in signals])])
    chunk = chunk_records or max(1, int(64e6 // max(hdr["record_bytes"], 1)))
    pieces: list[list[np.ndarray]] = [[] for _ in selected]
    ann_bytes: list[bytes] = []
    # Annotations may be stored in any record, so a partial read still scans
    # the annotation signal of every record (signals are only decoded inside
    # the requested window).
    scan_a, scan_b = (0, n_rec) if ann_idx else (rec_a, rec_b)
    with open(hdr["path"], "rb") as f:
        for start in range(scan_a, scan_b, chunk):
            stop = min(scan_b, start + chunk)
            f.seek(hdr["header_bytes"] + start * hdr["record_bytes"])
            buf = f.read((stop - start) * hdr["record_bytes"])
            block = np.frombuffer(buf, dtype=np.uint8).reshape(stop - start, hdr["record_bytes"])
            for i in ann_idx:
                seg = block[:, offsets[i]: offsets[i + 1]]
                ann_bytes.extend(bytes(row) for row in seg)
            lo, hi = max(start, rec_a), min(stop, rec_b)
            if lo >= hi:
                continue
            sub = block[lo - start: hi - start]
            for j, k in enumerate(selected):
                i = data_idx[k]
                pieces[j].append(_decode_samples(sub, offsets[i], signals[i]["n_samples"], bps))

    n_out = int(round((rec_b - rec_a) * rec_dur * sfreq))
    data = np.empty((len(selected), n_out))
    ch_names, ch_types, units = [], [], []
    for j, k in enumerate(selected):
        s = signals[data_idx[k]]
        dig = np.concatenate(pieces[j]) if pieces[j] else np.zeros(0)
        label = names[k]
        unit = s["unit"]
        ch_type = infer_channel_type(label, unit) if infer_types else "eeg"
        is_status = hdr["format"] == "bdf" and s["label"].lower() == "status"
        if is_status:
            ch_type = "stim"
            values = dig.astype(float)
        else:
            dmin, dmax = s["digital_min"], s["digital_max"]
            pmin, pmax = s["physical_min"], s["physical_max"]
            if dmax == dmin:
                gain, offset = 1.0, 0.0
            else:
                gain = (pmax - pmin) / (dmax - dmin)
                offset = pmin - gain * dmin
            values = dig * gain + offset
            scale = voltage_scale_to_uv(unit)
            if scale is not None and ch_type != "stim":
                values = values * scale
                unit = "uV"
        if len(values) != n_out:
            values = _match_length(values, n_out, is_stim=ch_type == "stim")
        data[j] = values
        ch_names.append(label)
        ch_types.append(ch_type)
        units.append(unit or ("" if ch_type == "stim" else "a.u."))

    # ---- annotations (onsets are relative to the start of the first record)
    onsets, durs, descs = [], [], []
    first_record_time = 0.0
    for rec_i, rb in enumerate(ann_bytes):
        for n_tal, (onset, dur, texts) in enumerate(_parse_tals(rb)):
            if rec_i == 0 and n_tal == 0:
                first_record_time = onset  # time-keeping TAL of the first record
            for t in texts:
                onsets.append(onset)
                durs.append(dur)
                descs.append(t)
    ann = Annotations(np.asarray(onsets) - first_record_time, durs, descs)
    a = int(round((t0 - rec_a * rec_dur) * sfreq))
    b = a + int(round((t1 - t0) * sfreq))
    if a > 0 or b < n_out:
        data = data[:, a:b]
    if len(ann) and (t0 > 0 or tmax is not None):
        ann = ann.crop(t0, t0 + data.shape[1] / sfreq, shift=True)
    if hdr["discontinuous"]:
        logger.warning("%s is EDF+D (discontinuous); records are concatenated as continuous data",
                       os.path.basename(path))

    meta = {
        "filename": hdr["path"], "format": hdr["format"], "subtype": hdr["subtype"],
        "meas_date": hdr["meas_date"], "subject": _patient_info(hdr["patient"]),
        "recording_info": hdr["recording"],
        "original_ch_names": [labels[k] for k in selected],
        "prefilter": {names[k]: signals[data_idx[k]]["prefilter"] for k in selected},
        "transducer": {names[k]: signals[data_idx[k]]["transducer"] for k in selected},
    }
    if t0 > 0:
        meta["first_time"] = t0
    return RawEEG(data, sfreq, ch_names, ch_types, units, annotations=ann, meta=meta)


def _match_length(x: np.ndarray, n: int, is_stim: bool = False) -> np.ndarray:
    """Resample a lower-rate signal to ``n`` samples."""
    if len(x) == 0:
        return np.zeros(n)
    if is_stim:
        idx = np.minimum((np.arange(n) * len(x) / n).astype(int), len(x) - 1)
        return x[idx]
    frac = Fraction(n, len(x)).limit_denominator(1000)
    if frac.numerator * len(x) == frac.denominator * n and frac.numerator <= 1000:
        from scipy.signal import resample_poly

        return resample_poly(x, frac.numerator, frac.denominator)[:n]
    t_new = np.linspace(0, len(x) - 1, n)
    return np.interp(t_new, np.arange(len(x)), x)


# ===================================================================== write
def _fmt(value, width: int = 8) -> str:
    """Format a number into at most ``width`` characters (no exponent)."""
    v = float(value)
    if v.is_integer() and len(str(int(v))) <= width:
        return str(int(v))
    for dec in range(width, -1, -1):
        s = f"{v:.{dec}f}"
        if "." in s:
            s = s.rstrip("0").rstrip(".")
        if len(s) <= width:
            return s
    raise ValueError(f"Cannot represent {value} in {width} characters")


def _field(s: str, width: int) -> str:
    s = str(s).encode("ascii", "replace").decode("ascii")
    return s[:width].ljust(width)


def _choose_record(sfreq: float, n_samples: int) -> tuple[float, int]:
    """Pick (record_duration, samples_per_record) with an exactly representable duration."""
    frac = Fraction(sfreq).limit_denominator(1000)
    if abs(float(frac) - sfreq) > 1e-9 * sfreq:
        raise ValueError(f"Sampling rate {sfreq} cannot be stored exactly in EDF")
    base_dur, base_n = frac.denominator, frac.numerator  # base_n samples per base_dur seconds
    candidates = []
    # durations that are multiples/fractions of the base record
    for mult in (1, 2, 4, 5, 10):
        candidates.append((Fraction(base_dur * mult), base_n * mult))
    for div in (2, 4, 5, 8, 10, 16, 20, 25, 32, 40, 50, 64, 100, 128, 256):
        if base_n % div == 0:
            candidates.append((Fraction(base_dur, div), base_n // div))
    valid = []
    for dur, n in candidates:
        s = _fmt(float(dur))
        if Fraction(s) != dur:
            continue
        valid.append((dur, n))
    fitting = [(d, n) for d, n in valid if n_samples % n == 0]
    pool = fitting or valid
    dur, n = min(pool, key=lambda dn: (abs(float(dn[0]) - 1.0), -dn[1]))
    return float(dur), int(n)


def _tal(onset: float, duration: float, texts: list[str]) -> bytes:
    s = ("+" if onset >= 0 else "-") + _fmt(abs(onset), 20)
    if duration > 0:
        s += "\x15" + _fmt(duration, 20)
    s += "\x14" + "".join(t.replace("\x14", " ").replace("\x00", " ") + "\x14" for t in texts)
    return s.encode("utf-8") + b"\x00"


def write_edf(raw: RawEEG, path: str | os.PathLike, bdf: bool | None = None,
              picks=None, write_annotations: bool = True) -> str:
    """Write a recording to EDF+ (16 bit) or BDF+ (24 bit).

    Physical ranges are chosen per channel from the data, so each channel uses
    the full integer range. Voltage channels are written in microvolts.
    """
    path = os.fspath(path)
    if bdf is None:
        bdf = path.lower().endswith(".bdf")
    bps = 3 if bdf else 2
    dmin, dmax = (-(1 << 23), (1 << 23) - 1) if bdf else (-32768, 32767)
    idx = raw.pick_indices(picks)
    data = raw.data[idx]
    n_ch, n_samples = data.shape
    rec_dur, spr = _choose_record(raw.sfreq, n_samples)
    n_records = int(np.ceil(n_samples / spr))
    pad = n_records * spr - n_samples
    if pad:
        logger.warning("Padding %d samples to fill the last EDF record", pad)
        data = np.concatenate([data, np.repeat(data[:, -1:], pad, axis=1)], axis=1)

    labels, units, pmins, pmaxs, digital = [], [], [], [], []
    for j, i in enumerate(idx):
        x = np.nan_to_num(data[j])
        is_stim = raw.ch_types[i] == "stim"
        if is_stim:
            lo, hi = float(dmin), float(dmax)
            pmin_s, pmax_s = _fmt(lo), _fmt(hi)
        else:
            lo, hi = float(np.min(x)), float(np.max(x))
            if hi - lo < 1e-6:
                lo, hi = lo - 1.0, hi + 1.0
            margin = 1e-4 * (hi - lo)
            pmin_s, pmax_s = _fmt(_round_out(lo - margin, down=True)), _fmt(_round_out(hi + margin, down=False))
        pmin, pmax = float(pmin_s), float(pmax_s)
        gain = (pmax - pmin) / (dmax - dmin)
        dig = np.rint((np.clip(x, pmin, pmax) - pmin) / gain + dmin)
        digital.append(np.clip(dig, dmin, dmax).astype(np.int32))
        labels.append(raw.ch_names[i])
        units.append(raw.units[i] if raw.units[i] not in ("a.u.",) else "")
        pmins.append(pmin_s)
        pmaxs.append(pmax_s)

    ann_tals: list[bytes] = []
    if write_annotations:
        ann = raw.annotations.sorted()
        ann_tals = [_tal(o, d, [s]) for o, d, s in zip(ann.onset, ann.duration, ann.description)]
    # Every record starts with a time-keeping TAL: "+<start>\x14\x14\x00".
    keep_times = [_tal(r * rec_dur, 0.0, [""]) for r in range(n_records)]
    # distribute annotation TALs over records, choosing a capacity that fits them all
    max_keep = max(len(k) for k in keep_times)
    cap = max_keep + (max((len(t) for t in ann_tals), default=0))
    total = sum(len(t) for t in ann_tals)
    cap = max(cap, max_keep + int(np.ceil(total / n_records)))
    while True:
        cap += cap % 2
        assign: list[list[bytes]] = [[] for _ in range(n_records)]
        used = [len(k) for k in keep_times]
        r = 0
        ok = True
        for t in ann_tals:
            while r < n_records and used[r] + len(t) > cap:
                r += 1
            if r >= n_records:
                ok = False
                break
            assign[r].append(t)
            used[r] += len(t)
        if ok:
            break
        cap = int(cap * 1.25) + 2
    ann_samples = cap // 2 if not bdf else int(np.ceil(cap / 3))
    ann_nbytes = ann_samples * bps

    n_sig = n_ch + 1
    meas = raw.meta.get("meas_date")
    if not isinstance(meas, _dt.datetime):
        meas = _dt.datetime(1985, 1, 1)
    subj = raw.meta.get("subject") or {}
    if isinstance(subj, dict):
        patient = " ".join(str(subj.get(k) or "X").replace(" ", "_")
                           for k in ("id", "sex", "birthdate", "name"))
    else:
        patient = f"{str(subj).replace(' ', '_')} X X X"
    recording = f"Startdate {meas.day:02d}-{_MONTHS[meas.month - 1]}-{meas.year} X X eegproc"

    hdr = b"\xffBIOSEMI" if bdf else _field("0", 8).encode()
    head = ""
    head += _field(patient, 80) + _field(recording, 80)
    head += meas.strftime("%d.%m.") + f"{meas.year % 100:02d}"
    head += meas.strftime("%H.%M.%S")
    head += _field(str(256 * (n_sig + 1)), 8)
    head += _field("BDF+C" if bdf else "EDF+C", 44)
    head += _field(str(n_records), 8) + _field(_fmt(rec_dur), 8) + _field(str(n_sig), 4)
    ann_label = "BDF Annotations" if bdf else "EDF Annotations"
    all_labels = labels + [ann_label]
    sig = "".join(_field(lb, 16) for lb in all_labels)
    sig += "".join(_field("", 80) for _ in all_labels)
    sig += "".join(_field(u, 8) for u in units) + _field("", 8)
    sig += "".join(_field(p, 8) for p in pmins) + _field("-1", 8)
    sig += "".join(_field(p, 8) for p in pmaxs) + _field("1", 8)
    sig += "".join(_field(str(dmin), 8) for _ in all_labels)
    sig += "".join(_field(str(dmax), 8) for _ in all_labels)
    sig += "".join(_field(_prefilter(raw), 80) for _ in labels) + _field("", 80)
    sig += "".join(_field(str(spr), 8) for _ in labels) + _field(str(ann_samples), 8)
    sig += "".join(_field("", 32) for _ in all_labels)

    rec_samples = n_ch * spr
    with open(path, "wb") as f:
        f.write(hdr + head.encode("ascii") + sig.encode("ascii"))
        dig = np.stack(digital)  # (n_ch, n_records * spr)
        dig = dig.reshape(n_ch, n_records, spr).transpose(1, 0, 2).reshape(n_records, rec_samples)
        if bdf:
            u = (dig & 0xFFFFFF).astype(np.uint32)
            sig_bytes = np.stack([u & 0xFF, (u >> 8) & 0xFF, (u >> 16) & 0xFF], axis=-1).astype(np.uint8)
            sig_bytes = sig_bytes.reshape(n_records, rec_samples * 3)
        else:
            sig_bytes = dig.astype("<i2").view(np.uint8).reshape(n_records, rec_samples * 2)
        ann_block = np.zeros((n_records, ann_nbytes), dtype=np.uint8)
        for r in range(n_records):
            payload = keep_times[r] + b"".join(assign[r])
            ann_block[r, : len(payload)] = np.frombuffer(payload, dtype=np.uint8)
        f.write(np.hstack([sig_bytes, ann_block]).tobytes())
    return path


def _round_out(v: float, down: bool) -> float:
    """Round a physical limit outward to a value that formats into 8 characters."""
    s = _fmt(v)
    out = float(s)
    if down and out > v or (not down and out < v):
        step = 10.0 ** (-(len(s.split(".")[1]) if "." in s else 0))
        out = out - step if down else out + step
    return out


def _prefilter(raw: RawEEG) -> str:
    hp, lp = raw.meta.get("highpass"), raw.meta.get("lowpass")
    parts = []
    if hp:
        parts.append(f"HP:{hp:g}Hz")
    if lp:
        parts.append(f"LP:{lp:g}Hz")
    return " ".join(parts)
