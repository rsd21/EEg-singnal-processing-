"""Channel names, channel types and physical units."""

from __future__ import annotations

import re
from typing import Iterable

from .montage import standard_name

CHANNEL_TYPES = ("eeg", "eog", "ecg", "emg", "stim", "resp", "misc")
VOLTAGE_TYPES = ("eeg", "eog", "ecg", "emg")
DATA_TYPES = ("eeg", "eog", "ecg", "emg", "resp", "misc")

_VOLT_TO_UV = {"v": 1e6, "volt": 1e6, "volts": 1e6, "mv": 1e3, "uv": 1.0, "µv": 1.0,
               "μv": 1.0, "microvolt": 1.0, "microvolts": 1.0, "nv": 1e-3}

_REF_SUFFIX = re.compile(r"[\s_\-/]+(ref|le|re|avg|ar|car|a1|a2|m1|m2|a1a2|m1m2|a1\+a2|m1\+m2|lm)$",
                         re.IGNORECASE)
_EEG_PREFIX = re.compile(r"^(eeg)[\s_:\-]+", re.IGNORECASE)

_TYPE_PATTERNS = [
    ("stim", re.compile(r"^(status|trigger|trig|stim|sti\d*|sti \d+|markers?|events?|"
                        r"edf annotations|bdf annotations|digital|di\d*)$", re.IGNORECASE)),
    ("eog", re.compile(r"(eog|heog|veog|^loc$|^roc$|^e1$|^e2$|eye|^lo\d?$|^ro\d?$|^so\d?$|^io\d?$)",
                       re.IGNORECASE)),
    ("ecg", re.compile(r"(ecg|ekg|heart|^ecg\d*)", re.IGNORECASE)),
    ("emg", re.compile(r"(emg|chin|^leg|tibial|masseter)", re.IGNORECASE)),
    ("resp", re.compile(r"(resp|thor|abdo|abd$|airflow|^flow|breath|nasal|thermistor|cannula)",
                        re.IGNORECASE)),
    ("misc", re.compile(r"(spo2|sao2|pulse|pleth|^hr$|temp|position|light|accel|^acc|gyro|^mic|"
                        r"snore|battery|^aux|gsr|eda|ppg|^[xyz]$|timestamp|counter|sample|"
                        r"photic|^sat|^pos$|^bp$|^cpap|^pap|ibi|^pr$)", re.IGNORECASE)),
]


def clean_channel_name(name: str, standardize: bool = True) -> str:
    """Normalise an EEG channel label.

    * removes an ``EEG`` prefix (``"EEG Fp1-REF"`` -> ``"Fp1"``),
    * removes a reference suffix (``-REF``, ``-LE``, ``-A1``, ``-M2`` ...),
    * removes trailing dots used by some files (``"Fc5."`` -> ``"FC5"``),
    * with ``standardize=True`` fixes capitalisation of known 10-10 names
      (``"FP1"`` -> ``"Fp1"``, ``"CZ"`` -> ``"Cz"``) and maps T3/T4/T5/T6 to
      T7/T8/P7/P8.

    Labels that do not become a known electrode name are only stripped of
    surrounding whitespace, so bipolar labels such as ``"Fp1-F7"`` survive.
    """
    raw = str(name).strip().strip("\x00").strip()
    candidate = _EEG_PREFIX.sub("", raw).strip(". ")
    candidate = _REF_SUFFIX.sub("", candidate).strip(". ") if standard_name(candidate) is None else candidate
    std = standard_name(candidate)
    if std is not None:
        if not standardize:
            return candidate
        return std
    return raw


def make_unique(names: Iterable[str]) -> list[str]:
    """Append ``-1``, ``-2`` ... to duplicated names."""
    seen: dict[str, int] = {}
    out = []
    for n in names:
        if n in seen:
            seen[n] += 1
            new = f"{n}-{seen[n]}"
            while new in seen:
                seen[n] += 1
                new = f"{n}-{seen[n]}"
            seen[new] = 0
            out.append(new)
        else:
            seen[n] = 0
            out.append(n)
    return out


def clean_channel_names(names: Iterable[str], standardize: bool = True) -> list[str]:
    return make_unique([clean_channel_name(n, standardize) for n in names])


def infer_channel_type(name: str, unit: str | None = None) -> str:
    """Guess a channel type from its label (and unit, if known)."""
    label = str(name).strip()
    if standard_name(clean_channel_name(label)) is not None:
        return "eeg"
    for ch_type, pattern in _TYPE_PATTERNS:
        if pattern.search(label):
            return ch_type
    if unit is not None and unit.strip() and voltage_scale_to_uv(unit) is None:
        return "misc"
    return "eeg"


def voltage_scale_to_uv(unit: str | None) -> float | None:
    """Factor that converts values in ``unit`` to microvolts (None if not a voltage)."""
    if unit is None:
        return None
    key = str(unit).strip().lower().replace(" ", "")
    return _VOLT_TO_UV.get(key)


def default_unit(ch_type: str) -> str:
    if ch_type in VOLTAGE_TYPES:
        return "uV"
    if ch_type == "stim":
        return ""
    return "a.u."
