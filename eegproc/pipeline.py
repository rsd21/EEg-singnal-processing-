"""Reproducible processing pipelines from a JSON/YAML description.

Example (YAML)::

    steps:
      - set_montage: {montage: standard_1010}
      - pick: {picks: [eeg, eog]}
      - filter: {l_freq: 1.0, h_freq: 40.0}
      - notch: {freqs: auto}
      - detect_bad_channels: {}
      - interpolate_bad_channels: {}
      - rereference: {ref: average}
      - ica: {n_components: 0.99, eog: true}
      - resample: {sfreq: 128}
    output: cleaned.edf

Each step maps a name to keyword arguments. ``Pipeline.run(raw)`` returns
the processed recording and keeps a log of every step with its duration.
"""

from __future__ import annotations

import json
import os
import time
from typing import Any, Callable

from .utils import logger


def _step_filter(raw, l_freq=None, h_freq=None, **kw):
    return raw.filter(l_freq, h_freq, **kw)


def _step_notch(raw, freqs=50.0, **kw):
    return raw.notch_filter(freqs, **kw)


def _step_resample(raw, sfreq):
    return raw.resample(float(sfreq))


def _step_rereference(raw, ref="average", **kw):
    return raw.set_reference(ref, **kw)


def _step_pick(raw, picks, exclude_bads=False):
    return raw.pick(picks, exclude_bads=exclude_bads)


def _step_drop(raw, channels):
    return raw.drop_channels(channels)


def _step_crop(raw, tmin=0.0, tmax=None):
    return raw.crop(tmin, tmax)


def _step_montage(raw, montage="standard_1010"):
    out = raw.copy()
    out.set_montage(montage)
    return out


def _step_rename(raw, mapping):
    out = raw.copy()
    out.rename_channels(mapping)
    return out


def _step_types(raw, mapping):
    out = raw.copy()
    out.set_channel_types(mapping)
    return out


def _step_standardize(raw):
    out = raw.copy()
    out.standardize_channel_names()
    return out


def _step_bads(raw, method: str = "quality", **kw):
    """Mark bad channels using the quality score (default) or the PREP criteria."""
    out = raw.copy()
    if method == "prep":
        from .preprocessing.bad_channels import find_bad_channels

        res = find_bad_channels(raw, **kw)
        bads, reasons = res.bads, res.reasons()
    else:
        from .quality.channel_quality import assess_channel_quality

        rep = assess_channel_quality(raw, **kw)
        bads = rep.bad_channels
        reasons = {r["channel"]: r["reasons"] + r["prep_flags"] for r in rep.rows if r["suggest_bad"]}
    out.add_bads(bads)
    out.meta["bad_channel_reasons"] = reasons
    out.history.append(f"marked bad {bads}")
    return out


def _step_set_bads(raw, channels):
    out = raw.copy()
    out.add_bads(channels)
    return out


def _step_interpolate(raw, **kw):
    return raw.interpolate_bads(**kw)


def _step_ica(raw, n_components=0.99, eog=True, ecg=True, muscle=False, random_state=0):
    from .preprocessing.ica import remove_artifacts_ica

    clean, ica = remove_artifacts_ica(raw, n_components=n_components, eog=eog, ecg=ecg, muscle=muscle,
                                      random_state=random_state)
    clean.meta["ica_excluded"] = {int(k): ica.labels_.get(k, "?") for k in ica.exclude}
    return clean


def _step_annotate(raw, amplitude=True, muscle=True, blinks=False, **kw):
    from .preprocessing.artifacts import annotate_artifacts

    return annotate_artifacts(raw, amplitude=amplitude, muscle=muscle, blinks=blinks, **kw)


def _step_laplacian(raw, method="hjorth", **kw):
    from .preprocessing.reference import laplacian

    return laplacian(raw, method=method, **kw)


def _step_bipolar(raw, pairs="double_banana", **kw):
    from .preprocessing.reference import bipolar_reference

    return bipolar_reference(raw, pairs, **kw)


STEPS: dict[str, Callable[..., Any]] = {
    "filter": _step_filter, "bandpass": _step_filter, "notch": _step_notch, "resample": _step_resample,
    "rereference": _step_rereference, "reference": _step_rereference, "pick": _step_pick,
    "drop_channels": _step_drop, "crop": _step_crop, "set_montage": _step_montage,
    "rename_channels": _step_rename, "set_channel_types": _step_types,
    "standardize_names": _step_standardize, "detect_bad_channels": _step_bads, "set_bads": _step_set_bads,
    "interpolate_bad_channels": _step_interpolate, "interpolate_bads": _step_interpolate, "ica": _step_ica,
    "annotate_artifacts": _step_annotate, "laplacian": _step_laplacian, "bipolar": _step_bipolar,
}


class Pipeline:
    """An ordered list of ``(step_name, kwargs)`` applied to a recording."""

    def __init__(self, steps: list | None = None, output: str | None = None, report: str | None = None):
        self.steps: list[tuple[str, dict]] = []
        for item in steps or []:
            self.add(item)
        self.output = output
        self.report = report
        self.log: list[dict] = []

    def add(self, item, **kwargs) -> "Pipeline":
        """Add a step: ``add('filter', l_freq=1)``, ``add({'filter': {...}})`` or ``add(('filter', {...}))``."""
        if isinstance(item, str):
            name, params = item, kwargs
        elif isinstance(item, dict):
            if "step" in item:
                params = {k: v for k, v in item.items() if k != "step"}
                name = item["step"]
            elif len(item) == 1:
                name, params = next(iter(item.items()))
                params = params or {}
            else:
                raise ValueError(f"Cannot parse pipeline step {item!r}")
        else:
            name, params = item
        if name not in STEPS:
            raise ValueError(f"Unknown step {name!r}. Available: {', '.join(sorted(STEPS))}")
        self.steps.append((name, dict(params)))
        return self

    @classmethod
    def from_config(cls, config: dict | str | os.PathLike) -> "Pipeline":
        """Build from a dict, or a ``.json`` / ``.yaml`` / ``.yml`` file."""
        if not isinstance(config, dict):
            path = os.fspath(config)
            with open(path, "r", encoding="utf-8") as f:
                text = f.read()
            if path.lower().endswith((".yaml", ".yml")):
                try:
                    import yaml
                except ImportError as err:
                    raise ImportError("YAML configs need pyyaml: pip install pyyaml (or use JSON)") from err
                config = yaml.safe_load(text)
            else:
                config = json.loads(text)
        if isinstance(config, list):
            config = {"steps": config}
        return cls(config.get("steps", []), config.get("output"), config.get("report"))

    def to_config(self) -> dict:
        return {"steps": [{n: p} for n, p in self.steps], "output": self.output, "report": self.report}

    def __repr__(self) -> str:
        return "<Pipeline | " + " -> ".join(n for n, _ in self.steps) + ">"

    def run(self, raw, verbose: bool = False):
        """Apply every step in order; returns the processed :class:`RawEEG`."""
        self.log = []
        current = raw
        for name, params in self.steps:
            t0 = time.perf_counter()
            current = STEPS[name](current, **params)
            dt = time.perf_counter() - t0
            entry = {"step": name, "params": params, "seconds": round(dt, 3),
                     "n_channels": current.n_channels, "sfreq": current.sfreq, "bads": list(current.bads)}
            self.log.append(entry)
            msg = f"{name}({', '.join(f'{k}={v}' for k, v in params.items())}) in {dt:.2f}s"
            if verbose:
                print("  ✓", msg)
            logger.info(msg)
        if self.output:
            current.save(self.output)
        if self.report:
            from .report import generate_report

            generate_report(current, self.report)
        return current
