"""Analysis state and the EEG tools the assistant calls.

Everything here runs on the user's machine. A tool returns text and figure
files; only that text and those figures go to the language model, never the
signals themselves. Patient fields and the recording date from file headers
are left out of every summary.
"""

from __future__ import annotations

import contextlib
import io
import traceback
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

import matplotlib.pyplot as plt
import numpy as np

from .. import viz
from ..analysis import CONNECTIVITY_METHODS, compute_connectivity, compute_psd, extract_features, summarize_bands
from ..core.channels import CHANNEL_TYPES
from ..core.epochs import Epochs, find_events, make_fixed_length_epochs
from ..core.raw import RawEEG
from ..datasets import fetch_openneuro_recording, simulate_eeg
from ..io import read_raw
from ..preprocessing import (ICA, annotate_artifacts, bad_time_fraction, bipolar_reference, detect_line_noise,
                             filter_data, find_blinks, laplacian)
from ..preprocessing.artifacts import FRONTAL_EOG_PROXIES
from ..quality import erd_ers, rank_channels
from ..report import generate_report
from ..utils import format_table
from ..viz import _style as S

MAX_TEXT = 15000
SAVE_FORMATS = (".edf", ".bdf", ".vhdr", ".csv", ".tsv", ".txt", ".npz", ".mat", ".h5", ".hdf5", ".fif")
EPOCH_RANKINGS = ("erp", "erd", "discriminability", "decoding")
RAW_RANKINGS = ("quality", "snr", "delta", "theta", "alpha", "beta", "gamma", "connectivity")
DEMOS = {
    "resting": dict(duration=180, montage="32", eog=True, seed=1),
    "blinks": dict(duration=120, montage="32", eog=True, bad_channels={}, seed=42),
    "oddball": dict(duration=300, montage="32", eog=True, task="oddball", seed=2),
    "motor_imagery": dict(duration=300, montage="64", task="motor_imagery", bad_channels={}, seed=3),
}


@dataclass
class ToolOutput:
    text: str
    figures: list[Path] = field(default_factory=list)
    is_error: bool = False


class ToolError(Exception):
    """A problem the model should read and fix (wrong argument, missing step)."""


# ---------------------------------------------------------------- tool registry
_TOOLS: dict[str, dict] = {}


def _tool(description: str, properties: dict | None = None, required: tuple = ()):
    def deco(fn):
        _TOOLS[fn.__name__.removeprefix("tool_")] = {
            "method": fn.__name__,
            "description": " ".join(description.split()),
            "input_schema": {"type": "object", "properties": properties or {},
                             "required": list(required), "additionalProperties": False},
        }
        return fn
    return deco


def _p(kind, description: str, **extra) -> dict:
    return {"type": kind, "description": description, **extra}


def _names(description: str) -> dict:
    return _p("array", description, items={"type": "string"})


def _range(description: str) -> dict:
    return _p("array", description, items={"type": "number"}, minItems=2, maxItems=2)


_PY_TYPES = {"string": str, "boolean": bool, "object": dict, "array": list, "null": type(None)}


def _type_ok(value, kind) -> bool:
    kinds = kind if isinstance(kind, list) else [kind]
    for k in kinds:
        if k == "number" and isinstance(value, (int, float)) and not isinstance(value, bool):
            return True
        if k == "integer" and isinstance(value, int) and not isinstance(value, bool):
            return True
        if k in _PY_TYPES and isinstance(value, _PY_TYPES[k]):
            return True
    return False


def validate_input(schema: dict, args) -> list[str]:
    """Problems with a tool call's arguments (empty list = valid)."""
    if not isinstance(args, dict):
        return ["arguments must be a JSON object"]
    props = schema["properties"]
    problems = [f"unknown argument '{k}'" for k in args if k not in props]
    problems += [f"missing required argument '{k}'" for k in schema["required"] if k not in args]
    for key, value in args.items():
        spec = props.get(key)
        if spec is None:
            continue
        if not _type_ok(value, spec["type"]):
            problems.append(f"'{key}' must be {spec['type']}, got {type(value).__name__}")
        elif "enum" in spec and value not in spec["enum"]:
            problems.append(f"'{key}' must be one of {spec['enum']}")
        elif isinstance(value, list):
            item = spec.get("items", {})
            if "type" in item and not all(_type_ok(v, item["type"]) for v in value):
                problems.append(f"every item of '{key}' must be {item['type']}")
            if len(value) < spec.get("minItems", 0) or len(value) > spec.get("maxItems", len(value)):
                problems.append(f"'{key}' has the wrong number of items")
    return problems


def _clip(text: str) -> str:
    if len(text) <= MAX_TEXT:
        return text
    return text[:MAX_TEXT] + f"\n... [{len(text) - MAX_TEXT} more characters not shown]"


def _overview(raw: RawEEG) -> str:
    """Recording summary without patient fields or the recording date."""
    mins, secs = divmod(raw.duration, 60)
    types = ", ".join(f"{v} {k}" for k, v in raw.type_counts().items())
    lines = [
        f"File        : {Path(raw.filename).name if raw.filename else '(simulated / in memory)'}",
        f"Format      : {raw.meta.get('format') or 'n/a'}",
        f"Sampling    : {raw.sfreq:g} Hz, {int(mins)} min {secs:.1f} s",
        f"Channels    : {raw.n_channels} ({types})",
        f"Positions   : {len(raw.positions)} channels have electrode positions",
        f"Bad channels: {', '.join(raw.bads) if raw.bads else 'none'}",
        f"Filters     : highpass {raw.meta.get('highpass', 'n/a')} Hz, lowpass {raw.meta.get('lowpass', 'n/a')} Hz",
    ]
    if len(raw.annotations):
        counts = ", ".join(f"{k} (x{v})" for k, v in list(raw.annotations.count().items())[:15])
        lines.append(f"Annotations : {len(raw.annotations)} -> {counts}")
    if raw.meta.get("subject") or raw.meta.get("meas_date"):
        lines.append("Privacy     : patient fields and recording date in the file header are not shown")
    rows = [{k: r[k] for k in ("name", "type", "unit", "bad", "std", "min", "max")} for r in raw.channel_table()]
    lines += ["", "Per-channel amplitude (std/min/max in the channel's unit):", format_table(rows)]
    return "\n".join(lines)


class AnalysisSession:
    """The data being analysed, its processing history, and the tools.

    Parameters
    ----------
    output_dir : folder for figures, saved data, reports and feature tables.
    confirm : ``callable(question) -> bool`` asked before running model-written Python.
    max_undo : how many processing steps can be undone.
    """

    def __init__(self, output_dir: str | Path = "eeg_assistant_output",
                 confirm: Callable[[str], bool] | None = None, max_undo: int = 5, figure_dpi: int = 100):
        self.output_dir = Path(output_dir)
        self.figures_dir = self.output_dir / "figures"
        self.figures_dir.mkdir(parents=True, exist_ok=True)
        self.confirm = confirm or (lambda question: False)
        self.max_undo = max_undo
        self.figure_dpi = figure_dpi
        self.raw: RawEEG | None = None
        self.ica: ICA | None = None
        self.epochs: Epochs | None = None
        self.quality = None
        self.steps: list[str] = []
        self._undo: list[tuple] = []
        self._n_fig = 0

    # ------------------------------------------------------------ plumbing
    @staticmethod
    def tool_definitions(allow_python: bool = True) -> list[dict]:
        """Tool schemas for the Messages API, in a fixed order."""
        return [{"name": name, "description": spec["description"], "input_schema": spec["input_schema"]}
                for name, spec in _TOOLS.items() if allow_python or name != "run_python"]

    def call(self, name: str, args: dict) -> ToolOutput:
        """Validate and run one tool; errors come back as text for the model."""
        spec = _TOOLS.get(name)
        if spec is None:
            return ToolOutput(f"Unknown tool '{name}'.", is_error=True)
        problems = validate_input(spec["input_schema"], args)
        if problems:
            return ToolOutput("Invalid arguments: " + "; ".join(problems), is_error=True)
        try:
            out = getattr(self, spec["method"])(**args)
        except ToolError as err:
            return ToolOutput(str(err), is_error=True)
        except Exception as err:  # the toolkit's own errors are informative for the model
            return ToolOutput(f"{type(err).__name__}: {err}", is_error=True)
        finally:
            plt.close("all")
        out.text = _clip(out.text)
        return out

    def _need_raw(self) -> RawEEG:
        if self.raw is None:
            raise ToolError("No recording is loaded yet. Call load_recording first.")
        return self.raw

    def _need_epochs(self) -> Epochs:
        if self.epochs is None:
            raise ToolError("No epochs yet. Call make_epochs first.")
        return self.epochs

    def _check_channels(self, names) -> list[str]:
        missing = [n for n in names if n not in self._need_raw().ch_names]
        if missing:
            raise ToolError(f"Unknown channels {missing}. Available: {', '.join(self.raw.ch_names)}")
        return list(names)

    def _commit(self, new_raw: RawEEG, step: str, keep_ica: bool = False) -> str:
        self._undo.append((step, self.raw, self.ica, self.epochs, self.quality))
        del self._undo[:-self.max_undo]
        self.raw = new_raw
        self.steps.append(step)
        notes = []
        if self.epochs is not None:
            notes.append("epochs were cleared (make them again from the new data)")
        if self.ica is not None and not keep_ica:
            notes.append("the ICA fit was cleared (run_ica again if needed)")
        self.epochs = None
        self.quality = None
        if not keep_ica:
            self.ica = None
        return f"Done: {step}." + (f" Note: {'; '.join(notes)}." if notes else "")

    def _figure(self, fig, name: str) -> Path:
        self._n_fig += 1
        path = self.figures_dir / f"{self._n_fig:02d}_{name}.png"
        fig.savefig(path, dpi=self.figure_dpi, bbox_inches="tight")
        plt.close(fig)
        return path

    def _output_path(self, filename: str, allowed: tuple) -> Path:
        name = Path(filename).name
        if not name.lower().endswith(allowed):
            raise ToolError(f"File name must end with one of {', '.join(allowed)}")
        return self.output_dir / name

    # ------------------------------------------------------------ data
    @_tool("""Load an EEG recording (EDF/EDF+, BDF, BrainVision .vhdr, CSV/TSV/TXT, MATLAB .mat,
           EEGLAB .set, .npz, HDF5, XDF, and through MNE: FIF, GDF, CNT, MFF ...). When the user has no
           file, use 'demo:resting', 'demo:blinks', 'demo:oddball' or 'demo:motor_imagery' for simulated
           data, or a real public recording from OpenNeuro: 'openneuro:<dataset>/<file>', e.g.
           'openneuro:ds004362/sub-001/eeg/sub-001_task-motion_run-1_eeg.set' (eyes open; run-2 is eyes
           closed, runs 4/8/12 imagined left/right fist) or
           'openneuro:ds003061/sub-001/eeg/sub-001_task-P300_run-1_eeg.set' (auditory oddball, 64 MB).
           Replaces the current data and clears the processing history.""",
           {"path": _p("string", "File path, demo:<kind> or openneuro:<dataset>/<file>"),
            "montage": _p("string", "Electrode layout for channels without positions: 'standard_1010', "
                                    "'standard_1020' or 'none'. Default standard_1010.")},
           ("path",))
    def tool_load_recording(self, path: str, montage: str = "standard_1010") -> ToolOutput:
        if path.startswith("demo:"):
            kind = path[5:]
            if kind not in DEMOS:
                raise ToolError(f"Unknown demo '{kind}'. Choose from {', '.join(DEMOS)}.")
            raw = simulate_eeg(**DEMOS[kind])
        elif path.startswith("openneuro:"):
            dataset, _, rel = path[len("openneuro:"):].partition("/")
            if not rel:
                raise ToolError("Use openneuro:<dataset>/<file path inside the dataset>.")
            raw = read_raw(fetch_openneuro_recording(dataset, rel))
        else:
            p = Path(path).expanduser()
            if not p.exists():
                here = sorted(f.name for f in Path.cwd().iterdir() if f.suffix.lower() in
                              (".edf", ".bdf", ".vhdr", ".csv", ".txt", ".tsv", ".mat", ".set", ".xdf",
                               ".fif", ".gdf", ".cnt", ".npz", ".h5"))
                raise ToolError(f"File not found: {path}. EEG files in {Path.cwd()}: {here or 'none'}")
            raw = read_raw(p)
        if montage != "none" and len(raw.positions) < len(raw.pick_names("eeg")):
            raw.set_montage(montage, overwrite=False)
        self.raw, self.ica, self.epochs, self.quality = raw, None, None, None
        self.steps, self._undo = [], []
        notes = []
        if not raw.pick_names("eeg"):
            notes.append("No channel is typed as EEG; use set_channel_types to fix the types.")
        if not raw.pick_names("eog"):
            notes.append("No EOG channel; blink detection and ICA eye labelling will use frontal channels.")
        return ToolOutput(_overview(raw) + ("\n\n" + "\n".join(notes) if notes else ""))

    @_tool("""Summary of the data as it is now: channels, types, bad channels, annotations, the
           processing steps applied so far, and whether ICA or epochs exist. Call this whenever you
           need to re-check the current state.""")
    def tool_describe_data(self) -> ToolOutput:
        raw = self._need_raw()
        lines = [_overview(raw), "", "Processing steps: " + (" -> ".join(self.steps) or "none yet")]
        if self.ica is not None:
            lines.append(f"ICA: fitted, {self.ica.n_components_} components")
        if self.epochs is not None:
            lines.append(f"Epochs: {self.epochs.n_epochs} ({', '.join(self.epochs.event_id)}), "
                         f"{self.epochs.tmin:g} to {self.epochs.times[-1]:g} s")
        return ToolOutput("\n".join(lines))

    @_tool("""Change channel types, e.g. mark 'EXG1' as eog or a trigger channel as stim. Use it when
           load_recording shows channels with the wrong type.""",
           {"types": _p("object", f"Map of channel name to type ({', '.join(CHANNEL_TYPES)})")}, ("types",))
    def tool_set_channel_types(self, types: dict) -> ToolOutput:
        self._check_channels(types)
        wrong = {k: v for k, v in types.items() if v not in CHANNEL_TYPES}
        if wrong:
            raise ToolError(f"Unknown types {wrong}; use one of {CHANNEL_TYPES}")
        new = self.raw.copy().set_channel_types(types)
        return ToolOutput(self._commit(new, f"set channel types {types}"))

    @_tool("Remove channels from the data entirely (unlike mark_bad_channels, which keeps them).",
           {"channels": _names("Channel names to remove")}, ("channels",))
    def tool_drop_channels(self, channels: list) -> ToolOutput:
        self._check_channels(channels)
        return ToolOutput(self._commit(self.raw.drop_channels(channels), f"drop {', '.join(channels)}"))

    # ------------------------------------------------------------ quality
    @_tool("""Score every EEG channel 0-100 from 12 metrics (flat, noise, line noise, drift, artifacts,
           correlation with neighbours, PREP criteria ...) and list suspected bad channels with reasons.
           Use it early in every analysis and again after cleaning. Read-only; also draws a dashboard.""")
    def tool_assess_quality(self) -> ToolOutput:
        raw = self._need_raw()
        self.quality = raw.assess_quality()
        text = (f"Overall score {self.quality.overall_score:.0f}/100. Suggested bad channels: "
                f"{', '.join(self.quality.bad_channels) or 'none'}\n\n" + self.quality.summary())
        return ToolOutput(text, [self._figure(self.quality.plot(), "quality")])

    @_tool("""Mark channels as bad (or un-mark them). Bad channels are skipped by analyses and can be
           repaired with interpolate_bad_channels. Ask the user before marking.""",
           {"channels": _names("Channel names"),
            "action": _p("string", "add (default), remove, or replace the whole list",
                         enum=["add", "remove", "replace"])},
           ("channels",))
    def tool_mark_bad_channels(self, channels: list, action: str = "add") -> ToolOutput:
        self._check_channels(channels)
        bads = {"add": set(self.raw.bads) | set(channels), "remove": set(self.raw.bads) - set(channels),
                "replace": set(channels)}[action]
        new = self.raw.copy().set_bads(sorted(bads, key=self.raw.ch_names.index))
        return ToolOutput(self._commit(new, f"bad channels -> {', '.join(new.bads) or 'none'}", keep_ica=True))

    # ------------------------------------------------------------ cleaning
    @_tool("""Zero-phase band-pass, high-pass (h_freq omitted) or low-pass (l_freq omitted) filter on
           all data channels. Changes the data; agree the cut-offs with the user first.""",
           {"l_freq": _p("number", "High-pass cut-off in Hz"), "h_freq": _p("number", "Low-pass cut-off in Hz")})
    def tool_filter_data(self, l_freq: float | None = None, h_freq: float | None = None) -> ToolOutput:
        raw = self._need_raw()
        if l_freq is None and h_freq is None:
            raise ToolError("Give l_freq, h_freq or both.")
        label = (f"band-pass {l_freq:g}-{h_freq:g} Hz" if l_freq and h_freq else
                 f"high-pass {l_freq:g} Hz" if l_freq else f"low-pass {h_freq:g} Hz")
        return ToolOutput(self._commit(raw.filter(l_freq, h_freq), label))

    @_tool("""Remove power-line interference (and harmonics) with a notch filter. Without freq, the line
           frequency (50 or 60 Hz) is detected from the data and nothing changes if none is found.""",
           {"freq": _p("number", "Line frequency in Hz")})
    def tool_notch_filter(self, freq: float | None = None) -> ToolOutput:
        raw = self._need_raw()
        if freq is None:
            freq = detect_line_noise(raw.get_data("eeg"), raw.sfreq)
            if freq is None:
                return ToolOutput("No line noise detected; the data were not changed.")
        return ToolOutput(self._commit(raw.notch_filter(freq), f"notch {freq:g} Hz"))

    @_tool("""Re-reference the EEG: average reference, reference to chosen channels (e.g. linked
           mastoids), bipolar montages, or a surface Laplacian (sharpens local activity, useful for
           motor imagery). Changes the data.""",
           {"kind": _p("string", "Reference scheme", enum=["average", "channels", "bipolar_double_banana",
                                                             "bipolar_transverse", "laplacian_hjorth",
                                                             "laplacian_spline"]),
            "channels": _names("Reference channels when kind is 'channels'")},
           ("kind",))
    def tool_set_reference(self, kind: str, channels: list | None = None) -> ToolOutput:
        raw = self._need_raw()
        if kind == "channels":
            if not channels:
                raise ToolError("Give the reference channels.")
            new = raw.set_reference(self._check_channels(channels))
        elif kind == "average":
            new = raw.set_reference("average")
        elif kind.startswith("bipolar"):
            new = bipolar_reference(raw, kind.split("_", 1)[1])
        else:
            new = laplacian(raw, kind.split("_", 1)[1])
        return ToolOutput(self._commit(new, f"reference: {kind}" + (f" {channels}" if channels else "")))

    @_tool("Repair the channels marked bad from their neighbours (spherical splines). Changes the data.")
    def tool_interpolate_bad_channels(self) -> ToolOutput:
        raw = self._need_raw()
        if not raw.bads:
            return ToolOutput("No channels are marked bad; nothing to interpolate.")
        bads = ", ".join(raw.bads)
        return ToolOutput(self._commit(raw.interpolate_bads(), f"interpolate {bads}"))

    @_tool("Change the sampling rate (anti-aliased; trigger codes are kept). Changes the data.",
           {"sfreq": _p("number", "New sampling rate in Hz")}, ("sfreq",))
    def tool_resample(self, sfreq: float) -> ToolOutput:
        raw = self._need_raw()
        return ToolOutput(self._commit(raw.resample(sfreq), f"resample {raw.sfreq:g} -> {sfreq:g} Hz"))

    @_tool("Keep only a time range of the recording. Changes the data.",
           {"tmin": _p("number", "Start in seconds"), "tmax": _p("number", "End in seconds (default: end)")})
    def tool_crop(self, tmin: float = 0.0, tmax: float | None = None) -> ToolOutput:
        raw = self._need_raw()
        end = raw.duration if tmax is None else tmax
        return ToolOutput(self._commit(raw.crop(tmin, tmax), f"crop {tmin:g}-{end:g} s"))

    @_tool("""Fit ICA to separate the EEG into independent components and label likely eye, heart and
           muscle components. Read-only: show the user the table and component maps, recommend which
           to remove, then call remove_ica_components once they agree.""",
           {"n_components": _p("number", "Number of components, or a fraction of variance (default 0.99)")})
    def tool_run_ica(self, n_components: float = 0.99) -> ToolOutput:
        raw = self._need_raw()
        ica = ICA(n_components=int(n_components) if n_components >= 1 else n_components,
                  random_state=0).fit(raw)
        found = {}
        for label, finder in (("eye", ica.find_bads_eog), ("heart", ica.find_bads_ecg),
                              ("muscle", ica.find_bads_muscle)):
            with contextlib.suppress(ValueError):
                found[label] = finder(raw)[0]
        self.ica = ica
        table = format_table(ica.component_table(raw),
                             ["component", "variance_pct", "kurtosis", "eog_r", "spectral_slope", "label"])
        suggestion = "; ".join(f"{k}: {v}" for k, v in found.items() if v) or "none"
        text = (f"ICA fitted with {ica.n_components_} components on {len(ica.ch_names)} channels.\n"
                f"Suggested for removal -> {suggestion}\n\n{table}")
        figs = [self._figure(ica.plot_components(), "ica_components")] if raw.has_positions("eeg") else []
        return ToolOutput(text, figs)

    @_tool("Remove the chosen ICA components from the data (after run_ica). Changes the data.",
           {"components": _p("array", "Component numbers", items={"type": "integer"})}, ("components",))
    def tool_remove_ica_components(self, components: list) -> ToolOutput:
        raw = self._need_raw()
        if self.ica is None:
            raise ToolError("Run run_ica on the current data first.")
        bad = [c for c in components if not 0 <= c < self.ica.n_components_]
        if bad:
            raise ToolError(f"Components {bad} do not exist (0-{self.ica.n_components_ - 1}).")
        new = self.ica.apply(raw, exclude=components)
        return ToolOutput(self._commit(new, f"ICA removed components {components}"))

    @_tool("""Find and mark bad time segments: large amplitude excursions and muscle bursts. Marked
           segments are skipped by spectra and dropped from epochs. Changes the annotations.""",
           {"amplitude": _p("boolean", "Mark peak-to-peak excursions (default true)"),
            "muscle": _p("boolean", "Mark muscle bursts (default true)"),
            "peak_to_peak": _p("number", "Amplitude threshold in µV (default 150)")})
    def tool_annotate_artifacts(self, amplitude: bool = True, muscle: bool = True,
                                peak_to_peak: float = 150.0) -> ToolOutput:
        raw = self._need_raw()
        new = annotate_artifacts(raw, amplitude=amplitude, muscle=muscle, peak_to_peak=peak_to_peak)
        added = len(new.annotations) - len(raw.annotations)
        msg = self._commit(new, f"mark artifacts ({added} segments)", keep_ica=True)
        return ToolOutput(f"{msg} {100 * bad_time_fraction(new):.1f}% of the recording is now marked bad.")

    # ------------------------------------------------------------ blinks
    @_tool("""Detect eye blinks on the EOG channel (or Fp1/Fp2) and report count, rate per minute,
           width, amplitude and regularity, with a figure. With add_annotations the blinks are added
           as 'blink' events so make_epochs can cut blink-locked epochs.""",
           {"channel": _p("string", "Channel to detect on (default: EOG, else frontal)"),
            "add_annotations": _p("boolean", "Add 'blink' annotations to the data (default false)")})
    def tool_detect_blinks(self, channel: str | None = None, add_annotations: bool = False) -> ToolOutput:
        raw = self._need_raw()
        ch = channel or next(iter(raw.pick_names("eog")), None) or \
            next((c for c in FRONTAL_EOG_PROXIES if c in raw.ch_names), None)
        if ch is None:
            raise ToolError("No EOG or frontal channel; pass channel=...")
        self._check_channels([ch])
        blinks = find_blinks(raw, channel=ch)
        minutes = raw.duration / 60
        n = len(blinks)
        lines = [f"Channel {ch}: {n} blinks in {minutes:.1f} min = {n / minutes:.1f} blinks/min"]
        # Shape and amplitude from a 20 Hz low-pass with a pre-blink baseline: the 1-10 Hz band used
        # for detection would add artificial dips on both sides of each blink.
        x = filter_data(raw[ch], raw.sfreq, None, 20.0)
        half, base = int(0.5 * raw.sfreq), int(0.1 * raw.sfreq)
        peaks = ((blinks.onset + blinks.duration / 2) * raw.sfreq).astype(int)
        peaks = peaks[(peaks >= half) & (peaks < raw.n_times - half)]
        shapes = np.array([x[p - half:p + half] - x[p - half:p - half + base].mean() for p in peaks])
        if n:
            ibi = np.diff(blinks.onset)
            lines.append(f"Width (at half height x2): median {1000 * np.median(blinks.duration):.0f} ms, "
                         f"range {1000 * blinks.duration.min():.0f}-{1000 * blinks.duration.max():.0f} ms")
            if len(shapes):
                amp = np.abs(shapes[:, half])
                lines.append(f"Amplitude (vs 500-400 ms before the peak): median {np.median(amp):.0f} µV, "
                             f"range {amp.min():.0f}-{amp.max():.0f} µV")
            if len(ibi) > 1:
                lines.append(f"Inter-blink interval: median {np.median(ibi):.2f} s, "
                             f"CV {ibi.std() / ibi.mean():.2f} (0 = perfectly regular)")
        figs = [self._figure(self._blink_figure(raw, ch, blinks, shapes), "blinks")] if n else []
        if add_annotations and n:
            new = raw.copy()
            new.annotations.extend(blinks)
            new.annotations = new.annotations.sorted()
            lines.append(self._commit(new, f"add {n} blink annotations", keep_ica=True))
        return ToolOutput("\n".join(lines), figs)

    @staticmethod
    def _blink_figure(raw, ch, blinks, shapes):
        t = raw.times
        y = filter_data(raw[ch], raw.sfreq, 0.5, 20.0)
        view = t < min(60.0, t[-1])
        lag = (np.arange(shapes.shape[1]) - shapes.shape[1] // 2) / raw.sfreq * 1000 if len(shapes) else []
        with S.style():
            fig, (a1, a2) = plt.subplots(2, 1, figsize=(10, 5.5), layout="constrained",
                                         gridspec_kw={"height_ratios": [3, 2]})
            for on, dur in zip(blinks.onset, blinks.duration):
                if on < t[view][-1]:
                    a1.axvspan(on, on + dur, color=S.CATEGORICAL[3], alpha=0.35, lw=0)
            a1.plot(t[view], y[view], color=S.CATEGORICAL[0], lw=0.8)
            a1.set(xlabel="Time (s)", ylabel=f"{ch} 0.5-20 Hz (µV)", xlim=(0, t[view][-1]))
            a1.set_title(f"{ch}: detected blinks shaded (first {t[view][-1]:.0f} s)", loc="left")
            for tr in shapes[:60]:
                a2.plot(lag, tr, color=S.AXIS, lw=0.5)
            if len(shapes):
                a2.plot(lag, shapes.mean(axis=0), color=S.CATEGORICAL[0], lw=2.2)
            a2.set(xlabel="Time from blink peak (ms)", ylabel="µV (low-pass 20 Hz)")
            a2.set_title(f"Blink shape: {len(shapes)} blinks (grey) and their average", loc="left")
        return fig

    # ------------------------------------------------------------ spectra
    @_tool("""Power spectrum of the EEG channels: band power per channel (delta, theta, alpha, beta,
           gamma), peak alpha frequency, 1/f exponent, plus a spectrum plot and band scalp maps.
           Read-only. Use it for resting-state questions and to check filtering and line noise.""",
           {"fmin": _p("number", "Lowest frequency (default 1)"), "fmax": _p("number", "Highest (default 45)"),
            "channels": _names("Channels to include (default: all EEG)"),
            "method": _p("string", "Estimator", enum=["welch", "multitaper"])})
    def tool_compute_spectrum(self, fmin: float = 1.0, fmax: float = 45.0, channels: list | None = None,
                              method: str = "welch") -> ToolOutput:
        raw = self._need_raw()
        picks = self._check_channels(channels) if channels else "eeg"
        spec = compute_psd(raw, picks=picks, method=method, fmin=fmin, fmax=fmax)
        paf = spec.peak_frequency()
        _, exponent = spec.aperiodic()
        rows = spec.band_power_table(relative=True)
        text = [f"{len(spec.ch_names)} channels, {fmin:g}-{fmax:g} Hz, {method}",
                "", "Strongest channel per band:", summarize_bands(spec),
                "", f"Peak alpha frequency: median {np.nanmedian(paf):.2f} Hz "
                    f"(range {np.nanmin(paf):.2f}-{np.nanmax(paf):.2f})",
                f"1/f exponent: median {np.nanmedian(exponent):.2f}",
                "", "Relative band power per channel (fraction of total):", format_table(rows)]
        figs = [self._figure(viz.plot_psd(raw, picks=picks, fmax=fmax), "spectrum")]
        if raw.has_positions("eeg") and not channels:
            figs.append(self._figure(viz.plot_band_topomaps(raw), "band_maps"))
        return ToolOutput("\n".join(text), figs)

    @_tool("""Functional connectivity between EEG channels in a frequency band (coherence, imaginary
           coherence, PLV, PLI, wPLI, amplitude-envelope correlation). Read-only. wPLI and imcoh are
           robust to volume conduction.""",
           {"method": _p("string", "Measure", enum=list(CONNECTIVITY_METHODS)),
            "band": _range("Frequency band [low, high] in Hz (default [8, 13])")})
    def tool_compute_connectivity(self, method: str = "wpli", band: list | None = None) -> ToolOutput:
        raw = self._need_raw()
        con = compute_connectivity(raw, method=method, band=tuple(band or (8.0, 13.0)))
        return ToolOutput(con.summary(15), [self._figure(con.plot(), f"connectivity_{method}")])

    # ------------------------------------------------------------ events
    @_tool("""List the events/markers in the recording (from annotations or a trigger channel) with
           counts and first times. Read-only. Call before make_epochs.""")
    def tool_list_events(self) -> ToolOutput:
        raw = self._need_raw()
        events, event_id = find_events(raw)
        lines = []
        if len(events):
            rows = [{"event": k, "code": v, "count": int((events[:, 2] == v).sum()),
                     "first_s": float(events[events[:, 2] == v, 0][0] / raw.sfreq)}
                    for k, v in event_id.items()]
            lines += [f"{len(events)} events:", format_table(rows)]
        else:
            lines.append("No events found. For resting-state data, use make_epochs with fixed_length_s.")
        bad = {k: v for k, v in raw.annotations.count().items() if k.upper().startswith("BAD")}
        if bad:
            lines.append(f"Bad-segment annotations: {bad} ({100 * bad_time_fraction(raw):.1f}% of time)")
        return ToolOutput("\n".join(lines))

    @_tool("""Cut epochs around events (task data) or fixed-length segments (resting state), with
           baseline correction and optional amplitude rejection. Needed before compute_erp,
           compute_erd_ers and the epoch-based rank_channels options.""",
           {"event_names": _names("Events to keep (default: all)"),
            "tmin": _p("number", "Start relative to the event, s (default -0.2)"),
            "tmax": _p("number", "End relative to the event, s (default 0.8)"),
            "baseline": _p(["array", "null"], "Baseline [start, end] in s (default [tmin, 0]); null for none",
                           items={"type": "number"}),
            "reject_uv": _p("number", "Drop epochs whose EEG peak-to-peak exceeds this many µV"),
            "fixed_length_s": _p("number", "Cut consecutive segments of this length instead of using events")})
    def tool_make_epochs(self, event_names: list | None = None, tmin: float = -0.2, tmax: float = 0.8,
                         baseline: list | None | str = "default", reject_uv: float | None = None,
                         fixed_length_s: float | None = None) -> ToolOutput:
        raw = self._need_raw()
        if fixed_length_s:
            ep = make_fixed_length_epochs(raw, duration=fixed_length_s, reject=reject_uv)
        else:
            events, event_id = find_events(raw)
            if not len(events):
                raise ToolError("No events in the data; use fixed_length_s for resting-state segments.")
            if event_names:
                unknown = [n for n in event_names if n not in event_id]
                if unknown:
                    raise ToolError(f"Unknown events {unknown}. Available: {list(event_id)}")
                event_id = {n: event_id[n] for n in event_names}
            if baseline == "default":
                baseline = (tmin, 0.0) if tmin < 0 else None
            elif baseline is not None:
                if len(baseline) != 2:
                    raise ToolError("baseline must be [start, end]")
                baseline = tuple(baseline)
            ep = Epochs(raw, events, event_id, tmin=tmin, tmax=tmax, baseline=baseline, reject=reject_uv)
        if ep.n_epochs == 0:
            raise ToolError(f"All {ep.n_dropped} epochs were dropped; relax reject_uv or check bad segments.")
        self.epochs = ep
        return ToolOutput(f"{ep.n_epochs} epochs kept, {ep.n_dropped} dropped, "
                          f"{ep.times[0]:g} to {ep.times[-1]:g} s\n{ep.summary_table()}")

    @_tool("""Average epochs into ERPs per condition and report the largest peak (channel, latency,
           amplitude) in a window, with waveform and scalp-map figures. Needs make_epochs.""",
           {"conditions": _names("Conditions to average (default: all)"),
            "channels": _names("Channels to draw (default: a few with the largest response)"),
            "window": _range("Peak search window [start, end] in s (default [0.05, 0.6])")})
    def tool_compute_erp(self, conditions: list | None = None, channels: list | None = None,
                         window: list | None = None) -> ToolOutput:
        ep = self._need_epochs()
        evokeds = ep.average(by_condition=True)
        if conditions:
            unknown = [c for c in conditions if c not in evokeds]
            if unknown:
                raise ToolError(f"Unknown conditions {unknown}. Available: {list(evokeds)}")
            evokeds = {c: evokeds[c] for c in conditions}
        lo, hi = window or (0.05, 0.6)
        rows, figs = [], []
        for name, ev in evokeds.items():
            for mode in ("pos", "neg"):
                ch, lat, amp = ev.get_peak(tmin=lo, tmax=hi, mode=mode)
                rows.append({"condition": name, "n_epochs": ev.nave, "polarity": mode, "channel": ch,
                             "latency_ms": 1000 * lat, "amplitude_uV": amp})
        for name, ev in list(evokeds.items())[:4]:
            picks = self._check_channels(channels) if channels else None
            figs.append(self._figure(viz.plot_evoked(ev, picks=picks, title=f"ERP: {name}"), f"erp_{name}"))
            if ev.positions:
                figs.append(self._figure(viz.plot_evoked_topomaps(ev), f"erp_maps_{name}"))
        return ToolOutput(f"Peaks between {1000 * lo:.0f} and {1000 * hi:.0f} ms:\n" + format_table(rows), figs)

    @_tool("""Event-related desynchronisation/synchronisation (ERD/ERS): band power change from
           baseline, in %, per channel and condition (negative = ERD), with a time-course figure.
           Use for motor imagery or any task-induced rhythm change. Needs make_epochs.""",
           {"band": _range("Frequency band [low, high] in Hz (default [8, 13], mu/alpha)"),
            "baseline": _range("Baseline [start, end] in s (default: epoch start to 0)"),
            "window": _range("Activity window [start, end] in s (default: 0 to epoch end)"),
            "channels": _names("Channels for the time-course figure (default: 3 most responsive)")})
    def tool_compute_erd_ers(self, band: list | None = None, baseline: list | None = None,
                             window: list | None = None, channels: list | None = None) -> ToolOutput:
        ep = self._need_epochs()
        band = tuple(band or (8.0, 13.0))
        texts, top = [], []
        for cond in [c for c, code in ep.event_id.items() if np.any(ep.labels == code)]:
            rk = erd_ers(ep, band=band, baseline=tuple(baseline) if baseline else None,
                         window=tuple(window) if window else None, condition=cond)
            texts.append(f"Condition '{cond}' ({band[0]:g}-{band[1]:g} Hz, % change, most responsive first):\n"
                         + rk.summary(max_rows=12))
            top += [c for c in rk.top(3) if c not in top]
        chans = self._check_channels(channels) if channels else top[:3]
        base = tuple(baseline) if baseline else (ep.times[0], 0.0)
        fig = viz.plot_erds_timecourse(ep, chans, band=band, baseline=base)
        return ToolOutput("\n\n".join(texts), [self._figure(fig, "erd_ers")])

    # ------------------------------------------------------------ channel performance
    @_tool("""Rank channels by how well they perform: quality, snr, a band (delta/theta/alpha/beta/gamma
           power), connectivity (raw data); or erp (ERP signal-to-noise), erd, discriminability
           (Fisher score between conditions) and decoding (cross-validated LDA accuracy per channel)
           on epochs. Read-only, with a figure.""",
           {"by": _p("string", "Ranking criterion", enum=list(RAW_RANKINGS + EPOCH_RANKINGS))}, ("by",))
    def tool_rank_channels(self, by: str) -> ToolOutput:
        inst = self._need_epochs() if by in EPOCH_RANKINGS else self._need_raw()
        ranking = rank_channels(inst, by=by)
        return ToolOutput(ranking.summary(max_rows=25), [self._figure(ranking.plot(), f"rank_{by}")])

    @_tool("""Compute a feature table (time-domain: Hjorth, line length...; spectral: band powers,
           peak frequency...; complexity: entropies, fractal dimensions) per channel, or per epoch and
           channel, and save it as CSV for machine learning or statistics. Read-only. Complexity
           features take minutes on many epochs; leave them out unless needed.""",
           {"source": _p("string", "raw (one row per channel) or epochs (one row per epoch and channel)",
                         enum=["raw", "epochs"]),
            "include": _p("array", "Feature groups", items={"type": "string",
                                                             "enum": ["time", "spectral", "complexity"]}),
            "filename": _p("string", "CSV file name (default features.csv)")})
    def tool_extract_features(self, source: str = "raw", include: list | None = None,
                              filename: str = "features.csv") -> ToolOutput:
        inst = self._need_epochs() if source == "epochs" else self._need_raw()
        table = extract_features(inst, include=include or ["time", "spectral"])
        path = self._output_path(filename, (".csv",))
        table.to_csv(path)
        cols = table.columns
        preview = format_table(table.rows[:8], cols[:10])
        return ToolOutput(f"Saved {len(table)} rows x {len(cols)} columns to {path}\n"
                          f"Columns: {', '.join(cols)}\n\nFirst rows:\n{preview}")

    # ------------------------------------------------------------ figures and output
    @_tool("""Draw a figure for the user (and you): raw_traces (stacked channel traces), psd,
           spectrogram (one channel), band_topomaps, sensors (electrode layout), quality_topomap
           (needs assess_quality), ica_components (needs run_ica).""",
           {"kind": _p("string", "Figure type", enum=["raw_traces", "psd", "spectrogram", "band_topomaps",
                                                       "sensors", "quality_topomap", "ica_components"]),
            "start": _p("number", "raw_traces: start time in s (default 0)"),
            "duration": _p("number", "raw_traces: seconds shown (default 10)"),
            "channels": _names("raw_traces/psd: channels to show (default all); spectrogram: first one is used"),
            "fmax": _p("number", "psd/spectrogram: highest frequency (default 45)")},
           ("kind",))
    def tool_plot(self, kind: str, start: float = 0.0, duration: float = 10.0, channels: list | None = None,
                  fmax: float = 45.0) -> ToolOutput:
        raw = self._need_raw()
        picks = self._check_channels(channels) if channels else None
        if kind == "raw_traces":
            fig = viz.plot_raw(raw, start=start, duration=duration, picks=picks or "data", quality=self.quality)
        elif kind == "psd":
            fig = viz.plot_psd(raw, picks=picks or "eeg", fmax=fmax)
        elif kind == "spectrogram":
            fig = viz.plot_spectrogram(raw, (picks or raw.pick_names("eeg"))[0], fmax=fmax)
        elif kind == "band_topomaps":
            fig = viz.plot_band_topomaps(raw)
        elif kind == "sensors":
            fig = viz.plot_sensors(raw)
        elif kind == "quality_topomap":
            if self.quality is None:
                raise ToolError("Run assess_quality first.")
            fig = self.quality.plot_topomap()
        else:
            if self.ica is None:
                raise ToolError("Run run_ica first.")
            fig = self.ica.plot_components()
        return ToolOutput(f"Figure drawn: {kind}.", [self._figure(fig, kind)])

    @_tool("""Save the current (processed) data to a file in the output folder; the extension picks the
           format (.edf .bdf .vhdr .csv .npz .mat .h5 .fif).""",
           {"filename": _p("string", "File name, e.g. cleaned.edf")}, ("filename",))
    def tool_save_data(self, filename: str) -> ToolOutput:
        raw = self._need_raw()
        path = raw.save(self._output_path(filename, SAVE_FORMATS))
        return ToolOutput(f"Saved to {path}")

    @_tool("""Write a self-contained HTML report of the current data (quality, spectra, connectivity,
           events/ERPs) into the output folder.""",
           {"filename": _p("string", "File name (default report.html)"),
            "title": _p("string", "Report title")})
    def tool_generate_report(self, filename: str = "report.html", title: str | None = None) -> ToolOutput:
        raw = self._need_raw()
        path = generate_report(raw, self._output_path(filename, (".html",)), title=title, quality=self.quality)
        return ToolOutput(f"Report written to {path}")

    @_tool("Undo the last step that changed the data (filter, reference, ICA removal, bad channels ...).")
    def tool_undo_last_step(self) -> ToolOutput:
        if not self._undo:
            raise ToolError("Nothing to undo.")
        step, self.raw, self.ica, self.epochs, self.quality = self._undo.pop()
        self.steps.pop()
        return ToolOutput(f"Undid: {step}. Steps now: {' -> '.join(self.steps) or 'none'}")

    @_tool("""Run a short Python snippet when no other tool can do what the user needs. The user sees
           the code and must approve every run. Available names: raw (copy of the current data, a
           RawEEG), epochs, ica, np, plt, ep (the eegproc package), output_dir. Print what you need to
           see; new matplotlib figures are shown. To replace the working data, assign a RawEEG to
           new_raw.""",
           {"code": _p("string", "Python code"),
            "purpose": _p("string", "One sentence for the user on what the code does")},
           ("code", "purpose"))
    def tool_run_python(self, code: str, purpose: str) -> ToolOutput:
        question = f"The assistant wants to run this Python code ({purpose}):\n\n{code}\n\nRun it?"
        if not self.confirm(question):
            return ToolOutput("The user declined to run this code.")
        import eegproc

        ns = {"raw": self.raw.copy() if self.raw is not None else None, "epochs": self.epochs, "ica": self.ica,
              "np": np, "plt": plt, "ep": eegproc, "output_dir": str(self.output_dir)}
        before = set(plt.get_fignums())
        buf = io.StringIO()
        error = None
        with contextlib.redirect_stdout(buf):
            try:
                exec(compile(code, "<assistant>", "exec"), ns)
            except Exception:
                error = traceback.format_exc(limit=3)
        figs = [self._figure(plt.figure(n), "python") for n in plt.get_fignums() if n not in before]
        text = buf.getvalue() or "(no printed output)"
        if error:
            return ToolOutput(f"{text}\n{error}", figs, is_error=True)
        if isinstance(ns.get("new_raw"), RawEEG):
            text += "\n" + self._commit(ns["new_raw"], f"custom step: {purpose}")
        return ToolOutput(text, figs)
