"""Command-line interface: ``eegproc <command> ...`` (or ``python -m eegproc``)."""

from __future__ import annotations

import argparse
import json
import os
import sys

from . import __version__


def _read(args):
    from .io import read_raw

    kwargs = {}
    if getattr(args, "sfreq", None):
        kwargs["sfreq"] = args.sfreq
    raw = read_raw(args.file, fmt=getattr(args, "format", None), **kwargs)
    montage = getattr(args, "montage", None)
    if montage and montage.lower() != "none":
        raw.set_montage(montage)
    return raw


def _add_read_args(p):
    p.add_argument("file", help="EEG file (EDF, BDF, BrainVision, CSV/TXT, MAT, SET, NPZ, H5, FIF, ...)")
    p.add_argument("--sfreq", type=float, help="sampling rate for files that do not store it (CSV, NPY)")
    p.add_argument("--format", help="force the file format, e.g. edf or csv")
    p.add_argument("--montage", default="standard_1010",
                   help="electrode layout: standard_1010 (default), standard_1020, a positions file, or none")


def cmd_info(args) -> int:
    raw = _read(args)
    if args.json:
        print(json.dumps(raw.summary(), indent=2, default=str))
    else:
        print(raw.describe(channels=not args.brief))
    return 0


def cmd_formats(args) -> int:
    from .io import supported_formats
    from .utils import format_table

    print(format_table(supported_formats()))
    return 0


def cmd_convert(args) -> int:
    from .io import write_raw

    raw = _read(args)
    if args.picks:
        raw = raw.pick(args.picks.split(","))
    out = write_raw(raw, args.output, fmt=args.to)
    print(f"Wrote {out} ({raw.n_channels} channels, {raw.duration:.1f} s)")
    return 0


def cmd_quality(args) -> int:
    from .quality import assess_channel_quality

    raw = _read(args)
    line = args.line_freq if args.line_freq is not None else "auto"
    rep = assess_channel_quality(raw, line_freq=None if line == 0 else line)
    print(rep.summary())
    if args.csv:
        rep.to_csv(args.csv)
        print(f"\nTable written to {args.csv}")
    if args.json:
        rep.to_json(args.json)
        print(f"JSON written to {args.json}")
    if args.plot:
        import matplotlib

        matplotlib.use("Agg")
        rep.plot().savefig(args.plot, bbox_inches="tight", dpi=130)
        print(f"Figure written to {args.plot}")
    return 0


def cmd_rank(args) -> int:
    from .core.epochs import Epochs
    from .quality import rank_channels

    raw = _read(args)
    by = args.by.lower()
    if by in ("erp", "erd", "discriminability", "decoding"):
        if not args.keep_bads:
            from .quality import assess_channel_quality

            bads = assess_channel_quality(raw, use_prep=False).bad_channels
            if bads:
                raw.add_bads(bads)
                print(f"Excluding low-quality channels: {', '.join(bads)} (use --keep-bads to keep them)")
        uses_erp = by == "erp" or args.feature == "erp"
        l_freq, h_freq = args.l_freq, args.h_freq
        if l_freq is None and h_freq is None and uses_erp:
            l_freq, h_freq = 0.5, min(30.0, 0.45 * raw.sfreq)
            print(f"Band-pass {l_freq:g}-{h_freq:g} Hz for ERP analysis (override with --l-freq/--h-freq)")
        raw = raw.filter(l_freq, h_freq) if (l_freq or h_freq) else raw
        conds = args.conditions.split(",") if args.conditions else None
        ep = Epochs(raw, tmin=args.tmin, tmax=args.tmax, reject=args.reject,
                    baseline=(None, 0.0) if by == "erp" else None)
        if conds:
            ep = ep[conds]
        print(ep)
        kwargs = {}
        if by in ("discriminability", "decoding"):
            kwargs["feature"] = args.feature
            if args.window:
                kwargs["window"] = tuple(float(v) for v in args.window.split(","))
        if by == "erd" and args.window:
            kwargs["window"] = tuple(float(v) for v in args.window.split(","))
        ranking = rank_channels(ep, by, **kwargs)
    else:
        ranking = rank_channels(raw, by)
    print(ranking.summary(max_rows=args.top))
    if args.csv:
        ranking.to_csv(args.csv)
    if args.plot:
        import matplotlib

        matplotlib.use("Agg")
        ranking.plot().savefig(args.plot, bbox_inches="tight", dpi=130)
        print(f"Figure written to {args.plot}")
    return 0


def cmd_process(args) -> int:
    from .pipeline import Pipeline

    raw = _read(args)
    pipe = Pipeline.from_config(args.config)
    if args.output:
        pipe.output = args.output
    if args.report:
        pipe.report = args.report
    print(f"Running {pipe}")
    out = pipe.run(raw, verbose=True)
    print(out)
    if pipe.output:
        print(f"Saved {pipe.output}")
    if pipe.report:
        print(f"Report {pipe.report}")
    return 0


def cmd_report(args) -> int:
    import matplotlib

    matplotlib.use("Agg")
    from .report import generate_report

    raw = _read(args)
    if args.config:
        from .pipeline import Pipeline

        raw = Pipeline.from_config(args.config).run(raw, verbose=True)
    out = args.output or os.path.splitext(os.path.basename(args.file))[0] + "_report.html"
    generate_report(raw, out, epoch_window=(args.tmin, args.tmax), connectivity=not args.no_connectivity)
    print(f"Report written to {out}")
    return 0


def cmd_plot(args) -> int:
    import matplotlib

    matplotlib.use("Agg")
    from . import viz

    raw = _read(args)
    kind = args.kind
    if kind == "raw":
        fig = viz.plot_raw(raw, start=args.start, duration=args.duration)
    elif kind == "psd":
        fig = viz.plot_psd(raw, fmax=args.fmax)
    elif kind == "sensors":
        fig = viz.plot_sensors(raw)
    elif kind == "bands":
        fig = viz.plot_band_topomaps(raw)
    elif kind == "bandpower":
        fig = viz.plot_band_power(raw)
    elif kind == "spectrogram":
        fig = viz.plot_spectrogram(raw, args.channel or raw.pick_names("eeg")[0], fmax=args.fmax)
    elif kind == "quality":
        fig = raw.assess_quality().plot()
    elif kind == "connectivity":
        from .analysis import compute_connectivity

        fig = viz.plot_connectivity(compute_connectivity(raw, method="wpli", band=(8, 13)))
    else:
        raise ValueError(kind)
    fig.savefig(args.output, bbox_inches="tight", dpi=130)
    print(f"Figure written to {args.output}")
    return 0


def cmd_features(args) -> int:
    from .analysis.features import extract_features

    raw = _read(args)
    include = args.include.split(",")
    table = extract_features(raw, include=include)
    print(table)
    if args.output:
        table.to_csv(args.output)
        print(f"Features written to {args.output}")
    return 0


def cmd_simulate(args) -> int:
    from .datasets import simulate_eeg

    bads = "default" if args.bad_channels else None
    raw = simulate_eeg(args.duration, args.sfreq, args.montage_name, task=args.task, bad_channels=bads,
                       eog=True, seed=args.seed, stim_channel=args.stim)
    raw.save(args.output)
    print(f"Simulated {raw} -> {args.output}")
    if bads:
        print(f"Ground-truth bad channels: {raw.meta['ground_truth_bads']}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="eegproc", description="EEG signal processing toolkit")
    parser.add_argument("--version", action="version", version=f"eegproc {__version__}")
    parser.add_argument("-v", "--verbose", action="store_true", help="show log messages")
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("info", help="show file contents: channels, types, rate, events")
    _add_read_args(p)
    p.add_argument("--json", action="store_true", help="machine-readable output")
    p.add_argument("--brief", action="store_true", help="omit the channel table")
    p.set_defaults(func=cmd_info)

    p = sub.add_parser("formats", help="list supported file formats")
    p.set_defaults(func=cmd_formats)

    p = sub.add_parser("convert", help="convert between formats")
    _add_read_args(p)
    p.add_argument("output", help="output file; format from extension (.edf .bdf .vhdr .csv .npz .mat .h5 .fif)")
    p.add_argument("--to", help="force the output format")
    p.add_argument("--picks", help="comma-separated channels or types to keep")
    p.set_defaults(func=cmd_convert)

    p = sub.add_parser("quality", help="score every channel's signal quality (which channels perform)")
    _add_read_args(p)
    p.add_argument("--line-freq", type=float, help="50 or 60 Hz (default: auto-detect; 0 = none)")
    p.add_argument("--csv", help="save the table as CSV")
    p.add_argument("--json", help="save the full report as JSON")
    p.add_argument("--plot", help="save the dashboard figure (PNG/PDF/SVG)")
    p.set_defaults(func=cmd_quality)

    p = sub.add_parser("rank", help="rank channels by quality, band power, ERP, ERD, discriminability, decoding")
    _add_read_args(p)
    p.add_argument("--by", default="quality",
                   help="quality | snr | delta | theta | alpha | beta | gamma | band:LO-HI | connectivity | "
                        "erp | erd | discriminability | decoding")
    p.add_argument("--tmin", type=float, default=-0.2)
    p.add_argument("--tmax", type=float, default=0.8)
    p.add_argument("--reject", type=float, default=None, help="peak-to-peak rejection threshold (µV)")
    p.add_argument("--conditions", help="comma-separated event names to compare")
    p.add_argument("--feature", default="log_bandpower", help="log_bandpower | erp (for discriminability/decoding)")
    p.add_argument("--window", help="analysis window 'start,stop' in seconds")
    p.add_argument("--l-freq", type=float, default=None)
    p.add_argument("--h-freq", type=float, default=None)
    p.add_argument("--top", type=int, default=20)
    p.add_argument("--keep-bads", action="store_true",
                   help="do not exclude low-quality channels from event-related rankings")
    p.add_argument("--csv")
    p.add_argument("--plot")
    p.set_defaults(func=cmd_rank)

    p = sub.add_parser("process", help="run a JSON/YAML preprocessing pipeline")
    _add_read_args(p)
    p.add_argument("--config", required=True, help="pipeline description (.json/.yaml)")
    p.add_argument("-o", "--output", help="save the processed data")
    p.add_argument("--report", help="also write an HTML report")
    p.set_defaults(func=cmd_process)

    p = sub.add_parser("report", help="write a self-contained HTML report")
    _add_read_args(p)
    p.add_argument("-o", "--output")
    p.add_argument("--config", help="preprocess with this pipeline first")
    p.add_argument("--tmin", type=float, default=-0.2, help="epoch start for event sections")
    p.add_argument("--tmax", type=float, default=0.8, help="epoch end for event sections")
    p.add_argument("--no-connectivity", action="store_true")
    p.set_defaults(func=cmd_report)

    p = sub.add_parser("plot", help="save a figure")
    _add_read_args(p)
    p.add_argument("--kind", default="raw",
                   choices=["raw", "psd", "sensors", "bands", "bandpower", "spectrogram", "quality", "connectivity"])
    p.add_argument("-o", "--output", required=True)
    p.add_argument("--start", type=float, default=0.0)
    p.add_argument("--duration", type=float, default=10.0)
    p.add_argument("--fmax", type=float, default=60.0)
    p.add_argument("--channel")
    p.set_defaults(func=cmd_plot)

    p = sub.add_parser("features", help="per-channel features to CSV")
    _add_read_args(p)
    p.add_argument("-o", "--output")
    p.add_argument("--include", default="time,spectral", help="time,spectral,complexity")
    p.set_defaults(func=cmd_features)

    p = sub.add_parser("simulate", help="write a realistic synthetic recording (for trying things out)")
    p.add_argument("output", help="output file (.edf, .bdf, .vhdr, .csv, .npz, ...)")
    p.add_argument("--duration", type=float, default=120.0)
    p.add_argument("--sfreq", type=float, default=256.0)
    p.add_argument("--montage-name", default="clinical19", help="clinical19 | standard_1020 | 32 | 64")
    p.add_argument("--task", choices=["oddball", "motor_imagery"], default=None)
    p.add_argument("--bad-channels", action="store_true", help="inject some bad channels")
    p.add_argument("--stim", action="store_true", help="add a trigger channel")
    p.add_argument("--seed", type=int, default=0)
    p.set_defaults(func=cmd_simulate)
    return parser


def main(argv=None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.verbose:
        from .utils import set_log_level

        set_log_level("INFO")
    try:
        return int(args.func(args) or 0)
    except BrokenPipeError:  # output piped into e.g. `head`
        return 0
    except (FileNotFoundError, ValueError, KeyError, ImportError) as err:
        print(f"eegproc: error: {err}", file=sys.stderr)
        return 1


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
