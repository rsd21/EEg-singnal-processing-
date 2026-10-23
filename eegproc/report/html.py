"""Self-contained HTML report: file overview, channel quality, spectra, events.

All figures are embedded as PNG data URIs, so the report is a single file
that can be e-mailed or archived.
"""

from __future__ import annotations

import base64
import datetime as _dt
import html
import io
import os
from typing import Any

import numpy as np

from ..utils import logger


def _fig_to_uri(fig) -> str:
    import matplotlib.pyplot as plt

    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=110, bbox_inches="tight")
    plt.close(fig)
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode("ascii")


def _esc(x: Any) -> str:
    return html.escape(str(x))


def _fmt(v, digits: int = 3) -> str:
    if v is None:
        return "–"
    if isinstance(v, (float, np.floating)):
        if not np.isfinite(v):
            return "–"
        return f"{v:.{digits}g}"
    if isinstance(v, (list, tuple)):
        return ", ".join(_fmt(i) for i in v) or "–"
    return str(v)


def _table(rows: list[dict], columns: list[str], headers: list[str] | None = None,
           grade_col: str | None = None) -> str:
    headers = headers or columns
    out = ["<div class='table-wrap'><table><thead><tr>"]
    out += [f"<th>{_esc(h)}</th>" for h in headers]
    out.append("</tr></thead><tbody>")
    for r in rows:
        out.append("<tr>")
        for c in columns:
            v = r.get(c)
            if c == grade_col:
                out.append(f"<td><span class='grade {v}'>{_esc(v)}</span></td>")
            else:
                cls = " class='num'" if isinstance(v, (int, float, np.floating, np.integer)) and not isinstance(v, bool) else ""
                out.append(f"<td{cls}>{_esc(_fmt(v))}</td>")
        out.append("</tr>")
    out.append("</tbody></table></div>")
    return "".join(out)


def _section(title: str, body: str, anchor: str, intro: str = "") -> str:
    intro_html = f"<p class='intro'>{intro}</p>" if intro else ""
    return f"<section id='{anchor}'><h2>{_esc(title)}</h2>{intro_html}{body}</section>"


def _img(uri: str, alt: str) -> str:
    return f"<figure><img src='{uri}' alt='{_esc(alt)}'></figure>"


def _recommendations(raw, rep, blinks_per_min: float | None) -> list[str]:
    recs = []
    bads = rep.bad_channels
    if bads:
        can_interp = all(b in raw.positions for b in bads)
        them = "it" if len(bads) == 1 else "them"
        recs.append(f"Mark <b>{', '.join(bads)}</b> as bad" +
                    (f" and repair {them} with spherical-spline interpolation (<code>raw.interpolate_bads()</code>)."
                     if can_interp else " (set a montage to allow interpolation)."))
    lf = rep.info.get("line_freq")
    if lf:
        worst = max(rep.rows, key=lambda r: r["line_noise_db"])
        recs.append(f"Power-line interference at {lf:g} Hz (up to +{worst['line_noise_db']:.0f} dB on "
                    f"{worst['channel']}): apply a notch filter (<code>raw.notch_filter({lf:g})</code>).")
    drift = [r["channel"] for r in rep.rows if r["penalties"].get("drift", 0) > 3]
    if drift or float(np.nanmedian(rep.metric("drift_db"))) > 6:
        recs.append("Slow drifts present: high-pass filter at 0.5-1 Hz before analysis "
                    "(<code>raw.filter(1.0, None)</code>)." + (f" Worst: {', '.join(drift[:5])}." if drift else ""))
    if blinks_per_min and blinks_per_min > 2:
        recs.append(f"About {blinks_per_min:.0f} eye blinks per minute detected: remove them with ICA "
                    "(<code>remove_artifacts_ica(raw)</code>) or reject affected epochs.")
    emg = [r["channel"] for r in rep.rows if r["penalties"].get("hf_noise", 0) > 5 and r["channel"] not in bads]
    if emg:
        recs.append(f"High-frequency (muscle or electrode) noise on {', '.join(emg[:6])}: check electrode "
                    "impedance, or low-pass at 30-40 Hz if gamma is not of interest.")
    if not recs:
        recs.append("No major problems found. The recording is ready for analysis.")
    return recs


def generate_report(raw, path: str | os.PathLike, title: str | None = None, quality=None,
                    raw_window: tuple[float, float] | None = None, connectivity: bool = True,
                    events: bool = True, epoch_window: tuple[float, float] = (-0.2, 0.8),
                    max_channels_plot: int = 40) -> str:
    """Build an HTML report for a recording and write it to ``path``.

    Parameters
    ----------
    raw : RawEEG
    quality : ChannelQualityReport, optional (computed if omitted)
    raw_window : ``(start, duration)`` of the trace snapshot (default: first 10 s)
    connectivity : include an alpha-band wPLI network section
    events : include an event/ERP section when events are found
    epoch_window : ``(tmin, tmax)`` for event-related sections
    """
    import matplotlib

    matplotlib.use("Agg", force=False)
    from ..analysis.connectivity import compute_connectivity
    from ..analysis.spectral import BANDS
    from ..core.epochs import Epochs, find_events
    from ..preprocessing.artifacts import find_blinks
    from ..quality.channel_quality import assess_channel_quality
    from ..quality.ranking import (band_activity, decoding_accuracy, discriminability, dominant_bands, erd_ers,
                                   erp_snr)
    from ..viz import (plot_band_topomaps, plot_connectivity, plot_erds_timecourse, plot_evoked,
                       plot_evoked_topomaps, plot_psd, plot_quality_report, plot_ranking, plot_raw)

    path = os.fspath(path)
    rep = quality or assess_channel_quality(raw)
    eeg = raw.pick_names("eeg")
    name = os.path.basename(raw.filename) if raw.filename else "in-memory recording"
    title = title or f"EEG report · {name}"
    sections: list[str] = []
    nav: list[tuple[str, str]] = []

    # --------------------------------------------------------------- overview
    s = raw.summary()
    mins, secs = divmod(raw.duration, 60)
    blinks_per_min = None
    try:
        if raw.duration >= 20 and (raw.pick_names("eog") or any(c in raw.ch_names for c in ("Fp1", "Fp2"))):
            blinks_per_min = len(find_blinks(raw)) / (raw.duration / 60)
    except Exception:  # noqa: BLE001 - blink detection is optional context
        blinks_per_min = None
    best = rep.ranked()[0]
    worst = rep.ranked()[-1]
    counts = {g: len(v) for g, v in rep.by_grade().items()}
    tiles = [
        ("Median quality", f"{rep.overall_score:.0f}<small>/100</small>", "across EEG channels"),
        ("Good channels", f"{counts['good']}<small>/{len(rep.rows)}</small>",
         f"{counts['fair']} fair · {counts['poor']} poor · {counts['bad']} bad"),
        ("Best channel", _esc(best["channel"]), f"score {best['score']:.0f}"),
        ("Worst channel", _esc(worst["channel"]), _esc(worst["reasons"][0] if worst["reasons"] else f"score {worst['score']:.0f}")),
    ]
    tile_html = "<div class='tiles'>" + "".join(
        f"<div class='tile'><div class='label'>{a}</div><div class='value'>{b}</div><div class='sub'>{c}</div></div>"
        for a, b, c in tiles) + "</div>"
    info_rows = [
        ("File", name), ("Format", s["format"] or "–"), ("Recorded", s["meas_date"] or "–"),
        ("Duration", f"{int(mins)} min {secs:.1f} s ({raw.n_times:,} samples)"),
        ("Sampling rate", f"{raw.sfreq:g} Hz"),
        ("Channels", ", ".join(f"{v} {k.upper()}" for k, v in s["channel_types"].items())),
        ("Electrode positions", f"{s['has_positions']} of {len(eeg)} EEG channels"),
        ("Annotations", ", ".join(f"{k} ×{v}" for k, v in list(s["annotations"].items())[:10]) or "none"),
        ("Line noise", f"{rep.info.get('line_freq'):g} Hz" if rep.info.get("line_freq") else "not detected"),
        ("Processing history", " → ".join(raw.history) or "none"),
    ]
    info_html = "<dl class='info'>" + "".join(f"<dt>{_esc(k)}</dt><dd>{_esc(v)}</dd>" for k, v in info_rows) + "</dl>"
    recs = _recommendations(raw, rep, blinks_per_min)
    rec_html = "<div class='callout'><h3>Recommendations</h3><ul>" + "".join(f"<li>{r}</li>" for r in recs) + "</ul></div>"
    sections.append(_section("Overview", tile_html + rec_html + info_html, "overview"))
    nav.append(("overview", "Overview"))

    # ---------------------------------------------------------------- quality
    q_rows = [{"rank": r["rank"], "channel": r["channel"], "score": r["score"], "grade": r["grade"],
               "issues": "; ".join(r["reasons"]) or "–", "prep": ", ".join(r["prep_flags"]) or "–",
               "amp": r["amplitude_uv"], "corr": r["neighbor_corr"], "snr": r["snr_db"],
               "alpha": r["alpha_peak_hz"] if r["alpha_peak_db"] > 3 else None}
              for r in rep.ranked()]
    body = _img(_fig_to_uri(plot_quality_report(rep)), "channel quality dashboard")
    body += _table(q_rows, ["rank", "channel", "score", "grade", "issues", "prep", "amp", "corr", "snr", "alpha"],
                   ["#", "Channel", "Score", "Grade", "Issues", "PREP flags", "Amplitude µV", "Neighbour r",
                    "SNR dB", "Alpha peak Hz"], grade_col="grade")
    sections.append(_section("Channel quality", body, "quality",
                             "Each channel starts at 100 points and loses points per problem found "
                             "(flat, clipping, abnormal amplitude, poor correlation with neighbours, noise, "
                             "drift, artifacts, pops). Grades: good ≥ 75, fair ≥ 50, poor ≥ 25, bad &lt; 25."))
    nav.append(("quality", "Channel quality"))

    # ------------------------------------------------------------------ traces
    start, dur = raw_window or (0.0, min(10.0, raw.duration))
    fig = plot_raw(raw, start=start, duration=dur, quality=rep, picks=raw.pick_indices("data")[:max_channels_plot],
                   display_filter=(0.5, None) if raw.sfreq > 2 else None)
    sections.append(_section("Signal snapshot", _img(_fig_to_uri(fig), "raw traces"), "traces",
                             f"{start:g}-{start + dur:g} s, high-passed at 0.5 Hz for display. "
                             "Traces are coloured by quality grade; the number after each name is its score."))
    nav.append(("traces", "Signals"))

    # --------------------------------------------------------------- spectra
    analysed = raw.copy()
    analysed.set_bads(sorted(set(analysed.bads) | set(rep.bad_channels)))
    body = _img(_fig_to_uri(plot_psd(analysed, fmax=min(60.0, raw.sfreq / 2))), "power spectra")
    good_eeg = [c for c in eeg if c not in analysed.bads]
    if len(good_eeg) >= 3 and (len([c for c in good_eeg if c in raw.positions]) >= 3):
        body += _img(_fig_to_uri(plot_band_topomaps(analysed)), "band topomaps")
    if good_eeg:
        acts = band_activity(analysed)
        lead = [{"band": b, "range": f"{BANDS[b][0]:g}-{BANDS[b][1]:g} Hz", "top channels": ", ".join(r.top(3)),
                 "max relative power": float(np.nanmax(r.values)), "median": float(np.nanmedian(r.values))}
                for b, r in acts.items()]
        body += "<h3>Which channel leads each rhythm</h3>"
        body += _table(lead, ["band", "range", "top channels", "max relative power", "median"],
                       ["Band", "Range", "Top channels", "Max relative power", "Median across channels"])
        dom = dominant_bands(analysed, picks=good_eeg)
        body += "<h3>Dominant band per channel</h3>"
        body += _table(dom, ["channel", "dominant_band", "dominant_fraction", "peak_hz"],
                       ["Channel", "Dominant band", "Fraction of power", "Spectral peak Hz"])
    sections.append(_section("Spectral activity", body, "spectra",
                             "Channels marked bad are drawn in red and left out of the maps and tables."))
    nav.append(("spectra", "Spectra"))

    # ---------------------------------------------------------- connectivity
    if connectivity and len(good_eeg) >= 4 and raw.duration >= 10 and raw.sfreq >= 30:
        try:
            con = compute_connectivity(analysed, method="wpli", band=(8.0, 13.0))
            body = _img(_fig_to_uri(plot_connectivity(con)), "alpha wPLI connectivity")
            hubs = sorted(zip(con.ch_names, con.node_strength()), key=lambda t: -t[1])[:8]
            body += _table([{"channel": c, "strength": float(v)} for c, v in hubs], ["channel", "strength"],
                           ["Hub channel", "Mean wPLI"])
            sections.append(_section("Alpha-band connectivity", body, "connectivity",
                                     "Weighted phase-lag index (8-13 Hz), which ignores zero-lag coupling "
                                     "caused by volume conduction. Large dots are hubs."))
            nav.append(("connectivity", "Connectivity"))
        except Exception as err:  # noqa: BLE001 - optional section
            logger.warning("Connectivity section skipped: %s", err)

    # ----------------------------------------------------------------- events
    if events:
        try:
            ev, event_id = find_events(analysed)
        except Exception:  # noqa: BLE001
            ev, event_id = np.zeros((0, 3), int), {}
        if len(ev) >= 4 and event_id:
            body = ""
            ep = Epochs(analysed, ev, event_id, tmin=epoch_window[0], tmax=epoch_window[1],
                        picks=[c for c in eeg if c not in analysed.bads], reject=250.0)
            counts_rows = [{"condition": k, "code": v, "epochs": int((ep.labels == v).sum())}
                           for k, v in event_id.items()]
            body += _table(counts_rows, ["condition", "code", "epochs"], ["Condition", "Code", "Epochs kept"])
            oscillatory = (ep.times[-1] - ep.times[0]) > 1.5
            conds = [k for k, v in event_id.items() if (ep.labels == v).sum() >= 5]
            if ep.n_epochs >= 4 and not oscillatory:
                evk = ep.average()
                snr = erp_snr(ep, window=(max(0.0, ep.times[0]), ep.times[-1]))
                body += _img(_fig_to_uri(plot_evoked(evk, picks=snr.top(3))), "evoked response")
                if len([c for c in ep.ch_names if c in raw.positions]) >= 3:
                    body += _img(_fig_to_uri(plot_evoked_topomaps(evk)), "evoked topomaps")
                body += "<h3>Channels with the clearest event-related response</h3>"
                body += _img(_fig_to_uri(plot_ranking(snr)), "ERP SNR ranking")
            elif ep.n_epochs >= 4:
                base = (ep.times[0], 0.0) if ep.times[0] < -0.2 else (ep.times[0], ep.times[0] + 0.2 * (ep.times[-1] - ep.times[0]))
                win = (max(0.5, base[1] + 0.1), ep.times[-1])
                tops: list[str] = []
                for cond in conds or [None]:
                    r = erd_ers(ep, band=(8.0, 13.0), baseline=base, window=win, condition=cond)
                    body += f"<h3>Mu/alpha (8-13 Hz) power change{' · ' + _esc(cond) if cond else ''}</h3>"
                    body += _img(_fig_to_uri(plot_ranking(r)), "ERD/ERS ranking")
                    tops += [c for c in r.top(2) if c not in tops]
                body += "<h3>Band-power time course of the most responsive channels</h3>"
                body += _img(_fig_to_uri(plot_erds_timecourse(ep, tops[:3], baseline=base)), "ERD/ERS time course")
            if ep.n_epochs >= 4 and len(conds) >= 2:
                sub = ep[conds]
                feature = "log_bandpower" if oscillatory else "erp"
                win = (0.1, min(0.6, ep.times[-1])) if feature == "erp" else (max(0.0, ep.times[0]), ep.times[-1])
                disc = discriminability(sub, feature=feature, window=win)
                dec = decoding_accuracy(sub, feature=feature, window=win, n_repeats=2)
                body += f"<h3>Channels that best separate {', '.join(map(_esc, conds))}</h3>"
                body += _img(_fig_to_uri(plot_ranking(dec)), "decoding ranking")
                rows = [{"channel": c, "accuracy": dec.value(c), "p": dec.details["p_value"][dec.ch_names.index(c)],
                         "fisher": disc.value(c), "auc": disc.details["auc"][disc.ch_names.index(c)]}
                        for c in dec.top(10)]
                body += _table(rows, ["channel", "accuracy", "p", "fisher", "auc"],
                               ["Channel", "CV accuracy", "p (binomial)", "Fisher score", "AUC"])
                body += (f"<p class='note'>All channels together: {dec.details['all_channels_accuracy']:.2f} "
                         f"balanced accuracy (chance {dec.details['chance_level']:.2f}). Features: "
                         f"{'log band power (mu, beta)' if oscillatory else 'mean amplitude in 5 time bins'}.</p>")
            sections.append(_section("Events and responses", body, "events",
                                     f"{len(ev)} events; epochs {epoch_window[0]:g} to {epoch_window[1]:g} s, "
                                     "peak-to-peak rejection at 250 µV, bad channels excluded."))
            nav.append(("events", "Events"))

    # ---------------------------------------------------------------- methods
    methods = """
    <ul>
      <li><b>Quality score</b>: 100 minus penalties for flat signal, clipping, abnormal robust amplitude,
      low correlation with spatial neighbours, high-frequency noise / low SNR (1-30 Hz vs &gt;45 Hz),
      line noise, slow drift, channel-specific artifacts, pops (kurtosis of the channel minus its
      neighbours), unstable amplitude and noise-like spectra (1/f exponent).</li>
      <li><b>PREP flags</b>: deviation, correlation, high-frequency noise, dropout and RANSAC criteria of
      the PREP pipeline (Bigdely-Shamlo et al., 2015).</li>
      <li><b>Spectra</b>: Welch's method, 2-4 s Hann windows, 50 % overlap, median averaging.</li>
      <li><b>ERP SNR</b>: evoked power over noise power estimated from random sign-flipped averages.</li>
      <li><b>Decoding</b>: shrinkage-LDA per channel, stratified cross-validation, balanced accuracy,
      binomial test against chance.</li>
    </ul>"""
    sections.append(_section("Methods", methods, "methods"))
    nav.append(("methods", "Methods"))

    generated = _dt.datetime.now().strftime("%Y-%m-%d %H:%M")
    nav_html = "".join(f"<a href='#{a}'>{_esc(t)}</a>" for a, t in nav)
    doc = f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{_esc(title)}</title>
<style>{_CSS}</style></head>
<body>
<header><div class="wrap"><h1>{_esc(title)}</h1>
<p class="meta">Generated by eegproc · {generated}</p><nav>{nav_html}</nav></div></header>
<main class="wrap">{''.join(sections)}</main>
</body></html>"""
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write(doc)
    return path


_CSS = """
:root { --surface:#fcfcfb; --page:#f9f9f7; --ink:#0b0b0b; --ink2:#52514e; --muted:#898781;
  --grid:#e1e0d9; --axis:#c3c2b7; --accent:#2a78d6; --good:#0ca30c; --fair:#fab219; --poor:#ec835a;
  --bad:#d03b3b; --card:#ffffff; }
@media (prefers-color-scheme: dark) { :root:not([data-theme="light"]) { --surface:#1a1a19; --page:#0d0d0d;
  --ink:#ffffff; --ink2:#c3c2b7; --muted:#898781; --grid:#2c2c2a; --axis:#383835; --accent:#3987e5;
  --card:#1a1a19; } :root:not([data-theme="light"]) figure img { background:#fcfcfb; } }
* { box-sizing:border-box; }
body { margin:0; background:var(--page); color:var(--ink); font:15px/1.55 system-ui,-apple-system,"Segoe UI",sans-serif; }
.wrap { max-width:1200px; margin:0 auto; padding:0 16px; }
header { background:var(--surface); border-bottom:1px solid var(--grid); padding:20px 0 0; }
@media (min-width:900px) { header { position:sticky; top:0; z-index:2; } section { scroll-margin-top:140px; } }
h1 { font-size:22px; margin:0; }
.meta { color:var(--muted); margin:2px 0 10px; font-size:13px; }
nav { display:flex; gap:18px; overflow-x:auto; }
nav a { color:var(--ink2); text-decoration:none; padding:8px 0; border-bottom:2px solid transparent; white-space:nowrap; font-size:14px; }
nav a:hover { border-color:var(--accent); color:var(--ink); }
section { background:var(--card); border:1px solid var(--grid); border-radius:10px; padding:20px 22px; margin:20px 0; }
h2 { font-size:18px; margin:0 0 6px; } h3 { font-size:15px; margin:22px 0 8px; }
.intro, .note { color:var(--ink2); margin:0 0 14px; font-size:14px; }
figure { margin:12px 0; } figure img { max-width:100%; height:auto; border-radius:6px; display:block; }
.tiles { display:grid; grid-template-columns:repeat(auto-fit,minmax(200px,1fr)); gap:12px; margin:8px 0 16px; }
.tile { border:1px solid var(--grid); border-radius:8px; padding:12px 14px; background:var(--surface); }
.tile .label { color:var(--muted); font-size:12px; text-transform:uppercase; letter-spacing:.04em; }
.tile .value { font-size:28px; font-weight:600; margin:2px 0; } .tile .value small { font-size:15px; color:var(--muted); font-weight:400; }
.tile .sub { color:var(--ink2); font-size:13px; }
.callout { border-left:3px solid var(--accent); background:var(--surface); padding:10px 16px; border-radius:0 8px 8px 0; margin:0 0 16px; }
.callout h3 { margin:4px 0 6px; } .callout ul { margin:0; padding-left:20px; } .callout li { margin:4px 0; }
code { background:var(--page); border:1px solid var(--grid); border-radius:4px; padding:0 4px; font-size:13px; }
dl.info { display:grid; grid-template-columns:max-content 1fr; gap:6px 18px; margin:0; font-size:14px; }
dl.info dt { color:var(--muted); } dl.info dd { margin:0; overflow-wrap:anywhere; }
.table-wrap { overflow-x:auto; margin:10px 0; }
table { border-collapse:collapse; width:100%; font-size:13px; }
th { text-align:left; color:var(--muted); font-weight:500; border-bottom:1px solid var(--axis); padding:6px 10px; white-space:nowrap; }
td { border-bottom:1px solid var(--grid); padding:6px 10px; vertical-align:top; }
td.num { text-align:right; font-variant-numeric:tabular-nums; white-space:nowrap; }
.grade::before { content:""; display:inline-block; width:9px; height:9px; border-radius:50%; margin-right:6px; }
.grade.good::before { background:var(--good); } .grade.fair::before { background:var(--fair); }
.grade.poor::before { background:var(--poor); } .grade.bad::before { background:var(--bad); }
@media (max-width:640px) { dl.info { grid-template-columns:1fr; } section { padding:16px 14px; } }
"""
