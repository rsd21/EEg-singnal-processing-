"""Results on real EEG recordings from OpenNeuro (public, CC0).

Runs four analyses with eegproc and writes the figures to docs/images/results/
and the numbers to docs/results.md:

1. Eyes closed vs eyes open: posterior alpha power and blink rate
2. Channel quality of real recordings
3. Executed and imagined left/right fist movement: mu/beta ERD and decoding
4. Auditory oddball: the P300

The data (about 1.7 GB) are downloaded once into ~/eegproc_data (or $EEGPROC_DATA).

    python examples/real_data_openneuro.py                 # 20 + 13 subjects, about 10 minutes
    python examples/real_data_openneuro.py --subjects 3    # quick check
"""

from __future__ import annotations

import argparse
import re
from collections import Counter
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from scipy import stats  # noqa: E402
from scipy.ndimage import uniform_filter1d  # noqa: E402

import eegproc as ep  # noqa: E402
from eegproc.datasets import fetch_openneuro  # noqa: E402
from eegproc.preprocessing import filter_data, find_blinks, laplacian  # noqa: E402
from eegproc.quality import erd_ers  # noqa: E402
from eegproc.quality.ranking import cross_val_accuracy, epoch_features  # noqa: E402
from eegproc.viz import _style as S  # noqa: E402
from eegproc.viz import plot_topomap  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
FIG_DIR = ROOT / "docs" / "images" / "results"
DOC = ROOT / "docs" / "results.md"
MMI, ODDBALL = "ds004362", "ds003061"
POSTERIOR = ["O1", "Oz", "O2", "PO7", "PO3", "POz", "PO4", "PO8"]
OPEN_C, CLOSED_C, THIRD_C = S.CATEGORICAL[0], S.CATEGORICAL[1], S.CATEGORICAL[2]


def mmi_file(sub: str, run: int) -> Path:
    return fetch_openneuro(MMI, f"{sub}/eeg/{sub}_task-motion_run-{run}_eeg.set")[0]


def p_text(p: float) -> str:
    return f"{p:.2g}" if p >= 0.001 else f"{p:.1e}".replace("e-0", "e-")


def sem(x: np.ndarray, axis: int = 0) -> np.ndarray:
    return x.std(axis=axis, ddof=1) / np.sqrt(x.shape[axis])


def save(fig, name: str) -> str:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG_DIR / name, dpi=130, bbox_inches="tight")
    plt.close(fig)
    return f"images/results/{name}"


# ---------------------------------------------------------------- 1 + 2: resting baselines
def baseline_analysis(n_sub: int) -> dict:
    rows, psd_open, psd_closed, ratio_maps, flagged, reasons = [], [], [], [], Counter(), Counter()
    example = None
    for s in range(1, n_sub + 1):
        sub = f"sub-{s:03d}"
        row = {"subject": sub}
        for cond, run in (("open", 1), ("closed", 2)):
            raw = ep.read_raw(mmi_file(sub, run))
            if cond == "open":
                quality = raw.assess_quality()
                row["quality"] = float(np.median(quality.scores))
                row["flagged"] = quality.bad_channels
                row["hf_noise"] = sum(any("high-frequency noise" in x for x in r["reasons"]) for r in quality.ranked())
                row["line_noise"] = sum(any("line noise" in x for x in r["reasons"]) for r in quality.ranked())
                flagged.update(quality.bad_channels)
                for r in quality.ranked():
                    if r["suggest_bad"] and r["reasons"]:
                        reasons[re.sub(r"\s*\(.*|\s+in \d+% .*", "", r["reasons"][0])] += 1
                if s == 1:
                    save(quality.plot(), "quality_dashboard_sub-001.png")
            blinks = find_blinks(raw)  # this dataset has no EOG channel: Fp1 is used
            row[f"blinks_{cond}"] = len(blinks) / (raw.duration / 60)
            if s == 1:
                example = example or {}
                example[cond] = (raw.times, filter_data(raw["Fp1"], raw.sfreq, 0.5, 15.0), blinks)
            clean = raw.filter(1.0, 45.0).set_reference("average")
            spec = ep.compute_psd(clean, picks="eeg", fmin=1.0, fmax=45.0)
            alpha = spec.band_power({"alpha": (8.0, 13.0)})["alpha"]
            post = [spec.ch_names.index(c) for c in POSTERIOR]
            row[f"alpha_{cond}"] = float(alpha[post].mean())
            row[f"paf_{cond}"] = float(np.nanmedian(spec.peak_frequency()[post]))
            (psd_open if cond == "open" else psd_closed).append(10 * np.log10(spec.data[post].mean(axis=0)))
            row[f"alpha_map_{cond}"] = alpha
        ratio_maps.append(10 * np.log10(row.pop("alpha_map_closed") / row.pop("alpha_map_open")))
        rows.append(row)
        print(f"  {sub}: alpha x{row['alpha_closed'] / row['alpha_open']:.1f}, "
              f"blinks {row['blinks_open']:.0f} -> {row['blinks_closed']:.0f} /min")
    return {"rows": rows, "freqs": spec.freqs, "psd_open": np.array(psd_open), "psd_closed": np.array(psd_closed),
            "ratio_map": np.mean(ratio_maps, axis=0), "ch_names": spec.ch_names, "positions": clean.positions,
            "flagged": flagged, "reasons": reasons, "example": example}


def fig_alpha(b: dict) -> str:
    f = b["freqs"]
    keep = f <= 30
    with S.style():
        fig, (ax, ax2) = plt.subplots(1, 2, figsize=(11, 4.2), layout="constrained",
                                      gridspec_kw={"width_ratios": [1.6, 1]})
        ax.axvspan(8, 13, color=S.GRID, alpha=0.6, lw=0)
        ax.text(10.5, 0.98, "alpha", transform=ax.get_xaxis_transform(), ha="center", va="top",
                color=S.MUTED, fontsize=8)
        for data, color, label in ((b["psd_open"], OPEN_C, "eyes open"), (b["psd_closed"], CLOSED_C, "eyes closed")):
            m, e = data.mean(axis=0)[keep], sem(data)[keep]
            ax.fill_between(f[keep], m - e, m + e, color=color, alpha=0.18, lw=0)
            ax.plot(f[keep], m, color=color, lw=2, label=label)
        ax.set(xlim=(1, 30), xlabel="Frequency (Hz)", ylabel="Power (dB re 1 µV²/Hz)")
        ax.set_title(f"Posterior EEG spectrum, mean of {len(b['rows'])} subjects (±SEM)", loc="left")
        ax.legend(loc="upper right")
        S.grid_y(ax)
        plot_topomap(b["ratio_map"], b["ch_names"], b["positions"], ax=ax2, cmap=S.CMAP_SEQ,
                     cbar_label="dB (closed vs open)", contours=5)
        ax2.set_title("Alpha increase with eyes closed", loc="left")
    return save(fig, "alpha_eyes_open_closed.png")


def fig_blinks(b: dict) -> str:
    rows = b["rows"]
    with S.style():
        fig = plt.figure(figsize=(11, 4.2), layout="constrained")
        grid = fig.add_gridspec(2, 2, width_ratios=[2, 1])
        lim = max(np.abs(b["example"][c][1][b["example"][c][0] < 20]).max() for c in ("open", "closed"))
        for i, (cond, color) in enumerate((("open", OPEN_C), ("closed", CLOSED_C))):
            ax = fig.add_subplot(grid[i, 0])
            t, x, blinks = b["example"][cond]
            view = t < 20
            for on, dur in zip(blinks.onset, blinks.duration):
                if on < 20:
                    ax.axvspan(on, on + dur, color=S.CATEGORICAL[3], alpha=0.35, lw=0)
            ax.plot(t[view], x[view], color=color, lw=0.9)
            ax.set(xlim=(0, 20), ylim=(-lim * 1.05, lim * 1.05), ylabel="Fp1 (µV)")
            ax.set_title(f"sub-001, eyes {cond}: first 20 s, detected blinks shaded", loc="left", fontsize=10)
            if i == 1:
                ax.set_xlabel("Time (s)")
        ax = fig.add_subplot(grid[:, 1])
        bo = np.array([r["blinks_open"] for r in rows])
        bc = np.array([r["blinks_closed"] for r in rows])
        for a, c in zip(bo, bc):
            ax.plot([0, 1], [a, c], color=S.AXIS, lw=1, zorder=1)
        ax.scatter(np.zeros_like(bo), bo, s=36, color=OPEN_C, zorder=3, edgecolor=S.SURFACE, linewidth=1.5)
        ax.scatter(np.ones_like(bc), bc, s=36, color=CLOSED_C, zorder=3, edgecolor=S.SURFACE, linewidth=1.5)
        for xpos, v in ((0, bo), (1, bc)):
            ax.plot([xpos - 0.18, xpos + 0.18], [np.median(v)] * 2, color=S.INK, lw=2.2, zorder=4)
        ax.set(xticks=[0, 1], xticklabels=["eyes open", "eyes closed"], xlim=(-0.4, 1.4), ylabel="Blinks per minute")
        ax.set_title(f"Blink rate, {len(rows)} subjects (bar = median)", loc="left", fontsize=10)
        S.grid_y(ax)
    return save(fig, "blinks_eyes_open_closed.png")


def fig_quality(b: dict) -> str:
    rows = sorted(b["rows"], key=lambda r: r["quality"])
    pct = np.array([100 * b["flagged"].get(c, 0) / len(rows) for c in b["ch_names"]])
    with S.style():
        fig, (ax, ax2) = plt.subplots(1, 2, figsize=(11, 4.6), layout="constrained",
                                      gridspec_kw={"width_ratios": [1.3, 1]})
        y = np.arange(len(rows))
        ax.barh(y, [r["quality"] for r in rows], color=OPEN_C, height=0.7)
        for yi, r in zip(y, rows):
            ax.text(r["quality"] + 1, yi, f"{r['quality']:.0f}  ({len(r['flagged'])} flagged)", va="center",
                    fontsize=7.5, color=S.INK_2)
        ax.set(yticks=y, yticklabels=[r["subject"] for r in rows], xlim=(0, 125), xlabel="Median channel score (0-100)")
        ax.tick_params(axis="y", labelsize=7.5, length=0)
        ax.set_title("Signal quality per recording (eyes-open baseline)", loc="left")
        plot_topomap(pct, b["ch_names"], b["positions"], ax=ax2, cmap=S.CMAP_SEQ0, vmin=0, vmax=max(pct.max(), 1),
                     cbar_label="% of subjects", contours=0)
        ax2.set_title("How often each channel was flagged", loc="left")
    return save(fig, "quality_real_recordings.png")


# ---------------------------------------------------------------- 3: motor execution and imagery
def motor_analysis(n_sub: int, runs: tuple, task: str) -> dict:
    rows, courses = [], []
    for s in range(1, n_sub + 1):
        sub = f"sub-{s:03d}"
        raws = [ep.read_raw(mmi_file(sub, r)) for r in runs]
        raw = raws[0]
        for other in raws[1:]:
            raw = raw.append(other)
        raw = laplacian(raw.filter(1.0, 40.0), "hjorth")
        events, eid = raw.find_events()
        eid = {"left": eid[f"{task}T1"], "right": eid[f"{task}T2"]}
        epochs = ep.Epochs(raw, events, eid, tmin=-1.0, tmax=4.0, baseline=None)
        row = {"subject": sub, "n": len(epochs.labels)}
        for band_name, band in (("mu", (8.0, 13.0)), ("beta", (13.0, 30.0))):
            for cond in ("left", "right"):
                rk = erd_ers(epochs, band=band, baseline=(-1.0, 0.0), window=(0.5, 4.0), condition=cond)
                vals = {r["channel"]: r["erd_ers_pct"] for r in rk.ranked()}
                row[f"{band_name}_{cond}_C3"], row[f"{band_name}_{cond}_C4"] = vals["C3"], vals["C4"]
        row["lateral"] = ((row["mu_left_C4"] - row["mu_left_C3"]) - (row["mu_right_C4"] - row["mu_right_C3"]))
        idx = [epochs.ch_names.index(c) for c in ("C3", "Cz", "C4")]
        feats, _ = epoch_features(epochs, idx, "log_bandpower", {"mu": (8.0, 13.0), "beta": (13.0, 30.0)}, (0.5, 4.0))
        row["acc"] = cross_val_accuracy(feats.reshape(len(epochs.labels), -1), epochs.labels, n_folds=5, n_repeats=10)
        row["chance95"] = stats.binom.ppf(0.95, row["n"], 0.5) / row["n"]
        # mu power time course at C3/C4, % change from the 1 s before the cue; the median over
        # trials keeps a single artifact-laden trial from dominating a subject's curve
        pick = [epochs.ch_names.index(c) for c in ("C3", "C4")]
        power = filter_data(epochs.data[:, pick], epochs.sfreq, 8.0, 13.0) ** 2
        course = []
        for cond in ("left", "right"):
            p = uniform_filter1d(np.median(power[epochs.labels == eid[cond]], axis=0), int(0.25 * epochs.sfreq),
                                 axis=-1)
            base = p[:, epochs.times < 0].mean(axis=1, keepdims=True)
            course.append(100 * (p - base) / base)
        courses.append(course)
        rows.append(row)
        print(f"  {sub} {task}: mu C3/C4 left {row['mu_left_C3']:.0f}/{row['mu_left_C4']:.0f}%, "
              f"right {row['mu_right_C3']:.0f}/{row['mu_right_C4']:.0f}%, accuracy {row['acc']:.2f}")
    return {"rows": rows, "times": epochs.times, "courses": np.array(courses)}  # (subject, cond, ch, time)


def fig_motor(executed: dict, imagined: dict) -> str:
    with S.style():
        fig, axes = plt.subplots(2, 2, figsize=(11, 6.4), layout="constrained", sharex=True, sharey=True)
        for row_axes, (label, res) in zip(axes, (("Executed", executed), ("Imagined", imagined))):
            for j, ch in enumerate(("C3 (left motor cortex)", "C4 (right motor cortex)")):
                ax = row_axes[j]
                ax.axhline(0, color=S.AXIS, lw=0.8)
                ax.axvline(0, color=S.AXIS, lw=0.8)
                for k, (cond, color) in enumerate((("left fist", OPEN_C), ("right fist", CLOSED_C))):
                    data = res["courses"][:, k, j]
                    m, e = data.mean(axis=0), sem(data)
                    ax.fill_between(res["times"], m - e, m + e, color=color, alpha=0.16, lw=0)
                    ax.plot(res["times"], m, color=color, lw=2, label=cond)
                ax.set_title(f"{label} movement · {ch}", loc="left", fontsize=10)
                S.grid_y(ax)
                if j == 0:
                    ax.set_ylabel("Mu power vs rest (%)")
                if label == "Imagined":
                    ax.set_xlabel("Time from cue (s)")
        handles, labels = axes[0, 0].get_legend_handles_labels()
        fig.legend(handles, labels, loc="outside upper right", ncol=2, frameon=False)
        fig.suptitle(f"Mu-band (8-13 Hz) power (median over trials), mean of {len(executed['rows'])} subjects "
                     "(±SEM); a drop below 0 is desynchronisation (ERD)", x=0.01, ha="left", fontsize=10, color=S.INK_2)
    return save(fig, "motor_erd.png")


def fig_decoding(executed: dict, imagined: dict) -> str:
    acc_exec = np.array([r["acc"] for r in executed["rows"]]) * 100
    acc_imag = np.array([r["acc"] for r in imagined["rows"]]) * 100
    with S.style():
        fig, ax = plt.subplots(figsize=(6.2, 4.2), layout="constrained")
        ax.axhline(50, color=S.MUTED, lw=1, ls="--")
        ax.text(1.42, 50, "chance", va="center", fontsize=8, color=S.MUTED)
        for a, i in zip(acc_exec, acc_imag):
            ax.plot([0, 1], [a, i], color=S.AXIS, lw=1, zorder=1)
        ax.scatter(np.zeros_like(acc_exec), acc_exec, s=36, color=OPEN_C, zorder=3, edgecolor=S.SURFACE, linewidth=1.5)
        ax.scatter(np.ones_like(acc_imag), acc_imag, s=36, color=CLOSED_C, zorder=3, edgecolor=S.SURFACE, linewidth=1.5)
        for xpos, v in ((0, acc_exec), (1, acc_imag)):
            ax.plot([xpos - 0.18, xpos + 0.18], [np.mean(v)] * 2, color=S.INK, lw=2.2, zorder=4)
        ax.set(xticks=[0, 1], xticklabels=["executed", "imagined"], xlim=(-0.4, 1.6), ylim=(25, 102),
               ylabel="Left vs right accuracy (%)")
        ax.set_title("Decoding left vs right fist per subject (bar = mean)", loc="left", fontsize=10)
        S.grid_y(ax)
    return save(fig, "motor_decoding.png")


# ---------------------------------------------------------------- 4: auditory oddball (P300)
P300_GROUPS = {"standard": ["standard"], "target": ["oddball_with_reponse", "oddball"],
               "distractor": ["noise", "noise_with_reponse"]}
P300_CODES = {"standard": 1, "target": 2, "distractor": 3}


def p300_analysis(n_sub: int) -> dict:
    rows, waves, diff_maps = [], [], []
    for s in range(1, n_sub + 1):
        sub = f"sub-{s:03d}"
        raw = ep.read_raw(fetch_openneuro(ODDBALL, f"{sub}/eeg/{sub}_task-P300_run-1_eeg.set")[0])
        quality = raw.assess_quality()
        raw = raw.filter(0.1, 30.0)
        raw.set_bads(quality.bad_channels)
        raw = raw.interpolate_bads().set_reference("average")
        events, eid = raw.find_events()
        code_of = {eid[n]: P300_CODES[g] for g, names in P300_GROUPS.items() for n in names if n in eid}
        events = events[np.isin(events[:, 2], list(code_of))].copy()
        events[:, 2] = [code_of[c] for c in events[:, 2]]
        epochs = ep.Epochs(raw, events, P300_CODES, tmin=-0.2, tmax=0.8, baseline=(-0.2, 0.0), picks="eeg", reject=150)
        evoked = epochs.average(by_condition=True)
        times = evoked["target"].times
        win = (times >= 0.3) & (times <= 0.5)
        pz = evoked["target"].ch_names.index("Pz")
        amp = {c: float(evoked[c].data[pz, win].mean()) for c in P300_CODES}
        _, latency, _ = evoked["target"].get_peak("Pz", tmin=0.25, tmax=0.6, mode="pos")
        rows.append({"subject": sub, "bads": quality.bad_channels, "dropped": epochs.n_dropped,
                     "n": {c: int((epochs.labels == P300_CODES[c]).sum()) for c in P300_CODES},
                     "amp": amp, "latency": latency})
        waves.append([evoked[c].data[pz] for c in P300_CODES])
        diff_maps.append((evoked["target"].data[:, win] - evoked["standard"].data[:, win]).mean(axis=1))
        print(f"  {sub}: target - standard at Pz {amp['target'] - amp['standard']:+.1f} µV, peak {1000 * latency:.0f} ms")
    return {"rows": rows, "times": times, "waves": np.array(waves), "diff_map": np.mean(diff_maps, axis=0),
            "ch_names": evoked["target"].ch_names, "positions": evoked["target"].positions}


def fig_p300(p: dict) -> str:
    t = p["times"] * 1000
    with S.style():
        fig, (ax, ax2) = plt.subplots(1, 2, figsize=(11, 4.2), layout="constrained",
                                      gridspec_kw={"width_ratios": [1.6, 1]})
        ax.axvspan(300, 500, color=S.GRID, alpha=0.6, lw=0)
        ax.axhline(0, color=S.AXIS, lw=0.8)
        ax.axvline(0, color=S.AXIS, lw=0.8)
        for k, (cond, color) in enumerate(zip(P300_CODES, (OPEN_C, CLOSED_C, THIRD_C))):
            data = p["waves"][:, k]
            m, e = data.mean(axis=0), sem(data)
            ax.fill_between(t, m - e, m + e, color=color, alpha=0.16, lw=0)
            ax.plot(t, m, color=color, lw=2, label=f"{cond} tones")
        ax.set(xlim=(-200, 800), xlabel="Time from tone (ms)", ylabel="Pz (µV)")
        ax.set_title(f"Auditory oddball at Pz, mean of {len(p['rows'])} subjects (±SEM)", loc="left")
        ax.legend(loc="upper left")
        S.grid_y(ax)
        lim = np.abs(p["diff_map"]).max()
        plot_topomap(p["diff_map"], p["ch_names"], p["positions"], ax=ax2, cmap=S.CMAP_DIV, vmin=-lim, vmax=lim,
                     cbar_label="µV", contours=5)
        ax2.set_title("Target minus standard, 300-500 ms", loc="left")
    return save(fig, "p300_oddball.png")


# ---------------------------------------------------------------- results page
def table(header: list[str], rows: list[list]) -> str:
    lines = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    lines += ["| " + " | ".join(str(c) for c in r) + " |" for r in rows]
    return "\n".join(lines)


def write_doc(b, figs, executed, imagined, p) -> None:
    rows = b["rows"]
    ratio = np.array([r["alpha_closed"] / r["alpha_open"] for r in rows])
    p_alpha = stats.wilcoxon(np.log(ratio)).pvalue
    bo, bc = np.array([r["blinks_open"] for r in rows]), np.array([r["blinks_closed"] for r in rows])
    p_blinks = stats.wilcoxon(bo, bc).pvalue
    paf = np.array([r["paf_closed"] for r in rows])
    qual = np.array([r["quality"] for r in rows])
    top_flagged = ", ".join(f"{c} ({n})" for c, n in b["flagged"].most_common(8))
    top_reasons = "\n".join(f"| {reason} | {n} |" for reason, n in b["reasons"].most_common(6))
    fp1_flat = sum("Fp1" in r["bads"] for r in p["rows"])
    post_idx = [b["ch_names"].index(c) for c in POSTERIOR]
    other_idx = [i for i in range(len(b["ch_names"])) if i not in post_idx]
    top_alpha = b["ch_names"][int(np.argmax(b["ratio_map"]))]
    alpha_topo = (f"Averaged over subjects, the increase was {b['ratio_map'][post_idx].mean():.1f} dB over the "
                  f"posterior channels and {b['ratio_map'][other_idx].mean():.1f} dB elsewhere (largest at {top_alpha}).")
    worst = sorted(rows, key=lambda r: r["quality"])[:2]
    worst_text = "; ".join(f"{r['subject']} (median {r['quality']:.0f}) has high-frequency noise on {r['hf_noise']} "
                           f"of 64 channels" + (f" and line noise on {r['line_noise']}" if r["line_noise"] else "")
                           for r in worst)
    top_p300 = p["ch_names"][int(np.argmax(p["diff_map"]))]

    def motor_summary(res, label):
        rr = res["rows"]
        lat = np.array([r["lateral"] for r in rr])
        acc = np.array([r["acc"] for r in rr])
        above = sum(r["acc"] > r["chance95"] for r in rr)
        med = {k: np.median([r[k] for r in rr]) for k in rr[0] if k.startswith(("mu_", "beta_"))}
        return {
            "label": label, "med": med, "lat_n": int((lat < 0).sum()), "lat_p": stats.wilcoxon(lat).pvalue,
            "acc_mean": 100 * acc.mean(), "acc_median": 100 * np.median(acc), "acc_p": stats.wilcoxon(acc - 0.5).pvalue,
            "above": above, "n_sub": len(rr), "best": max(rr, key=lambda r: r["acc"]),
        }

    me, mi = motor_summary(executed, "executed"), motor_summary(imagined, "imagined")

    def contra(m):
        md = m["med"]
        return md["mu_left_C4"] < md["mu_left_C3"] and md["mu_right_C3"] < md["mu_right_C4"]

    def mean_drop(m):
        return np.mean([v for k, v in m["med"].items()])

    contra_text = {(True, True): "appears in the group medians for both tasks",
                   (False, True): "appears in the group medians for imagery only",
                   (True, False): "appears in the group medians for real movement only",
                   (False, False): "does not appear in the group medians"}[(contra(me), contra(mi))]
    drop_text = ("more for real movement than for imagery" if mean_drop(me) < mean_drop(mi)
                 else "about as much for imagery as for real movement")
    d = np.array([r["amp"]["target"] - r["amp"]["standard"] for r in p["rows"]])
    dd = np.array([r["amp"]["distractor"] - r["amp"]["standard"] for r in p["rows"]])
    lat = np.array([r["latency"] for r in p["rows"]]) * 1000
    n_sub, n_p = len(rows), len(p["rows"])

    erd_rows = []
    for m in (me, mi):
        for band in ("mu", "beta"):
            erd_rows.append([m["label"], band] + [f"{m['med'][f'{band}_{c}_{ch}']:.0f}%"
                                                  for c in ("left", "right") for ch in ("C3", "C4")])

    text = f"""# Results on real EEG data

These results come from public recordings analysed with eegproc. Every number
and figure on this page is produced by
[`examples/real_data_openneuro.py`](../examples/real_data_openneuro.py); run it to
reproduce them (it downloads about 1.7 GB of data once).

## Data

| Dataset | What it contains | Used here | Licence |
|---|---|---|---|
| [OpenNeuro {MMI}](https://openneuro.org/datasets/{MMI}) - EEG Motor Movement/Imagery (Schalk et al., 2004; PhysioNet EEGMMIDB) | 64-channel EEG at 160 Hz from 109 volunteers: 1-min eyes-open and eyes-closed baselines, real and imagined fist movements | first {n_sub} subjects: baselines (runs 1-2), executed (runs 3, 7, 11) and imagined (runs 4, 8, 12) left/right fist | CC0 |
| [OpenNeuro {ODDBALL}](https://openneuro.org/datasets/{ODDBALL}) - EEG data from an auditory oddball task (Delorme, 2020) | 64-channel BioSemi EEG at 256 Hz, 13 volunteers, 750 tones per run (70% standard, 15% target, 15% white-noise distractor), eyes closed, button press to targets | all {n_p} subjects, run 1 | CC0 |

Files were read with eegproc's own EEGLAB reader, which also took the electrode
positions, events and (from the BIDS `channels.tsv` files) the channel types from
the dataset.

## 1. Eyes closed vs eyes open: alpha (Berger effect)

Closing the eyes increases posterior alpha (8-13 Hz) power, one of the first EEG
effects ever described (Berger, 1929). Method: 1-45 Hz band-pass, average reference,
Welch spectrum, alpha power averaged over {", ".join(POSTERIOR)}.

**Result:** alpha power was higher with eyes closed in **{(ratio > 1).sum()} of {n_sub}**
subjects (median increase **x{np.median(ratio):.1f}**, range x{ratio.min():.1f} to
x{ratio.max():.0f}; Wilcoxon signed-rank p = {p_text(p_alpha)}). {alpha_topo}
Median peak alpha frequency with eyes closed: {np.median(paf):.1f} Hz.

![Alpha eyes open vs closed]({figs['alpha']})

## 2. Blinks with eyes open and closed

The dataset has no EOG channel, so blinks were detected on Fp1 with
`find_blinks`.

**Result:** median **{np.median(bo):.1f} blinks per minute with eyes open** and
**{np.median(bc):.1f} with eyes closed**; fewer with eyes closed in {(bc < bo).sum()} of
{n_sub} subjects (Wilcoxon p = {p_text(p_blinks)}). Detections with eyes closed are
probably eye movements or other frontal deflections rather than blinks.

![Blinks eyes open vs closed]({figs['blinks']})

## 3. Channel quality of real recordings

`assess_quality` was run on each eyes-open baseline without any cleaning.
Median channel score per recording: **{np.median(qual):.0f}/100** (range
{qual.min():.0f}-{qual.max():.0f}). Most often flagged: {top_flagged} (number of
recordings). Temporal and edge channels typically pick up jaw and neck muscle
activity. The two lowest-scoring recordings: {worst_text}.

Strongest reason given for each flagged channel:

| Reason | Channels |
|---|---|
{top_reasons}

In the oddball dataset, Fp1 was flat (about 0.2 µV) and flagged in
**{fp1_flat} of {n_p}** recordings; it was replaced by interpolation before the ERP
analysis.

![Quality of real recordings]({figs['quality']})

The full dashboard for one recording (sub-001, eyes open):

![Quality dashboard sub-001]({figs['dashboard']})

## 4. Real and imagined fist movements: ERD and decoding

Moving, or imagining moving, a hand lowers mu (8-13 Hz) and beta (13-30 Hz)
power over the motor cortex, more on the side opposite the hand (Pfurtscheller &
Lopes da Silva, 1999). Method: three runs per task joined, 1-40 Hz band-pass,
Hjorth Laplacian, epochs -1 to 4 s around the cue (about 45 per subject), band
power 0.5-4 s compared with the second before the cue. Decoding: shrinkage LDA
on log mu and beta power at C3, Cz and C4 (6 features, chosen in advance),
5-fold cross-validation repeated 10 times. The time-course figure shows the
median over trials, so a single artifact-laden trial cannot dominate it.

Median ERD/ERS (negative = power drop):

{table(["Task", "Band", "Left fist: C3", "Left fist: C4", "Right fist: C3", "Right fist: C4"], erd_rows)}

**Results:**

- Mu and beta power dropped after the cue for both tasks (median of the
  table above: {mean_drop(me):.0f}% for real movement, {mean_drop(mi):.0f}% for imagery),
  {drop_text}.
- The expected contralateral pattern (left fist -> stronger drop at C4, right
  fist -> at C3) {contra_text}; per subject it held in {mi["lat_n"]} of {n_sub} subjects for imagery
  (p = {p_text(mi["lat_p"])}) and {me["lat_n"]} of {n_sub} for real movement
  (p = {p_text(me["lat_p"])}).
- Left vs right decoding: **{me["acc_mean"]:.0f}%** on average for real movement
  (above 50% across subjects, p = {p_text(me["acc_p"])}; {me["above"]} of {n_sub}
  subjects individually above chance) and **{mi["acc_mean"]:.0f}%** for imagery
  (p = {p_text(mi["acc_p"])}; {mi["above"]} of {n_sub} individually above chance). The best
  subject reached {100 * mi["best"]["acc"]:.0f}% for imagery ({mi["best"]["subject"]}).

This is a deliberately simple three-channel analysis of untrained volunteers
with about 45 trials each; methods that use all channels and spatial filters
(such as CSP) usually do better. The imagery result is reported as it came out,
without tuning.

![Motor ERD]({figs['motor']})

![Motor decoding]({figs['decoding']})

## 5. Auditory oddball: the P300

Rare target tones evoke a large positive wave over the parietal scalp about
300-500 ms after the tone, the P300. Method: 0.1-30 Hz band-pass, bad channels
from `assess_quality` interpolated, average reference, epochs -0.2 to 0.8 s with
pre-tone baseline, epochs over 150 µV peak-to-peak rejected, average per
condition, mean amplitude at Pz 300-500 ms.

**Result:** targets produced a larger positive wave than standard tones in
**{(d > 0).sum()} of {n_p}** subjects (median **+{np.median(d):.1f} µV**, Wilcoxon
p = {p_text(stats.wilcoxon(d).pvalue)}); median target peak latency at Pz
**{np.median(lat):.0f} ms**. Distractor noises: median {np.median(dd):+.1f} µV vs
standard over the same window. The target-minus-standard map peaks at {top_p300}.

![P300]({figs['p300']})

## Per-subject numbers

Baselines (motor/imagery dataset):

{table(["Subject", "Median quality", "Flagged channels", "Alpha closed/open", "Peak alpha (Hz)", "Blinks/min open", "Blinks/min closed"],
       [[r["subject"], f"{r['quality']:.0f}", ", ".join(r["flagged"]) or "-",
         f"x{r['alpha_closed'] / r['alpha_open']:.1f}", f"{r['paf_closed']:.1f}",
         f"{r['blinks_open']:.1f}", f"{r['blinks_closed']:.1f}"] for r in rows])}

Movement decoding (balanced accuracy; chance threshold = 95th percentile of a
binomial with that subject's number of trials):

{table(["Subject", "Executed", "Imagined", "Chance threshold"],
       [[e["subject"], f"{100 * e['acc']:.0f}%", f"{100 * i['acc']:.0f}%", f"{100 * i['chance95']:.0f}%"]
        for e, i in zip(executed["rows"], imagined["rows"])])}

Oddball:

{table(["Subject", "Epochs standard/target/distractor", "Dropped", "Target - standard at Pz", "Peak latency", "Interpolated channels"],
       [[r["subject"], "/".join(str(r["n"][c]) for c in P300_CODES), r["dropped"],
         f"{r['amp']['target'] - r['amp']['standard']:+.1f} µV", f"{1000 * r['latency']:.0f} ms",
         ", ".join(r["bads"])] for r in p["rows"]])}

## References

- Berger, H. (1929). Über das Elektrenkephalogramm des Menschen. *Archiv für Psychiatrie und Nervenkrankheiten*, 87, 527-570.
- Delorme, A. (2020). EEG data from an auditory oddball task. OpenNeuro. doi:10.18112/openneuro.ds003061.v1.1.2
- Pfurtscheller, G., & Lopes da Silva, F. H. (1999). Event-related EEG/MEG synchronization and desynchronization: basic principles. *Clinical Neurophysiology*, 110, 1842-1857.
- Schalk, G., McFarland, D. J., Hinterberger, T., Birbaumer, N., & Wolpaw, J. R. (2004). BCI2000: a general-purpose brain-computer interface (BCI) system. *IEEE Transactions on Biomedical Engineering*, 51, 1034-1043.
"""
    DOC.write_text(text, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--subjects", type=int, default=20, help="subjects from the motor dataset (max 109)")
    parser.add_argument("--oddball-subjects", type=int, default=13, help="subjects from the oddball dataset (max 13)")
    args = parser.parse_args()
    print("1-2. Eyes open / closed baselines")
    b = baseline_analysis(args.subjects)
    print("3. Executed movement")
    executed = motor_analysis(args.subjects, (3, 7, 11), "TASK1")
    print("3. Imagined movement")
    imagined = motor_analysis(args.subjects, (4, 8, 12), "TASK2")
    print("4. Auditory oddball")
    p = p300_analysis(min(args.oddball_subjects, 13))
    figs = {"alpha": fig_alpha(b), "blinks": fig_blinks(b), "quality": fig_quality(b),
            "dashboard": "images/results/quality_dashboard_sub-001.png",
            "motor": fig_motor(executed, imagined), "decoding": fig_decoding(executed, imagined), "p300": fig_p300(p)}
    write_doc(b, figs, executed, imagined, p)
    print(f"Wrote {DOC} and {len(figs)} figures in {FIG_DIR}")


if __name__ == "__main__":
    main()
