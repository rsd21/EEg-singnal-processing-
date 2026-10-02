"""Figures that show which channels perform well or badly."""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

from . import _style as S
from .topomap import plot_sensors, plot_topomap

PROBLEMS = ["flat", "clipping", "amplitude", "correlation", "hf_noise", "line_noise", "drift",
            "artifacts", "spikes", "instability", "spectrum"]
PROBLEM_LABELS = {"flat": "flat", "clipping": "clipping", "amplitude": "amplitude",
                  "correlation": "neighbour r", "hf_noise": "HF noise", "line_noise": "line noise",
                  "drift": "drift", "artifacts": "artifacts", "spikes": "pops/spikes",
                  "instability": "unstable", "spectrum": "flat PSD"}


def _grade_legend(ax, loc="lower center", anchor=(0.5, 1.0)):
    handles = [Line2D([], [], marker="s", ls="", markersize=8, markerfacecolor=c, markeredgecolor=c,
                      label=f"{g}") for g, c in S.GRADE_STATUS.items()]
    ax.legend(handles=handles, loc=loc, bbox_to_anchor=anchor, ncol=4, handletextpad=0.3,
              columnspacing=1.0)


def plot_quality_bars(report, ax=None, max_label: int = 38, show: bool = False):
    """Channels ranked by quality score; bars coloured by grade, problems written on weak channels."""
    rows = report.ranked()
    with S.style():
        if ax is None:
            fig, ax = plt.subplots(figsize=(8, 0.26 * len(rows) + 1.4), layout="constrained")
        fig = ax.figure
        y = np.arange(len(rows))
        scores = [r["score"] for r in rows]
        colors = [S.GRADE_STATUS[r["grade"]] for r in rows]
        ax.barh(y, scores, height=0.62, color=colors, edgecolor=S.SURFACE, linewidth=1)
        for k, r in enumerate(rows):
            if r["grade"] != "good" and r["reasons"]:
                txt = r["reasons"][0]
                txt = txt if len(txt) <= max_label else txt[: max_label - 1] + "…"
                ax.text(min(r["score"], 100) + 1.5, k, txt, va="center", fontsize=7, color=S.INK_2)
            elif r["grade"] != "good":
                ax.text(r["score"] + 1.5, k, "PREP: " + ", ".join(r["prep_flags"]), va="center", fontsize=7,
                        color=S.INK_2)
        ax.set_yticks(y)
        ax.set_yticklabels([r["channel"] for r in rows])
        ax.invert_yaxis()
        ax.set_xlim(0, 100)
        ax.set_xlabel("Quality score (0-100)")
        for lim in (25, 50, 75):
            ax.axvline(lim, color=S.GRID, lw=0.8, zorder=0)
        ax.tick_params(axis="y", length=0)
        ax.spines["left"].set_visible(False)
        _grade_legend(ax)
        ax.set_title("Channel ranking by signal quality", loc="left", pad=22)
    return S.finish(fig, show)


def plot_quality_topomap(report, metric: str = "score", ax=None, show: bool = False, **kwargs):
    """Scalp map of the quality score (or any metric column of the report)."""
    vals = report.scores if metric == "score" else report.metric(metric)
    worst = [r["channel"] for r in report.ranked()[::-1] if r["grade"] in ("bad", "poor")]
    kwargs.setdefault("show_names", True)
    with S.style():
        fig = plot_topomap(vals, report.ch_names, report.positions, ax=ax, highlight=worst,
                           title=f"{metric.replace('_', ' ')} (rings: poor/bad)", diverging=False,
                           vmin=0 if metric == "score" else None, vmax=100 if metric == "score" else None,
                           sensor_colors=[S.GRADE_STATUS[g] for g in report.grades], **kwargs)
    return S.finish(fig, show)


def plot_quality_heatmap(report, ax=None, show: bool = False):
    """Which problem costs each channel how many points (channels x problems)."""
    rows = report.ranked()
    problems = [p for p in PROBLEMS if any(r["penalties"].get(p, 0) > 0 for r in rows)] or PROBLEMS[:4]
    mat = np.array([[r["penalties"].get(p, 0.0) for p in problems] for r in rows])
    with S.style():
        if ax is None:
            fig, ax = plt.subplots(figsize=(0.7 * len(problems) + 2.2, 0.24 * len(rows) + 1.6))
        fig = ax.figure
        im = ax.imshow(mat, aspect="auto", cmap=S.CMAP_SEQ0, vmin=0, vmax=max(10.0, float(mat.max())))
        ax.set_xticks(range(len(problems)))
        ax.set_xticklabels([PROBLEM_LABELS[p] for p in problems], rotation=40, ha="right")
        ax.set_yticks(range(len(rows)))
        ax.set_yticklabels([r["channel"] for r in rows])
        ax.tick_params(length=0)
        for spine in ax.spines.values():
            spine.set_visible(False)
        cb = fig.colorbar(im, ax=ax, fraction=0.05, pad=0.02)
        cb.outline.set_visible(False)
        cb.set_label("points lost", fontsize=8)
        ax.set_title("Why channels lose points", loc="left")
    return S.finish(fig, show)


def plot_quality_report(report, show: bool = False):
    """Dashboard: ranked scores, sensor map by grade, score topomap and problem heatmap."""
    n = len(report.rows)
    with S.style():
        fig = plt.figure(figsize=(14, max(7.5, 0.26 * n + 2.5)), layout="constrained")
        gs = fig.add_gridspec(2, 3, width_ratios=[1.35, 1.0, 1.0], height_ratios=[1, 1])
        ax_bar = fig.add_subplot(gs[:, 0])
        plot_quality_bars(report, ax=ax_bar)
        ax_sens = fig.add_subplot(gs[0, 1])
        has_pos = len(report.positions) >= 3 or _has_standard(report.ch_names)
        if has_pos:
            colors = {r["channel"]: S.GRADE_STATUS[r["grade"]] for r in report.rows}
            labels = {r["channel"]: f"{r['channel']}\n{r['score']:.0f}" for r in report.rows}
            plot_sensors(report.ch_names, report.positions, colors=colors, labels=labels,
                         title="Grade per electrode", ax=ax_sens)
            ax_topo = fig.add_subplot(gs[0, 2])
            plot_quality_topomap(report, ax=ax_topo, show_names=False)
        else:
            ax_sens.text(0.5, 0.5, "No electrode positions:\nset a montage to see scalp maps",
                         ha="center", va="center", color=S.MUTED)
            ax_sens.axis("off")
        ax_heat = fig.add_subplot(gs[1, 1:])
        plot_quality_heatmap(report, ax=ax_heat)
        counts = {g: len(v) for g, v in report.by_grade().items()}
        fig.suptitle(f"Channel quality · median score {report.overall_score:.0f}/100 · " +
                     " · ".join(f"{v} {k}" for k, v in counts.items()), x=0.01, ha="left",
                     fontsize=13, color=S.INK, fontweight="bold")
    return S.finish(fig, show)


def _has_standard(names) -> bool:
    from ..core.montage import standard_name

    return sum(standard_name(n) is not None for n in names) >= 3


def plot_ranking(ranking, kind: str = "both", top: int = 20, show: bool = False):
    """Bar chart of the best channels and (if positions are known) a scalp map."""
    rows = ranking.ranked()[:top]
    has_pos = len(ranking.positions) >= 3 or _has_standard(ranking.ch_names)
    want_topo = kind in ("both", "topomap") and has_pos
    want_bar = kind in ("both", "bar") or not want_topo
    with S.style():
        ncols = int(want_bar) + int(want_topo)
        fig, axes = plt.subplots(1, ncols, figsize=(5.2 * ncols + 0.6, max(3.6, 0.24 * len(rows) + 1.4)),
                                 gridspec_kw={"width_ratios": [1.2, 1.0][:ncols]}, layout="constrained")
        axes = np.atleast_1d(axes)
        k = 0
        if want_bar:
            ax = axes[k]
            k += 1
            vals = [r[ranking.metric] for r in rows]
            y = np.arange(len(rows))
            colors = [S.CATEGORICAL[0]] * len(rows)
            ax.barh(y, vals, height=0.62, color=colors, edgecolor=S.SURFACE)
            ax.set_yticks(y)
            ax.set_yticklabels([r["channel"] for r in rows])
            ax.invert_yaxis()
            ax.tick_params(axis="y", length=0)
            ax.spines["left"].set_visible(False)
            ax.axvline(0, color=S.AXIS, lw=0.8)
            S.grid_x(ax)
            ax.set_xlabel(ranking.metric.replace("_", " ") + (f" ({ranking.unit.strip()})" if ranking.unit else ""))
            best = rows[0]
            ax.text(best[ranking.metric], 0, f"  {best[ranking.metric]:.3g}", va="center", fontsize=8,
                    color=S.INK)
            chance = ranking.details.get("chance_level")
            if chance is not None:
                ax.axvline(chance, color=S.INK_2, lw=1)
                ax.text(chance, len(rows) - 0.4, " chance", fontsize=7, color=S.INK_2, va="bottom")
            ax.set_title(f"Top channels by {ranking.metric.replace('_', ' ')}", loc="left")
        if want_topo:
            ax = axes[k]
            signed = ranking.details.get("erd_ers_pct")
            vals = np.asarray(signed) if signed is not None else ranking.values
            plot_topomap(vals, ranking.ch_names, ranking.positions, ax=ax, highlight=ranking.top(3),
                         show_names=len(ranking.ch_names) <= 32,
                         title="ERD/ERS %" if signed is not None else ranking.metric.replace("_", " "),
                         cbar_label=ranking.unit.strip())
        if ranking.description:
            fig.suptitle(ranking.description, x=0.01, ha="left", fontsize=8, color=S.MUTED,
                         fontweight="normal")
    return S.finish(fig, show)
