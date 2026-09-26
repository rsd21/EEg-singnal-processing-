"""Connectivity matrices and scalp networks."""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np

from ..core.montage import project_to_2d
from . import _style as S
from .topomap import _resolve_positions, draw_head


def plot_connectivity(con, top: int = 20, show: bool = False):
    """Matrix (left) and the ``top`` strongest links drawn on the head (right)."""
    m = con.matrix.copy()
    np.fill_diagonal(m, np.nan)
    signed = np.nanmin(m) < 0
    keep, xyz = _resolve_positions(con.ch_names, con.positions)
    with S.style():
        ncols = 2 if len(keep) >= 3 else 1
        fig, axes = plt.subplots(1, ncols, figsize=(5.4 * ncols + 0.8, 4.8), layout="constrained")
        axes = np.atleast_1d(axes)
        ax = axes[0]
        lim = np.nanmax(np.abs(m))
        im = ax.imshow(m, cmap=S.CMAP_DIV if signed else S.CMAP_SEQ, vmin=-lim if signed else 0, vmax=lim)
        n = len(con.ch_names)
        step = 1 if n <= 32 else 2
        ax.set_xticks(range(0, n, step))
        ax.set_xticklabels(con.ch_names[::step], rotation=90, fontsize=6 if n > 32 else 7)
        ax.set_yticks(range(0, n, step))
        ax.set_yticklabels(con.ch_names[::step], fontsize=6 if n > 32 else 7)
        ax.tick_params(length=0)
        for spine in ax.spines.values():
            spine.set_visible(False)
        cb = fig.colorbar(im, ax=ax, fraction=0.045, pad=0.02)
        cb.outline.set_visible(False)
        band = f"{con.band[0]:g}-{con.band[1]:g} Hz" if con.band else ""
        ax.set_title(f"{con.method} {band}", loc="left")
        if ncols == 2:
            ax = axes[1]
            xy = project_to_2d(xyz)
            pos = {con.ch_names[i]: xy[k] for k, i in enumerate(keep)}
            draw_head(ax, max(1.0, float(np.max(np.hypot(xy[:, 0], xy[:, 1]))) * 1.04))
            links = [l for l in con.strongest(top) if l["channel_a"] in pos and l["channel_b"] in pos]
            vmax = max(abs(l["value"]) for l in links) if links else 1
            for l in links[::-1]:
                a, b = pos[l["channel_a"]], pos[l["channel_b"]]
                w = abs(l["value"]) / vmax
                ax.plot([a[0], b[0]], [a[1], b[1]], color=S.CATEGORICAL[0], lw=0.6 + 2.6 * w,
                        alpha=0.35 + 0.6 * w, zorder=3)
            strength = con.node_strength()
            ax.scatter(xy[:, 0], xy[:, 1], s=20 + 160 * (strength[keep] / max(strength.max(), 1e-12)),
                       c=S.INK, zorder=5, edgecolors=S.SURFACE, linewidths=2)
            for (x, y), i in zip(xy, keep):
                ax.text(x, y - 0.08, con.ch_names[i], ha="center", va="top", fontsize=6.5, color=S.INK_2,
                        zorder=6)
            ax.set_title(f"Top {len(links)} links · dot size = node strength", loc="left")
    return S.finish(fig, show)
