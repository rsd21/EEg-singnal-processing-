"""Topographic scalp maps and sensor layouts (matplotlib only).

Values are interpolated over the head with spherical splines evaluated on the
sphere (not on the flat projection), so maps are smooth and physically
consistent with the bad-channel interpolation used elsewhere.
"""

from __future__ import annotations

from typing import Sequence

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Circle, Ellipse, Polygon

from .._spline import interpolation_matrix
from ..core.montage import Montage, project_to_2d, unproject_from_2d
from . import _style as S


def _resolve_positions(ch_names: Sequence[str], positions: dict | None) -> tuple[list[int], np.ndarray]:
    """Indices of channels with a position and their (n, 3) coordinates."""
    std = Montage.standard("standard_1010")
    keep, xyz = [], []
    for i, n in enumerate(ch_names):
        p = positions.get(n) if positions else None
        if p is None:
            p = std.get(n)
        if p is not None:
            keep.append(i)
            xyz.append(p)
    return keep, np.array(xyz).reshape(-1, 3)


def draw_head(ax, radius: float = 1.0, color: str = S.AXIS, lw: float = 1.5) -> None:
    ax.add_patch(Circle((0, 0), radius, fill=False, ec=color, lw=lw, zorder=4))
    nose = Polygon([[-0.09 * radius, 0.995 * radius], [0, 1.1 * radius], [0.09 * radius, 0.995 * radius]],
                   closed=False, fill=False, ec=color, lw=lw, zorder=4)
    ax.add_patch(nose)
    for side in (-1, 1):
        ax.add_patch(Ellipse((side * 1.04 * radius, 0), 0.08 * radius, 0.3 * radius, fill=False, ec=color,
                             lw=lw, zorder=4))
    ax.set_xlim(-1.2 * radius, 1.2 * radius)
    ax.set_ylim(-1.15 * radius, 1.2 * radius)
    ax.set_aspect("equal")
    ax.axis("off")


def plot_topomap(values, ch_names: Sequence[str], positions: dict | None = None, ax=None,
                 cmap=None, vmin: float | None = None, vmax: float | None = None,
                 diverging: bool | None = None, show_names: bool = False, sensors: bool = True,
                 highlight: Sequence[str] | None = None, contours: int = 6, res: int = 72,
                 title: str | None = None, colorbar: bool = True, cbar_label: str = "",
                 sensor_colors: Sequence[str] | None = None):
    """Draw a scalp map of one value per channel.

    Parameters
    ----------
    values : array (n_channels,)
    ch_names : channel names (positions come from ``positions`` or the
        standard 10-10 layout).
    diverging : use the blue-red map centred on 0 (default: when values
        take both signs).
    highlight : channels drawn with a larger ring (e.g. the best channels).
    """
    values = np.asarray(values, dtype=float)
    keep, xyz = _resolve_positions(ch_names, positions)
    if len(keep) < 3:
        raise ValueError("A topomap needs at least 3 channels with known positions")
    vals = values[keep]
    names = [ch_names[i] for i in keep]
    finite = np.isfinite(vals)
    if ax is None:
        with S.style():
            fig, ax = plt.subplots(figsize=(3.6, 3.4))
    fig = ax.figure
    xy = project_to_2d(xyz)
    head_r = max(1.0, float(np.max(np.hypot(xy[:, 0], xy[:, 1]))) * 1.04)
    g = np.linspace(-head_r, head_r, res)
    gx, gy = np.meshgrid(g, g)
    inside = np.hypot(gx, gy) <= head_r
    grid_xyz = unproject_from_2d(np.column_stack([gx[inside], gy[inside]]))
    mat = interpolation_matrix(xyz[finite], grid_xyz)
    z = np.full(gx.shape, np.nan)
    z[inside] = mat @ vals[finite]

    if diverging is None:
        diverging = np.nanmin(vals) < 0 < np.nanmax(vals)
    if diverging:
        lim = np.nanmax(np.abs(vals)) if vmax is None else max(abs(vmax), abs(vmin or vmax))
        vmin, vmax = -lim, lim
        cmap = cmap or S.CMAP_DIV
    else:
        vmin = np.nanmin(vals) if vmin is None else vmin
        vmax = np.nanmax(vals) if vmax is None else vmax
        cmap = cmap or S.CMAP_SEQ
    if vmax == vmin:
        vmax = vmin + 1e-9
    im = ax.imshow(z, extent=(-head_r, head_r, -head_r, head_r), origin="lower", cmap=cmap,
                   vmin=vmin, vmax=vmax, interpolation="bilinear", zorder=1)
    im.set_clip_path(Circle((0, 0), head_r, transform=ax.transData))
    if contours and np.nanmax(z) > np.nanmin(z):
        cs = ax.contour(gx, gy, np.clip(z, vmin, vmax), levels=np.linspace(vmin, vmax, contours + 2)[1:-1],
                        colors=S.SURFACE, linewidths=0.6, linestyles="solid", alpha=0.8, zorder=2)
        for c in getattr(cs, "collections", []):
            c.set_clip_path(Circle((0, 0), head_r, transform=ax.transData))
    draw_head(ax, head_r)
    if sensors:
        colors = [sensor_colors[i] for i in keep] if sensor_colors is not None else [S.INK] * len(keep)
        ax.scatter(xy[:, 0], xy[:, 1], s=9 if len(keep) > 40 else 14, c=colors, zorder=5,
                   edgecolors=S.SURFACE, linewidths=0.8)
    if highlight:
        hl = [k for k, n in enumerate(names) if n in set(highlight)]
        ax.scatter(xy[hl, 0], xy[hl, 1], s=90, facecolors="none", edgecolors=S.INK, linewidths=1.6, zorder=6)
    if show_names:
        fs = 6 if len(keep) > 40 else 7
        for (x, y), n in zip(xy, names):
            ax.text(x, y + 0.055 * head_r, n, ha="center", va="bottom", fontsize=fs, color=S.INK_2, zorder=6)
    if title:
        ax.set_title(title, pad=4)
    if colorbar:
        cb = fig.colorbar(im, ax=ax, fraction=0.045, pad=0.02)
        cb.outline.set_visible(False)
        cb.ax.tick_params(labelsize=7, colors=S.MUTED)
        if cbar_label:
            cb.set_label(cbar_label, fontsize=8, color=S.INK_2)
    return fig


def plot_sensors(raw_or_names, positions: dict | None = None, colors: dict | None = None,
                 labels: dict | None = None, title: str | None = "Sensor layout", ax=None,
                 legend: dict | None = None, show: bool = False):
    """Electrode layout; optionally colour each sensor (e.g. by quality grade).

    ``raw_or_names`` is a :class:`RawEEG` (EEG channels) or a list of names.
    ``colors`` maps channel -> colour, ``legend`` maps label -> colour.
    """
    from ..core.raw import RawEEG

    if isinstance(raw_or_names, RawEEG):
        names = raw_or_names.pick_names("eeg")
        positions = positions or raw_or_names.positions
        bads = set(raw_or_names.bads)
        if colors is None and bads:
            colors = {n: (S.STATUS["critical"] if n in bads else S.INK) for n in names}
            legend = legend or {"good": S.INK, "marked bad": S.STATUS["critical"]}
    else:
        names = list(raw_or_names)
    keep, xyz = _resolve_positions(names, positions)
    xy = project_to_2d(xyz)
    with S.style():
        if ax is None:
            fig, ax = plt.subplots(figsize=(4.4, 4.4))
        fig = ax.figure
        head_r = max(1.0, float(np.max(np.hypot(xy[:, 0], xy[:, 1]))) * 1.04) if len(xy) else 1.0
        draw_head(ax, head_r)
        cols = [(colors or {}).get(names[i], S.INK) for i in keep]
        ax.scatter(xy[:, 0], xy[:, 1], s=60 if len(keep) <= 40 else 30, c=cols, zorder=5,
                   edgecolors=S.SURFACE, linewidths=2)
        fs = 7 if len(keep) <= 40 else 5.5
        for k, i in enumerate(keep):
            txt = names[i] if labels is None else labels.get(names[i], names[i])
            ax.text(xy[k, 0], xy[k, 1] - 0.075 * head_r, txt, ha="center", va="top", fontsize=fs,
                    color=S.INK_2, zorder=6)
        if legend:
            from matplotlib.lines import Line2D

            handles = [Line2D([], [], marker="o", ls="", markersize=7, markerfacecolor=c,
                              markeredgecolor=S.SURFACE, label=l) for l, c in legend.items()]
            ax.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.5, -0.1), ncol=len(handles))
        if title:
            ax.set_title(title)
        missing = [names[i] for i in range(len(names)) if i not in keep]
        if missing:
            ax.text(0, -1.12 * head_r, f"no position: {', '.join(missing[:8])}{' ...' if len(missing) > 8 else ''}",
                    ha="center", fontsize=7, color=S.MUTED)
    return S.finish(fig, show)


def plot_band_topomaps(raw, bands: dict | None = None, relative: bool = True, show: bool = False,
                       show_names: bool = False):
    """One scalp map per frequency band, marking the channel with the most power."""
    from ..analysis.spectral import BANDS, compute_psd

    bands = bands or BANDS
    names = [c for c in raw.pick_names("eeg") if c not in raw.bads]
    spec = compute_psd(raw, picks=names, window_sec=2.0, average="median")
    bands = {k: v for k, v in bands.items() if v[0] < spec.freqs[-1]}
    power = spec.band_power(bands, relative=relative)
    with S.style():
        fig, axes = plt.subplots(1, len(bands), figsize=(2.9 * len(bands), 3.4), layout="constrained")
        axes = np.atleast_1d(axes)
        for ax, (b, vals) in zip(axes, power.items()):
            best = spec.ch_names[int(np.nanargmax(vals))]
            plot_topomap(vals * (100 if relative else 1), spec.ch_names, spec.positions, ax=ax,
                         title=f"{b} {bands[b][0]:g}-{bands[b][1]:g} Hz", highlight=[best],
                         show_names=show_names, cbar_label="% of power" if relative else "µV²")
            ax.text(0, -1.25, f"max: {best}", ha="center", fontsize=8, color=S.INK_2)
        fig.suptitle("Where is each rhythm strongest?" + (" (relative power)" if relative else ""),
                     color=S.INK, fontsize=11, x=0.01, ha="left")
    return S.finish(fig, show)


def plot_ica_components(ica, picks: Sequence[int] | None = None, ncols: int = 5, show: bool = False):
    """Scalp maps of ICA components, titled with their label and variance."""
    comps = ica.get_components()
    picks = list(range(comps.shape[1])) if picks is None else list(picks)
    nrows = int(np.ceil(len(picks) / ncols))
    with S.style():
        fig, axes = plt.subplots(nrows, ncols, figsize=(2.3 * ncols, 2.5 * nrows), layout="constrained")
        axes = np.atleast_1d(axes).ravel()
        for ax, k in zip(axes, picks):
            label = ica.labels_.get(k, "")
            title = f"IC{k} · {100 * ica.explained_variance_[k]:.1f}%" + (f"\n{label}" if label else "\n ")
            plot_topomap(comps[:, k], ica.ch_names, ica.positions, ax=ax, colorbar=False, title=title,
                         diverging=True, contours=4)
            if k in ica.exclude:
                ax.text(0, -1.25, "✕ excluded", ha="center", fontsize=8, color=S.INK_2)
        for ax in axes[len(picks):]:
            ax.axis("off")
    return S.finish(fig, show)
