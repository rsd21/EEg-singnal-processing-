"""Repair bad channels by spherical-spline interpolation from good neighbours."""

from __future__ import annotations

from typing import Iterable

import numpy as np

from .._spline import interpolation_matrix
from ..utils import logger


def interpolate_bads(raw, bads: Iterable[str] | None = None, reset_bads: bool = True,
                     alpha: float = 1e-5):
    """Replace bad EEG channels with values interpolated from the good ones.

    Parameters
    ----------
    bads : list of str, optional
        Channels to interpolate (default: ``raw.bads``).
    reset_bads : bool
        Remove the repaired channels from ``raw.bads``.
    """
    bads = list(raw.bads if bads is None else bads)
    eeg = raw.pick_indices("eeg")
    bad_idx = [i for i in eeg if raw.ch_names[i] in bads]
    if not bad_idx:
        logger.info("No bad EEG channels to interpolate")
        return raw.copy()
    good_idx = [i for i in eeg if raw.ch_names[i] not in bads and raw.ch_names[i] in raw.positions]
    no_pos = [raw.ch_names[i] for i in bad_idx if raw.ch_names[i] not in raw.positions]
    if no_pos:
        raise ValueError(f"Cannot interpolate {no_pos}: no electrode positions. "
                         "Call raw.set_montage('standard_1010') first.")
    if len(good_idx) < 4:
        raise ValueError("Interpolation needs at least 4 good channels with positions")
    pos_good = np.array([raw.positions[raw.ch_names[i]] for i in good_idx])
    pos_bad = np.array([raw.positions[raw.ch_names[i]] for i in bad_idx])
    mat = interpolation_matrix(pos_good, pos_bad, alpha=alpha)
    out = raw.copy()
    out.data[bad_idx] = mat @ raw.data[good_idx]
    repaired = [raw.ch_names[i] for i in bad_idx]
    if reset_bads:
        out.bads = [b for b in out.bads if b not in repaired]
    out.history.append(f"interpolate {repaired}")
    return out
