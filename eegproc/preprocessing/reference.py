"""Re-referencing: average, median, channel(s), bipolar chains and Laplacians."""

from __future__ import annotations

from typing import Iterable, Sequence

import numpy as np

from .._spline import csd_matrix
from ..core.montage import find_neighbors
from ..utils import logger

# Longitudinal bipolar ("double banana") montage used in clinical EEG review.
DOUBLE_BANANA = [
    ("Fp1", "F7"), ("F7", "T7"), ("T7", "P7"), ("P7", "O1"),
    ("Fp2", "F8"), ("F8", "T8"), ("T8", "P8"), ("P8", "O2"),
    ("Fp1", "F3"), ("F3", "C3"), ("C3", "P3"), ("P3", "O1"),
    ("Fp2", "F4"), ("F4", "C4"), ("C4", "P4"), ("P4", "O2"),
    ("Fz", "Cz"), ("Cz", "Pz"),
]
# Transverse bipolar montage.
TRANSVERSE = [
    ("F7", "Fp1"), ("Fp1", "Fp2"), ("Fp2", "F8"), ("F7", "F3"), ("F3", "Fz"), ("Fz", "F4"),
    ("F4", "F8"), ("T7", "C3"), ("C3", "Cz"), ("Cz", "C4"), ("C4", "T8"), ("P7", "P3"),
    ("P3", "Pz"), ("Pz", "P4"), ("P4", "P8"), ("O1", "O2"),
]


def set_reference(raw, ref: str | Sequence[str] = "average", picks="eeg", exclude_bads: bool = True,
                  drop_ref: bool = False):
    """Re-reference EEG channels.

    ``ref`` may be ``'average'`` (common average of good channels),
    ``'median'`` (robust common reference), ``'laplacian'`` / ``'csd'``, a
    channel name, or a list of channels whose mean is the new reference
    (e.g. ``['M1', 'M2']`` for linked mastoids). Only EEG channels are
    changed; bad channels are re-referenced but not used to build the
    reference.
    """
    if isinstance(ref, str) and ref.lower() in ("laplacian", "hjorth"):
        return laplacian(raw, method="hjorth")
    if isinstance(ref, str) and ref.lower() in ("csd", "spline"):
        return laplacian(raw, method="spline")
    targets = raw.pick_indices(picks)
    if not targets:
        raise ValueError("No channels to re-reference")
    out = raw.copy()
    if isinstance(ref, str) and ref.lower() in ("average", "avg", "car", "median"):
        pool = [i for i in targets if not (exclude_bads and raw.ch_names[i] in raw.bads)]
        if not pool:
            raise ValueError("No good channels available to build an average reference")
        block = raw.data[pool]
        reference = np.median(block, axis=0) if ref.lower() == "median" else block.mean(axis=0)
        label = "median" if ref.lower() == "median" else "average"
    else:
        names = [ref] if isinstance(ref, str) else list(ref)
        ref_idx = raw.pick_indices(names)
        reference = raw.data[ref_idx].mean(axis=0)
        label = "+".join(raw.ch_names[i] for i in ref_idx)
    out.data[targets] = raw.data[targets] - reference
    out.meta["reference"] = label
    out.history.append(f"re-reference: {label}")
    if drop_ref and not (isinstance(ref, str) and ref.lower() in ("average", "avg", "car", "median")):
        out = out.drop_channels([raw.ch_names[i] for i in ref_idx])
    return out


def add_reference_channel(raw, name: str):
    """Add the (implicit) recording reference back as an all-zero channel.

    Useful before average referencing when the reference electrode (e.g. Cz
    or FCz) was not stored in the file.
    """
    out = raw.add_channels(np.zeros((1, raw.n_times)), [name], ["eeg"], ["uV"])
    from ..core.montage import Montage

    xyz = Montage.standard("standard_1010").get(name)
    if xyz is not None:
        out.positions[name] = xyz
    return out


def bipolar_reference(raw, pairs: Iterable[tuple[str, str]] | str = "double_banana",
                      keep_others: bool = False, on_missing: str = "skip"):
    """Create derived channels ``anode - cathode``.

    ``pairs`` is a list of ``(anode, cathode)`` names, ``'double_banana'`` or
    ``'transverse'``. Pairs with missing channels are skipped (or raise with
    ``on_missing='raise'``).
    """
    if isinstance(pairs, str):
        pairs = {"double_banana": DOUBLE_BANANA, "transverse": TRANSVERSE}[pairs.lower()]
    rows, names, used = [], [], set()
    for a, c in pairs:
        try:
            ia, ic = raw.channel_index(a), raw.channel_index(c)
        except KeyError:
            if on_missing == "raise":
                raise
            continue
        rows.append(raw.data[ia] - raw.data[ic])
        names.append(f"{raw.ch_names[ia]}-{raw.ch_names[ic]}")
        used.update((ia, ic))
    if not rows:
        raise ValueError("None of the bipolar pairs are present in the recording")
    from ..core.raw import RawEEG

    data = np.array(rows)
    ch_names, ch_types, units = names, ["eeg"] * len(names), ["uV"] * len(names)
    positions = {}
    for n, (a, c) in zip(names, [n.split("-", 1) for n in names]):
        if a in raw.positions and c in raw.positions:
            positions[n] = (raw.positions[a] + raw.positions[c]) / 2
    if keep_others:
        others = [i for i in range(raw.n_channels) if raw.ch_types[i] != "eeg"]
        data = np.vstack([data, raw.data[others]]) if others else data
        ch_names = ch_names + [raw.ch_names[i] for i in others]
        ch_types = ch_types + [raw.ch_types[i] for i in others]
        units = units + [raw.units[i] for i in others]
    out = RawEEG(data, raw.sfreq, ch_names, ch_types, units, raw.annotations, positions,
                 meta=dict(raw.meta), history=list(raw.history))
    out.meta["reference"] = "bipolar"
    out.history.append(f"bipolar montage ({len(names)} derivations)")
    return out


def laplacian(raw, method: str = "hjorth", n_neighbors: int = 4, stiffness: int = 4,
              lambda2: float = 1e-5):
    """Spatial high-pass filter that sharpens local activity.

    * ``'hjorth'``: each channel minus the mean of its nearest neighbours.
    * ``'spline'``: spherical-spline surface Laplacian / current source
      density (reference-free, units µV/r² on the unit sphere).
    Requires electrode positions (``raw.set_montage(...)``).
    """
    eeg = [i for i in raw.pick_indices("eeg") if raw.ch_names[i] not in raw.bads]
    missing = [raw.ch_names[i] for i in eeg if raw.ch_names[i] not in raw.positions]
    if missing:
        raise ValueError(f"Laplacian needs electrode positions; missing for {missing}. "
                         "Call raw.set_montage('standard_1010') first.")
    if len(eeg) < 5:
        raise ValueError("Laplacian needs at least 5 EEG channels")
    pos = np.array([raw.positions[raw.ch_names[i]] for i in eeg])
    out = raw.copy()
    if method == "hjorth":
        neigh = find_neighbors(pos, n_neighbors)
        block = raw.data[eeg]
        new = np.empty_like(block)
        for k, nb in enumerate(neigh):
            new[k] = block[k] - block[nb].mean(axis=0)
        out.data[eeg] = new
    elif method == "spline":
        out.data[eeg] = csd_matrix(pos, lambda2, stiffness) @ raw.data[eeg]
        for i in eeg:
            out.units[i] = "uV/r2"
    else:
        raise ValueError("method must be 'hjorth' or 'spline'")
    if raw.bads:
        logger.info("Bad channels %s were left unchanged by the Laplacian", raw.bads)
    out.meta["reference"] = f"laplacian-{method}"
    out.history.append(f"laplacian ({method})")
    return out
