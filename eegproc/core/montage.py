"""Electrode positions: standard 10-20 / 10-10 layouts, custom files, projection.

Positions live on a unit sphere in head coordinates: +x towards the right ear,
+y towards the nose and +z towards the vertex (Cz = (0, 0, 1)).

The standard layouts are built geometrically from the idealised spherical head
used by the 10-20 system: the nasion-inion arc and the 10 % circumference are
split into equal steps and intermediate rows are placed by great-circle
interpolation. This is accurate enough for topographic maps, spline
interpolation and neighbour search, and needs no external data files.
"""

from __future__ import annotations

import csv
import os
import re
from typing import Iterable

import numpy as np

# Old 10-20 names and common alternative spellings mapped to 10-10 names.
ALIASES = {
    "t3": "T7", "t4": "T8", "t5": "P7", "t6": "P8",
    "a1": "A1", "a2": "A2", "m1": "M1", "m2": "M2",
    "tp9": "TP9", "tp10": "TP10",
}


def _sph_to_cart(theta_deg: float, phi_deg: float) -> np.ndarray:
    """theta: angle from the vertex, phi: azimuth (0 = right ear, 90 = nose)."""
    t, p = np.deg2rad(theta_deg), np.deg2rad(phi_deg)
    return np.array([np.sin(t) * np.cos(p), np.sin(t) * np.sin(p), np.cos(t)])


def _slerp(p: np.ndarray, q: np.ndarray, t: float) -> np.ndarray:
    omega = np.arccos(np.clip(np.dot(p, q), -1.0, 1.0))
    if omega < 1e-9:
        return p.copy()
    return (np.sin((1 - t) * omega) * p + np.sin(t * omega) * q) / np.sin(omega)


def _build_standard_1010() -> dict[str, np.ndarray]:
    pos: dict[str, np.ndarray] = {}
    # Midline, nasion (front) to inion (back), 18 degree (10 %) steps.
    for name, theta in [("Nz", 90), ("Fpz", 72), ("AFz", 54), ("Fz", 36), ("FCz", 18), ("Cz", 0)]:
        pos[name] = _sph_to_cart(theta, 90)
    for name, theta in [("CPz", 18), ("Pz", 36), ("POz", 54), ("Oz", 72), ("Iz", 90)]:
        pos[name] = _sph_to_cart(theta, -90)
    # 10 % circumference ring (theta = 72) and the ring below it (theta = 90).
    ring_right = [("Fp2", 72), ("AF8", 54), ("F8", 36), ("FT8", 18), ("T8", 0),
                  ("TP8", -18), ("P8", -36), ("PO8", -54), ("O2", -72)]
    lower_right = [("AF10", 54), ("F10", 36), ("FT10", 18), ("T10", 0), ("TP10", -18),
                   ("P10", -36), ("PO10", -54), ("O10", -72)]
    for name, phi in ring_right:
        pos[name] = _sph_to_cart(72, phi)
        left = _left_name(name)
        pos[left] = _sph_to_cart(72, 180 - phi)
    for name, phi in lower_right:
        pos[name] = _sph_to_cart(90, phi)
        pos[_left_name(name)] = _sph_to_cart(90, 180 - phi)
    # Inner rows: ring electrode -> midline split into quarters.
    rows = [("AF", "AFz", "AF7", "AF8"), ("F", "Fz", "F7", "F8"), ("FC", "FCz", "FT7", "FT8"),
            ("C", "Cz", "T7", "T8"), ("CP", "CPz", "TP7", "TP8"), ("P", "Pz", "P7", "P8"),
            ("PO", "POz", "PO7", "PO8")]
    for prefix, mid, left_ring, right_ring in rows:
        for numbers, ring in (((5, 3, 1), left_ring), ((6, 4, 2), right_ring)):
            for k, num in enumerate(numbers, start=1):
                pos[f"{prefix}{num}"] = _slerp(pos[ring], pos[mid], k / 4)
    # Reference sites (approximate): earlobes and mastoids below T9/T10.
    pos["A1"] = _sph_to_cart(108, 180)
    pos["A2"] = _sph_to_cart(108, 0)
    pos["M1"] = _sph_to_cart(110, 200)
    pos["M2"] = _sph_to_cart(110, -20)
    return {k: v / np.linalg.norm(v) for k, v in pos.items()}


def _left_name(right_name: str) -> str:
    """Mirror an even-numbered (right) electrode name to its odd (left) twin."""
    m = re.match(r"^([A-Za-z]+)(\d+)$", right_name)
    prefix, num = m.group(1), int(m.group(2))
    return f"{prefix}{num - 1}"


_STANDARD_1010 = _build_standard_1010()

STANDARD_1020_NAMES = ["Fp1", "Fpz", "Fp2", "F7", "F3", "Fz", "F4", "F8", "T7", "C3", "Cz",
                       "C4", "T8", "P7", "P3", "Pz", "P4", "P8", "O1", "Oz", "O2", "A1", "A2"]
# The classic 19-channel clinical set.
CLINICAL_19 = ["Fp1", "Fp2", "F7", "F3", "Fz", "F4", "F8", "T7", "C3", "Cz", "C4", "T8",
               "P7", "P3", "Pz", "P4", "P8", "O1", "O2"]

STANDARD_NAMES = sorted(_STANDARD_1010)
_LOOKUP = {n.lower(): n for n in STANDARD_NAMES}
_LOOKUP.update(ALIASES)


def standard_name(name: str) -> str | None:
    """Return the canonical 10-10 spelling of ``name`` or ``None`` if unknown."""
    key = str(name).strip().lower()
    if key in _LOOKUP:
        return _LOOKUP[key]
    return None


def _position_key(name: str) -> str | None:
    """Canonical name used to look a label up in the standard layout."""
    std = standard_name(name)
    if std is not None:
        return std
    from .channels import clean_channel_name  # local import avoids a cycle

    std = standard_name(clean_channel_name(name, standardize=False))
    return std


def fit_sphere(points: np.ndarray) -> tuple[np.ndarray, float]:
    """Least-squares sphere fit. Returns (center, radius)."""
    pts = np.asarray(points, dtype=float)
    if pts.shape[0] < 4:
        center = pts.mean(axis=0)
        radius = float(np.mean(np.linalg.norm(pts - center, axis=1))) or 1.0
        return center, radius
    A = np.column_stack([2 * pts, np.ones(len(pts))])
    b = (pts ** 2).sum(axis=1)
    sol, *_ = np.linalg.lstsq(A, b, rcond=None)
    center = sol[:3]
    radius = float(np.sqrt(max(sol[3] + center @ center, 1e-12)))
    return center, radius


def project_to_2d(xyz: np.ndarray) -> np.ndarray:
    """Azimuthal equidistant projection used for topographic maps.

    Cz maps to the origin and the head's equator (ear level) maps to radius 1,
    so the 10-20 perimeter electrodes sit at radius 0.8.
    """
    xyz = np.atleast_2d(np.asarray(xyz, dtype=float))
    norm = np.linalg.norm(xyz, axis=1, keepdims=True)
    norm[norm == 0] = 1.0
    u = xyz / norm
    theta = np.arccos(np.clip(u[:, 2], -1, 1))
    phi = np.arctan2(u[:, 1], u[:, 0])
    r = theta / (np.pi / 2)
    return np.column_stack([r * np.cos(phi), r * np.sin(phi)])


def unproject_from_2d(xy: np.ndarray) -> np.ndarray:
    """Inverse of :func:`project_to_2d`."""
    xy = np.atleast_2d(np.asarray(xy, dtype=float))
    r = np.hypot(xy[:, 0], xy[:, 1])
    phi = np.arctan2(xy[:, 1], xy[:, 0])
    theta = r * (np.pi / 2)
    return np.column_stack([np.sin(theta) * np.cos(phi), np.sin(theta) * np.sin(phi), np.cos(theta)])


class Montage:
    """A named set of electrode positions on the unit sphere."""

    def __init__(self, positions: dict[str, Iterable[float]], name: str = "custom",
                 normalize: bool = True):
        names = list(positions)
        xyz = np.array([np.asarray(positions[n], dtype=float) for n in names]) if names else np.zeros((0, 3))
        if normalize and len(names):
            if len(names) >= 4:
                center, _ = fit_sphere(xyz)
                # Only recentre when the points are clearly not centred already.
                if np.linalg.norm(center) > 0.05 * np.median(np.linalg.norm(xyz, axis=1)):
                    xyz = xyz - center
            norms = np.linalg.norm(xyz, axis=1, keepdims=True)
            norms[norms == 0] = 1.0
            xyz = xyz / norms
        self.name = name
        self._pos = {n: xyz[i] for i, n in enumerate(names)}
        self._lower = {n.lower(): n for n in names}

    # ------------------------------------------------------------------ access
    @property
    def ch_names(self) -> list[str]:
        return list(self._pos)

    def __len__(self) -> int:
        return len(self._pos)

    def __contains__(self, name: str) -> bool:
        return self._resolve(name) is not None

    def __repr__(self) -> str:
        return f"<Montage '{self.name}' | {len(self)} electrodes>"

    def _resolve(self, name: str) -> str | None:
        if name in self._pos:
            return name
        low = str(name).strip().lower()
        if low in self._lower:
            return self._lower[low]
        key = _position_key(name)
        if key is not None:
            if key in self._pos:
                return key
            if key.lower() in self._lower:
                return self._lower[key.lower()]
        return None

    def get(self, name: str) -> np.ndarray | None:
        key = self._resolve(name)
        return None if key is None else self._pos[key].copy()

    def positions_for(self, names: Iterable[str]) -> dict[str, np.ndarray]:
        """Positions for the given channel names (missing ones are skipped)."""
        out = {}
        for n in names:
            p = self.get(n)
            if p is not None:
                out[n] = p
        return out

    def as_array(self, names: Iterable[str] | None = None) -> np.ndarray:
        names = self.ch_names if names is None else list(names)
        return np.array([self.get(n) if n in self else np.full(3, np.nan) for n in names])

    def project(self, names: Iterable[str] | None = None) -> np.ndarray:
        return project_to_2d(self.as_array(names))

    # ------------------------------------------------------------ constructors
    @classmethod
    def standard(cls, kind: str = "standard_1010") -> "Montage":
        """Standard montage: ``'standard_1020'`` or ``'standard_1010'``."""
        kind = kind.lower().replace("-", "_")
        if kind in ("standard_1010", "1010", "10-10", "standard_1005"):
            return cls(dict(_STANDARD_1010), name="standard_1010", normalize=False)
        if kind in ("standard_1020", "1020", "10-20"):
            pos = {n: _STANDARD_1010[n] for n in STANDARD_1020_NAMES}
            return cls(pos, name="standard_1020", normalize=False)
        raise ValueError(f"Unknown standard montage {kind!r}; use 'standard_1020' or 'standard_1010'")

    @classmethod
    def from_file(cls, path: str | os.PathLike) -> "Montage":
        """Read electrode positions from a file.

        Supported: ``.elc`` (ASA), ``.sfp`` (BESA/EGI), ``.loc``/``.locs``
        (EEGLAB polar), ``.ced`` (EEGLAB), ``.csv``/``.tsv``/``.txt`` with
        columns ``name, x, y, z`` (a header row is optional), and BIDS
        ``*_electrodes.tsv`` files.
        """
        path = os.fspath(path)
        ext = os.path.splitext(path)[1].lower()
        with open(path, "r", encoding="utf-8", errors="replace") as f:
            text = f.read()
        if ext == ".elc":
            positions = _parse_elc(text)
        elif ext in (".loc", ".locs"):
            positions = _parse_eeglab_loc(text)
        elif ext == ".ced":
            positions = _parse_ced(text)
        else:
            positions = _parse_xyz_table(text)
        if not positions:
            raise ValueError(f"No electrode positions found in {path}")
        return cls(positions, name=os.path.basename(path))


def make_standard_montage(kind: str = "standard_1010") -> Montage:
    return Montage.standard(kind)


def get_montage(montage) -> Montage:
    """Accept a Montage, a standard name, a file path or a dict of positions."""
    if isinstance(montage, Montage):
        return montage
    if isinstance(montage, dict):
        return Montage(montage)
    if isinstance(montage, (str, os.PathLike)):
        s = os.fspath(montage)
        if os.path.exists(s):
            return Montage.from_file(s)
        return Montage.standard(s)
    raise TypeError(f"Cannot interpret {montage!r} as a montage")


def find_neighbors(xyz: np.ndarray, n_neighbors: int = 4, max_angle_deg: float | None = None) -> list[np.ndarray]:
    """Indices of the nearest electrodes on the sphere for each electrode."""
    xyz = np.asarray(xyz, dtype=float)
    u = xyz / np.linalg.norm(xyz, axis=1, keepdims=True)
    ang = np.arccos(np.clip(u @ u.T, -1, 1))
    np.fill_diagonal(ang, np.inf)
    out = []
    for i in range(len(u)):
        order = np.argsort(ang[i])
        k = min(n_neighbors, len(u) - 1)
        idx = order[:k]
        if max_angle_deg is not None:
            idx = idx[ang[i, idx] <= np.deg2rad(max_angle_deg)]
        out.append(idx)
    return out


# ---------------------------------------------------------------- file parsers
def _parse_elc(text: str) -> dict[str, np.ndarray]:
    lines = [ln.strip() for ln in text.splitlines()]
    pos_lines, labels, mode = [], [], None
    for ln in lines:
        low = ln.lower()
        if low.startswith("positions"):
            mode = "pos"
            continue
        if low.startswith("labels"):
            mode = "lab"
            continue
        if not ln or ln.startswith("#") or "=" in ln:
            continue
        if mode == "pos":
            if ":" in ln:  # "Fp1: x y z"
                name, rest = ln.split(":", 1)
                vals = rest.split()
                pos_lines.append((name.strip(), vals))
            else:
                pos_lines.append((None, ln.split()))
        elif mode == "lab":
            labels.extend(ln.split())
    out = {}
    for i, (name, vals) in enumerate(pos_lines):
        name = name or (labels[i] if i < len(labels) else f"E{i + 1}")
        out[name] = np.array([float(v) for v in vals[:3]])
    return out


def _parse_eeglab_loc(text: str) -> dict[str, np.ndarray]:
    # index  theta(deg, 0 = nose, positive = right)  radius(0.5 = equator)  label
    out = {}
    for ln in text.splitlines():
        parts = ln.split()
        if len(parts) < 4:
            continue
        try:
            theta, radius = float(parts[1]), float(parts[2])
        except ValueError:
            continue
        out[parts[3]] = _eeglab_polar_to_cart(theta, radius)
    return out


def _eeglab_polar_to_cart(theta: float, radius: float) -> np.ndarray:
    # EEGLAB: theta measured from the nose, clockwise (towards the right ear);
    # radius 0.5 corresponds to 90 degrees from the vertex.
    elev = radius * 180.0
    azim = 90.0 - theta
    return _sph_to_cart(elev, azim)


def _parse_ced(text: str) -> dict[str, np.ndarray]:
    rows = [ln.split("\t") if "\t" in ln else ln.split() for ln in text.splitlines() if ln.strip()]
    if not rows:
        return {}
    header = [h.strip().lower() for h in rows[0]]
    out = {}
    for r in rows[1:]:
        rec = dict(zip(header, [c.strip() for c in r]))
        name = rec.get("labels") or rec.get("label")
        if not name:
            continue
        try:
            xyz = np.array([float(rec["x"]), float(rec["y"]), float(rec["z"])])
            # EEGLAB Cartesian: +x = nose, +y = left ear. Convert to ours.
            out[name] = np.array([-xyz[1], xyz[0], xyz[2]])
        except (KeyError, ValueError):
            try:
                out[name] = _eeglab_polar_to_cart(float(rec["theta"]), float(rec["radius"]))
            except (KeyError, ValueError):
                continue
    return out


def _parse_xyz_table(text: str) -> dict[str, np.ndarray]:
    sample = text[:4096]
    try:
        dialect = csv.Sniffer().sniff(sample, delimiters=",;\t ")
        delim = dialect.delimiter
    except csv.Error:
        delim = None
    rows = []
    for ln in text.splitlines():
        if not ln.strip() or ln.lstrip().startswith(("#", "%")):
            continue
        parts = [p.strip() for p in (ln.split(delim) if delim and delim != " " else ln.split())]
        rows.append([p for p in parts if p != ""])
    if not rows:
        return {}
    header = [h.lower() for h in rows[0]]
    body = rows[1:] if {"x", "y", "z"} <= set(header) else rows
    if {"x", "y", "z"} <= set(header):
        name_col = next((header.index(k) for k in ("name", "label", "labels", "channel", "electrode")
                         if k in header), 0)
        ix, iy, iz = header.index("x"), header.index("y"), header.index("z")
    else:
        name_col, ix, iy, iz = 0, 1, 2, 3
    out = {}
    for r in body:
        try:
            xyz = np.array([float(r[ix]), float(r[iy]), float(r[iz])])
        except (IndexError, ValueError):
            continue
        if np.all(np.isfinite(xyz)) and np.any(xyz != 0):
            out[r[name_col]] = xyz
    return out
