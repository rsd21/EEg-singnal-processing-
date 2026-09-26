"""Shared helpers: logging, validation, robust statistics and text tables."""

from __future__ import annotations

import logging
import numbers
from typing import Iterable, Sequence

import numpy as np

logger = logging.getLogger("eegproc")
if not logger.handlers:
    logger.addHandler(logging.NullHandler())


def set_log_level(level: str | int = "INFO") -> None:
    """Send eegproc log messages to stderr at the given level."""
    if isinstance(level, str):
        level = getattr(logging, level.upper())
    logger.setLevel(level)
    if not any(isinstance(h, logging.StreamHandler) and not isinstance(h, logging.NullHandler)
               for h in logger.handlers):
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter("[eegproc] %(levelname)s: %(message)s"))
        logger.addHandler(handler)


def check_random_state(seed=None) -> np.random.Generator:
    """Turn ``None``, an int or a Generator into a ``numpy.random.Generator``."""
    if isinstance(seed, np.random.Generator):
        return seed
    if seed is None or isinstance(seed, numbers.Integral):
        return np.random.default_rng(seed)
    raise TypeError(f"Cannot use {seed!r} as a random seed")


def ensure_list(x) -> list:
    if x is None:
        return []
    if isinstance(x, (str, bytes)) or not isinstance(x, Iterable):
        return [x]
    return list(x)


def mad(x: np.ndarray, axis=None, scale: float = 1.4826) -> np.ndarray:
    """Median absolute deviation, scaled to match the std of a Gaussian."""
    x = np.asarray(x, dtype=float)
    med = np.median(x, axis=axis, keepdims=True)
    out = np.median(np.abs(x - med), axis=axis)
    return scale * out


def robust_zscore(x: np.ndarray) -> np.ndarray:
    """Z-score using median and MAD so a few outliers cannot hide themselves."""
    x = np.asarray(x, dtype=float)
    finite = np.isfinite(x)
    if not finite.any():
        return np.zeros_like(x)
    med = np.median(x[finite])
    spread = mad(x[finite])
    if spread <= np.finfo(float).eps:
        spread = np.std(x[finite])
    if spread <= np.finfo(float).eps:
        return np.zeros_like(x)
    return (x - med) / spread


def window_starts(n_samples: int, win: int, step: int | None = None) -> np.ndarray:
    """Start indices of full windows of length ``win`` stepping by ``step``."""
    step = win if step is None else step
    if win <= 0 or step <= 0:
        raise ValueError("window and step must be positive")
    if n_samples < win:
        return np.zeros(0, dtype=int)
    return np.arange(0, n_samples - win + 1, step, dtype=int)


def sliding_windows(x: np.ndarray, win: int, step: int | None = None) -> np.ndarray:
    """View ``x`` (..., n_samples) as (..., n_windows, win) without copying."""
    step = win if step is None else step
    starts = window_starts(x.shape[-1], win, step)
    if starts.size == 0:
        return np.empty(x.shape[:-1] + (0, win), dtype=x.dtype)
    view = np.lib.stride_tricks.sliding_window_view(x, win, axis=-1)
    return view[..., ::step, :][..., : starts.size, :]


def format_value(v, floatfmt: str = "{:.4g}") -> str:
    if v is None:
        return ""
    if isinstance(v, (bool, np.bool_)):
        return "yes" if v else "no"
    if isinstance(v, numbers.Integral):
        return str(int(v))
    if isinstance(v, numbers.Real):
        if not np.isfinite(v):
            return "nan" if np.isnan(v) else ("inf" if v > 0 else "-inf")
        return floatfmt.format(v)
    if isinstance(v, (list, tuple)):
        return "; ".join(format_value(i, floatfmt) for i in v)
    return str(v)


def format_table(rows: Sequence[dict], columns: Sequence[str] | None = None,
                 floatfmt: str = "{:.4g}", max_rows: int | None = None,
                 max_width: int = 60) -> str:
    """Render a list of dicts as an aligned plain-text table."""
    if not rows:
        return "(empty)"
    columns = list(columns) if columns is not None else list(rows[0].keys())
    shown = rows if max_rows is None else rows[:max_rows]
    cells = [[format_value(r.get(c), floatfmt)[:max_width] for c in columns] for r in shown]
    widths = [max(len(str(c)), *(len(row[i]) for row in cells)) for i, c in enumerate(columns)]
    lines = ["  ".join(str(c).ljust(w) for c, w in zip(columns, widths)),
             "  ".join("-" * w for w in widths)]
    lines += ["  ".join(v.ljust(w) for v, w in zip(row, widths)) for row in cells]
    if max_rows is not None and len(rows) > max_rows:
        lines.append(f"... ({len(rows) - max_rows} more rows)")
    return "\n".join(lines)


def write_csv(rows: Sequence[dict], path, columns: Sequence[str] | None = None) -> None:
    """Write a list of dicts to CSV using only the standard library."""
    import csv

    columns = list(columns) if columns is not None else (list(rows[0].keys()) if rows else [])
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(columns)
        for r in rows:
            writer.writerow([format_value(r.get(c), "{:.6g}") for c in columns])


def to_dataframe(rows: Sequence[dict], columns: Sequence[str] | None = None):
    """Convert rows to a pandas DataFrame (pandas is an optional dependency)."""
    try:
        import pandas as pd
    except ImportError as err:  # pragma: no cover - depends on environment
        raise ImportError("pandas is required for to_dataframe(); pip install pandas") from err
    return pd.DataFrame(list(rows), columns=columns)


def nextpow2(n: int) -> int:
    return 1 << int(np.ceil(np.log2(max(1, n))))
