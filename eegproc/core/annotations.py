"""Time-stamped annotations (events, artifacts, sleep stages ...)."""

from __future__ import annotations

import re
from collections import Counter
from typing import Callable, Iterable

import numpy as np

# Descriptions starting with these prefixes mark data to exclude, not events.
BAD_PREFIXES = ("bad", "edge", "boundary")


class Annotations:
    """A list of ``(onset, duration, description)`` entries in seconds.

    Onsets are relative to the first sample of the recording.
    """

    def __init__(self, onset: Iterable[float] = (), duration: Iterable[float] | float = 0.0,
                 description: Iterable[str] | str = ()):
        onset = np.atleast_1d(np.asarray(onset, dtype=float)).ravel()
        if np.isscalar(duration) or np.ndim(duration) == 0:
            duration = np.full(onset.shape, float(duration))
        duration = np.atleast_1d(np.asarray(duration, dtype=float)).ravel()
        if isinstance(description, str):
            description = [description] * len(onset)
        description = [str(d) for d in description]
        if not (len(onset) == len(duration) == len(description)):
            raise ValueError("onset, duration and description must have the same length")
        self.onset = onset
        self.duration = np.maximum(duration, 0.0)
        self.description = description

    # ----------------------------------------------------------------- basics
    def __len__(self) -> int:
        return len(self.onset)

    def __iter__(self):
        for o, d, s in zip(self.onset, self.duration, self.description):
            yield {"onset": float(o), "duration": float(d), "description": s}

    def __getitem__(self, key) -> "Annotations":
        idx = np.arange(len(self))[key]
        idx = np.atleast_1d(idx)
        return Annotations(self.onset[idx], self.duration[idx], [self.description[i] for i in idx])

    def __repr__(self) -> str:
        counts = ", ".join(f"{k}: {v}" for k, v in list(self.count().items())[:6])
        more = " ..." if len(self.count()) > 6 else ""
        return f"<Annotations | {len(self)} entries ({counts}{more})>"

    def copy(self) -> "Annotations":
        return Annotations(self.onset.copy(), self.duration.copy(), list(self.description))

    def count(self) -> dict[str, int]:
        return dict(sorted(Counter(self.description).items()))

    @property
    def unique_descriptions(self) -> list[str]:
        return sorted(set(self.description))

    # ---------------------------------------------------------------- editing
    def append(self, onset, duration, description) -> "Annotations":
        """Add entries in place (scalars or sequences). Returns ``self``."""
        onset = np.atleast_1d(np.asarray(onset, dtype=float))
        duration = np.broadcast_to(np.asarray(duration, dtype=float), onset.shape)
        if isinstance(description, str):
            description = [description] * len(onset)
        self.onset = np.concatenate([self.onset, onset])
        self.duration = np.concatenate([self.duration, np.maximum(duration, 0)])
        self.description = self.description + [str(d) for d in description]
        return self

    def extend(self, other: "Annotations") -> "Annotations":
        return self.append(other.onset, other.duration, other.description)

    def sorted(self) -> "Annotations":
        order = np.argsort(self.onset, kind="stable")
        return self[order]

    def select(self, match: str | Iterable[str] | Callable[[str], bool] | None = None,
               regexp: str | None = None) -> "Annotations":
        """Keep entries whose description matches."""
        if match is None and regexp is None:
            return self.copy()
        if regexp is not None:
            pat = re.compile(regexp)
            keep = [bool(pat.search(d)) for d in self.description]
        elif callable(match):
            keep = [bool(match(d)) for d in self.description]
        else:
            wanted = {match} if isinstance(match, str) else set(match)
            keep = [d in wanted for d in self.description]
        return self[np.flatnonzero(keep)]

    def drop_bad(self) -> "Annotations":
        return self.select(lambda d: not is_bad_description(d))

    def shift(self, dt: float) -> "Annotations":
        out = self.copy()
        out.onset = out.onset + dt
        return out

    def crop(self, tmin: float = 0.0, tmax: float | None = None, shift: bool = True) -> "Annotations":
        """Keep entries overlapping ``[tmin, tmax]``, clipped to that window."""
        tmax = np.inf if tmax is None else tmax
        ends = self.onset + self.duration
        keep = (ends >= tmin) & (self.onset <= tmax)
        out = self[np.flatnonzero(keep)]
        new_on = np.clip(out.onset, tmin, tmax)
        new_end = np.clip(out.onset + out.duration, tmin, tmax)
        out.onset, out.duration = new_on, new_end - new_on
        if shift:
            out.onset = out.onset - tmin
        return out

    # ---------------------------------------------------------------- events
    def to_events(self, sfreq: float, event_id: dict[str, int] | None = None,
                  include_bad: bool = False) -> tuple[np.ndarray, dict[str, int]]:
        """Convert to an ``(n_events, 3)`` array ``[sample, 0, code]``.

        When ``event_id`` is omitted every distinct description gets a code
        (numeric descriptions keep their numeric value when possible).
        """
        ann = self if include_bad else self.drop_bad()
        if event_id is None:
            event_id = auto_event_id(ann.unique_descriptions)
        rows = [(int(round(o * sfreq)), 0, event_id[d]) for o, d in zip(ann.onset, ann.description)
                if d in event_id]
        events = np.array(sorted(rows), dtype=int).reshape(-1, 3)
        used = {d: c for d, c in event_id.items() if c in set(events[:, 2].tolist())}
        return events, used

    @classmethod
    def from_events(cls, events: np.ndarray, sfreq: float,
                    event_desc: dict[int, str] | None = None) -> "Annotations":
        events = np.asarray(events).reshape(-1, 3)
        event_desc = event_desc or {}
        desc = [event_desc.get(int(c), str(int(c))) for c in events[:, 2]]
        return cls(events[:, 0] / float(sfreq), 0.0, desc)

    def bad_mask(self, n_samples: int, sfreq: float) -> np.ndarray:
        """Boolean mask of samples covered by ``BAD*``/``EDGE``/``boundary`` entries."""
        mask = np.zeros(n_samples, dtype=bool)
        for o, d, s in zip(self.onset, self.duration, self.description):
            if is_bad_description(s):
                a = max(0, int(np.floor(o * sfreq)))
                b = min(n_samples, int(np.ceil((o + d) * sfreq)) or a + 1)
                mask[a:max(b, a + 1)] = True
        return mask

    def to_list(self) -> list[dict]:
        return list(self)


def is_bad_description(desc: str) -> bool:
    return str(desc).strip().lower().startswith(BAD_PREFIXES)


def auto_event_id(descriptions: Iterable[str]) -> dict[str, int]:
    """Give each description an integer code, keeping numeric labels when possible."""
    descriptions = sorted(set(descriptions))
    event_id: dict[str, int] = {}
    used: set[int] = set()
    for d in descriptions:
        m = re.search(r"(-?\d+)\s*$", d)
        if m:
            code = int(m.group(1))
            if code > 0 and code not in used:
                event_id[d] = code
                used.add(code)
    nxt = 1
    for d in descriptions:
        if d in event_id:
            continue
        while nxt in used:
            nxt += 1
        event_id[d] = nxt
        used.add(nxt)
    return event_id
