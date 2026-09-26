"""Realistic synthetic EEG with known ground truth, for demos and tests.

The simulation mixes:

* spatially correlated 1/f ("pink") background from many cortical-like
  sources smoothed over the scalp,
* posterior alpha (~10 Hz) with waxing and waning amplitude,
* central mu (~11 Hz) and beta (~20 Hz) rhythms over C3/C4,
* eye blinks strongest at Fp1/Fp2 (plus optional EOG channels),
* power-line interference and sensor noise,
* task effects: an auditory ``'oddball'`` (N1/P2 + 15 µV P300 at Pz for targets) or
  ``'motor_imagery'`` (contralateral mu/beta desynchronisation),
* bad channels of known kinds (flat, noisy, drift, line, pop,
  disconnected, clipping, intermittent).
"""

from __future__ import annotations

import numpy as np

from ..core.annotations import Annotations
from ..core.montage import CLINICAL_19, Montage, _sph_to_cart
from ..core.raw import RawEEG
from ..utils import check_random_state

BAD_KINDS = ("flat", "noisy", "drift", "line", "pop", "disconnected", "clipping", "intermittent")
DEFAULT_BADS = {"T8": "noisy", "O2": "flat", "F7": "drift", "P4": "line"}

MONTAGES = {
    "clinical19": CLINICAL_19,
    "standard_1020": ["Fp1", "Fpz", "Fp2", "F7", "F3", "Fz", "F4", "F8", "T7", "C3", "Cz", "C4",
                      "T8", "P7", "P3", "Pz", "P4", "P8", "O1", "Oz", "O2"],
    "32": ["Fp1", "Fp2", "AF3", "AF4", "F7", "F3", "Fz", "F4", "F8", "FC5", "FC1", "FC2", "FC6",
           "T7", "C3", "Cz", "C4", "T8", "CP5", "CP1", "CP2", "CP6", "P7", "P3", "Pz", "P4", "P8",
           "PO3", "PO4", "O1", "Oz", "O2"],
    "64": ["Fp1", "Fpz", "Fp2", "AF7", "AF3", "AFz", "AF4", "AF8", "F7", "F5", "F3", "F1", "Fz",
           "F2", "F4", "F6", "F8", "FT7", "FC5", "FC3", "FC1", "FCz", "FC2", "FC4", "FC6", "FT8",
           "T7", "C5", "C3", "C1", "Cz", "C2", "C4", "C6", "T8", "TP7", "CP5", "CP3", "CP1",
           "CPz", "CP2", "CP4", "CP6", "TP8", "P7", "P5", "P3", "P1", "Pz", "P2", "P4", "P6",
           "P8", "PO7", "PO3", "POz", "PO4", "PO8", "O1", "Oz", "O2", "Iz", "TP9", "TP10"],
}


def _colored_noise(n: int, n_samples: int, sfreq: float, exponent: float, rng,
                   knee: float = 0.5) -> np.ndarray:
    """Noise with power ~ 1 / (knee^exponent + f^exponent): 1/f above the knee, flat below."""
    freqs = np.fft.rfftfreq(n_samples, 1.0 / sfreq)
    spec = rng.standard_normal((n, freqs.size)) + 1j * rng.standard_normal((n, freqs.size))
    scale = 1.0 / np.sqrt(knee ** exponent + freqs ** exponent)
    scale[0] = 0.0
    x = np.fft.irfft(spec * scale, n=n_samples, axis=-1)
    return x / (x.std(axis=-1, keepdims=True) + 1e-12)


def _oscillation(n: int, n_samples: int, sfreq: float, f0: float, bw: float, rng) -> np.ndarray:
    freqs = np.fft.rfftfreq(n_samples, 1.0 / sfreq)
    spec = rng.standard_normal((n, freqs.size)) + 1j * rng.standard_normal((n, freqs.size))
    x = np.fft.irfft(spec * np.exp(-((freqs - f0) ** 2) / (2 * bw ** 2)), n=n_samples, axis=-1)
    return x / (x.std(axis=-1, keepdims=True) + 1e-12)


def _slow_envelope(n_samples: int, sfreq: float, rng, depth: float = 0.6) -> np.ndarray:
    env = _colored_noise(1, n_samples, sfreq, 4.0, rng)[0]
    return np.exp(depth * env) / np.exp(depth * env).mean()


def _spatial(pos: np.ndarray, center: np.ndarray, width: float) -> np.ndarray:
    ang = np.arccos(np.clip(pos @ (center / np.linalg.norm(center)), -1, 1))
    return np.exp(-(ang / width) ** 2)


def simulate_eeg(duration: float = 60.0, sfreq: float = 256.0, montage: str | list[str] = "clinical19",
                 task: str | None = None, bad_channels: dict[str, str] | str | None = None,
                 line_freq: float = 50.0, line_amplitude: float = 1.5, blinks: bool = True,
                 eog: bool = False, ecg: bool = False, stim_channel: bool = False,
                 background_amplitude: float = 10.0, alpha_amplitude: float = 10.0,
                 n_trials: int | None = None, seed=None) -> RawEEG:
    """Simulate a multichannel EEG recording.

    Parameters
    ----------
    duration, sfreq : float
        Length (s) and sampling rate (Hz).
    montage : str or list
        ``'clinical19'``, ``'standard_1020'``, ``'32'``, ``'64'`` or a list of
        10-10 channel names.
    task : None, ``'oddball'`` or ``'motor_imagery'``
        Adds events (annotations ``standard``/``target`` or ``left``/``right``)
        and the matching brain responses.
    bad_channels : dict, ``'default'`` or None
        ``{channel: kind}`` with kind in :data:`BAD_KINDS`. The truth is stored
        in ``raw.meta['ground_truth_bads']``.
    eog, ecg, stim_channel : bool
        Add VEOG/HEOG, ECG and a trigger channel.
    """
    rng = check_random_state(seed)
    names = list(MONTAGES[montage]) if isinstance(montage, str) else list(montage)
    mont = Montage.standard("standard_1010")
    missing = [n for n in names if n not in mont]
    if missing:
        raise ValueError(f"Unknown electrode names {missing}")
    pos = np.array([mont.get(n) for n in names])
    n_ch = len(names)
    n = int(round(duration * sfreq))
    t = np.arange(n) / sfreq

    # --- background: many sources on the upper hemisphere, smoothly projected
    n_src = max(40, 2 * n_ch)
    theta = np.degrees(np.arccos(rng.uniform(0.0, 1.0, n_src)))  # uniform on the upper hemisphere
    phi = rng.uniform(0, 360, n_src)
    src_pos = np.array([_sph_to_cart(a, b) for a, b in zip(theta, phi)])
    ang = np.arccos(np.clip(pos @ src_pos.T, -1, 1))
    gain = np.exp(-(ang / 0.8) ** 2)  # volume conduction smears sources widely
    background = gain @ _colored_noise(n_src, n, sfreq, 1.6, rng, knee=1.0)
    background *= background_amplitude / background.std(axis=1, keepdims=True).mean()
    data = background

    # --- posterior alpha
    posterior = _sph_to_cart(60, -90)
    iaf = rng.uniform(9.5, 10.5)
    alpha = _oscillation(1, n, sfreq, iaf, 0.6, rng)[0] * _slow_envelope(n, sfreq, rng)
    data = data + alpha_amplitude * np.outer(_spatial(pos, posterior, 0.6), alpha)

    # --- sensorimotor mu (11 Hz) and beta (20 Hz), one generator per hemisphere
    c3, c4 = mont.get("C3"), mont.get("C4")
    mu_sources = {}
    for side, center in (("left", c3), ("right", c4)):
        mu = _oscillation(1, n, sfreq, 11.0, 0.8, rng)[0] * _slow_envelope(n, sfreq, rng, 0.4)
        beta = _oscillation(1, n, sfreq, 20.0, 1.5, rng)[0]
        mu_sources[side] = (center, 6.0 * mu, 2.5 * beta)

    # --- task
    ann = Annotations()
    stim = np.zeros(n)
    erd = {"left": np.ones(n), "right": np.ones(n)}  # mu envelopes per hemisphere
    if task == "oddball":
        isi = 1.2
        count = n_trials or int((duration - 1.5) / isi)
        onsets = 0.8 + np.arange(count) * isi + rng.uniform(-0.15, 0.15, count)
        onsets = onsets[onsets < duration - 1.0]
        is_target = rng.random(len(onsets)) < 0.2
        cz, pz = mont.get("Cz"), mont.get("Pz")
        n1_map = _spatial(pos, cz, 0.7)
        p3_map = _spatial(pos, pz, 0.6)
        for on, tgt in zip(onsets, is_target):
            k = int(round(on * sfreq))
            tt = np.arange(0, int(0.8 * sfreq)) / sfreq
            seg = slice(k, min(n, k + len(tt)))
            tt = tt[: seg.stop - seg.start]
            n1 = -5.0 * np.exp(-((tt - 0.10) / 0.025) ** 2)
            p2 = 3.0 * np.exp(-((tt - 0.20) / 0.035) ** 2)
            data[:, seg] += np.outer(n1_map, n1 + p2)
            if tgt:
                p3 = 15.0 * np.exp(-((tt - 0.35) / 0.1) ** 2)
                data[:, seg] += np.outer(p3_map, p3)
            ann.append(on, 0.0, "target" if tgt else "standard")
            stim[k: k + max(1, int(0.01 * sfreq))] = 2 if tgt else 1
    elif task == "motor_imagery":
        # 4 s of imagery per trial, random 6-8 s inter-trial intervals (as in real paradigms;
        # strictly periodic trials would alias slow background fluctuations into the averages)
        count = n_trials or int((duration - 2) / 7.0)
        itis = 6.0 + rng.uniform(0.0, 2.0, count)
        onsets = 1.0 + np.concatenate([[0.0], np.cumsum(itis[:-1])])
        onsets = onsets[onsets < duration - 4.5]
        classes = rng.permutation(np.resize(["left", "right"], len(onsets)))
        for on, cls in zip(onsets, classes):
            # imagining the LEFT hand desynchronises the RIGHT hemisphere (C4) and vice versa
            hemi = "right" if cls == "left" else "left"
            a, b = int(round((on + 0.5) * sfreq)), int(round((on + 4.0) * sfreq))
            ramp = np.ones(b - a)
            r = int(0.3 * sfreq)
            ramp[:r] = np.linspace(0, 1, r)
            ramp[-r:] = np.linspace(1, 0, r)
            erd[hemi][a:b] *= 1 - 0.7 * ramp
            ann.append(on, 4.0, cls)
            stim[int(round(on * sfreq)): int(round(on * sfreq)) + max(1, int(0.01 * sfreq))] = \
                1 if cls == "left" else 2
    elif task not in (None, "rest"):
        raise ValueError("task must be None, 'rest', 'oddball' or 'motor_imagery'")

    for side, (center, mu, beta) in mu_sources.items():
        data = data + np.outer(_spatial(pos, center, 0.5), (mu + beta) * erd[side])

    # --- blinks
    blink_wave = np.zeros(n)
    if blinks:
        tb = 1.0 + rng.exponential(3.5)
        while tb < duration - 0.5:
            k = int(tb * sfreq)
            L = int(0.4 * sfreq)
            shape = np.sin(np.linspace(0, np.pi, L)) ** 2
            amp = rng.uniform(90, 160)
            blink_wave[k: k + L] += amp * shape[: max(0, min(L, n - k))]
            tb += 1.0 + rng.exponential(3.5)
        eyes = np.array([0.0, 1.0, -0.35])
        eye_map = np.exp(-np.arccos(np.clip(pos @ (eyes / np.linalg.norm(eyes)), -1, 1)) / 0.35)
        eye_map /= eye_map.max()
        data = data + np.outer(eye_map, blink_wave)

    # --- line noise and sensor noise
    phase = rng.uniform(0, 2 * np.pi, n_ch)
    amps = line_amplitude * rng.uniform(0.5, 1.5, n_ch)
    data = data + amps[:, None] * np.sin(2 * np.pi * line_freq * t[None, :] + phase[:, None])
    data = data + 0.5 * rng.standard_normal((n_ch, n))

    # --- bad channels
    if bad_channels == "default":
        bad_channels = {k: v for k, v in DEFAULT_BADS.items() if k in names}
    bad_channels = dict(bad_channels or {})
    for name, kind in bad_channels.items():
        if name not in names:
            raise ValueError(f"Bad channel {name!r} is not in the montage")
        i = names.index(name)
        data[i] = _make_bad(data[i], kind, sfreq, line_freq, rng)

    # --- extra channels
    ch_names, ch_types = list(names), ["eeg"] * n_ch
    extra = []
    if eog:
        extra.append(("VEOG", "eog", 1.2 * blink_wave + 5 * _colored_noise(1, n, sfreq, 1.5, rng)[0]))
        sacc = np.cumsum(np.where(rng.random(n) < 0.3 / sfreq, rng.normal(0, 40, n), 0.0))
        sacc -= np.convolve(sacc, np.ones(int(2 * sfreq)) / int(2 * sfreq), mode="same")
        extra.append(("HEOG", "eog", sacc + 5 * _colored_noise(1, n, sfreq, 1.5, rng)[0]))
    if ecg:
        hr = rng.uniform(60, 80) / 60.0
        base = np.arange(0.3, duration, 1 / hr)
        beats = base + rng.normal(0, 0.02, base.size)
        ecg_sig = np.zeros(n)
        for b in beats:
            k = int(b * sfreq)
            tt = (np.arange(-int(0.05 * sfreq), int(0.05 * sfreq)) / sfreq)
            qrs = 800 * np.exp(-(tt / 0.012) ** 2) - 150 * np.exp(-((tt - 0.025) / 0.01) ** 2)
            lo, hi = max(0, k - len(tt) // 2), min(n, k - len(tt) // 2 + len(tt))
            ecg_sig[lo:hi] += qrs[: hi - lo]
        extra.append(("ECG", "ecg", ecg_sig + 10 * rng.standard_normal(n)))
        data = data + 0.004 * np.outer(np.ones(n_ch), ecg_sig)
    if stim_channel:
        extra.append(("STI", "stim", stim))
    if extra:
        data = np.vstack([data] + [e[2][None, :] for e in extra])
        ch_names += [e[0] for e in extra]
        ch_types += [e[1] for e in extra]

    positions = {name: p for name, p in zip(names, pos)}
    meta = {"format": "simulated", "task": task, "ground_truth_bads": bad_channels,
            "line_freq": line_freq, "iaf": iaf}
    raw = RawEEG(data, sfreq, ch_names, ch_types, annotations=ann, positions=positions, meta=meta)
    raw.meta["montage"] = "standard_1010"
    return raw


def _make_bad(x: np.ndarray, kind: str, sfreq: float, line_freq: float, rng) -> np.ndarray:
    n = len(x)
    t = np.arange(n) / sfreq
    if kind == "flat":
        return 0.05 * rng.standard_normal(n)
    if kind == "noisy":
        return x + 30.0 * rng.standard_normal(n)
    if kind == "drift":
        walk = np.cumsum(rng.standard_normal(n))
        walk = walk - np.convolve(walk, np.ones(int(30 * sfreq)) / int(30 * sfreq), mode="same")
        walk = walk / walk.std() * 150.0
        return x + walk
    if kind == "line":
        return x + 60.0 * np.sin(2 * np.pi * line_freq * t)
    if kind == "pop":
        y = x.copy()
        for k in rng.choice(n - int(sfreq), size=max(4, int(n / sfreq / 6)), replace=False):
            decay = np.exp(-np.arange(n - k) / (0.5 * sfreq))
            y[k:] += rng.choice([-1, 1]) * rng.uniform(150, 300) * decay
        return y
    if kind == "disconnected":
        return 40.0 * _colored_noise(1, n, sfreq, 1.0, rng)[0] + 20.0 * rng.standard_normal(n)
    if kind == "clipping":
        y = x * 6.0 + 200 * _colored_noise(1, n, sfreq, 3.0, rng)[0]
        return np.clip(y, -120.0, 120.0)
    if kind == "intermittent":
        y = x.copy()
        seg = int(2 * sfreq)
        for start in range(0, n, seg * 3):
            y[start: start + seg] = 80.0 * rng.standard_normal(min(seg, n - start))
        return y
    raise ValueError(f"Unknown bad channel kind {kind!r}; choose from {BAD_KINDS}")


def simulate_epochs_dataset(task: str = "motor_imagery", n_trials: int = 60, sfreq: float = 256.0,
                            montage="32", seed=None) -> RawEEG:
    """Convenience wrapper: a recording long enough for ``n_trials`` task trials."""
    per = 8.0 if task == "motor_imagery" else 1.35
    duration = 3.0 + n_trials * per + 2.0
    return simulate_eeg(duration, sfreq, montage, task=task, n_trials=n_trials, seed=seed,
                        stim_channel=False)
