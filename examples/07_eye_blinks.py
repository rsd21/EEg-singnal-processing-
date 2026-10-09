"""Eye blinks: detect them, measure them, and remove them with ICA.

    python examples/07_eye_blinks.py                  # simulated recording
    python examples/07_eye_blinks.py my_file.edf      # your own recording

Blinks are found on the EOG channel, or on Fp1/Fp2 when there is none.
"""

import os
import sys

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

import eegproc as ep  # noqa: E402
from eegproc.preprocessing import find_blinks  # noqa: E402
from eegproc.viz import CATEGORICAL, style  # noqa: E402

OUT = "outputs"
os.makedirs(OUT, exist_ok=True)

if len(sys.argv) > 1:
    raw = ep.read_raw(sys.argv[1])
    raw.set_montage("standard_1010")
else:
    raw = ep.simulate_eeg(duration=120, montage="32", eog=True, seed=42)

raw = raw.filter(1.0, 40.0)

# --- detect and measure
blinks = find_blinks(raw)
minutes = raw.n_times / raw.sfreq / 60
print(f"{len(blinks)} blinks in {minutes:.1f} min -> {len(blinks) / minutes:.1f} blinks/min")
print(f"Median blink width: {np.median(blinks.duration) * 1000:.0f} ms")

# --- remove with ICA
ica = ep.ICA(n_components=0.99, random_state=0).fit(raw)
eye, scores = ica.find_bads_eog(raw)
ica.exclude = eye
clean = ica.apply(raw)
removed = ", ".join(f"IC{i} (r = {scores[i]:.2f})" for i in eye)
print(f"ICA: {ica.n_components_} components, removed {len(eye)} eye component(s): {removed}")

# --- blink-locked average on a frontal channel, before vs after
ch = next((c for c in ("Fp1", "Fp2", "AF3", "F3") if c in raw.ch_names), raw.pick_names("eeg")[0])
half = int(0.5 * raw.sfreq)
peaks = ((blinks.onset + blinks.duration / 2) * raw.sfreq).astype(int)
peaks = peaks[(peaks >= half) & (peaks < raw.n_times - half)]


def blink_average(r):
    x = r[ch]
    return np.mean([x[p - half:p + half] for p in peaks], axis=0)


before, after = blink_average(raw), blink_average(clean)
print(f"Blink artifact at {ch}: {np.ptp(before):.0f} µV -> {np.ptp(after):.0f} µV peak-to-peak "
      f"({100 * (1 - np.ptp(after) / np.ptp(before)):.0f}% smaller)")

# --- figure
t = np.arange(raw.n_times) / raw.sfreq
view = t < 20
lag = (np.arange(2 * half) - half) / raw.sfreq * 1000
with style():
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 6), layout="constrained",
                                   gridspec_kw={"height_ratios": [3, 2]})
    for onset, dur in zip(blinks.onset, blinks.duration):
        if onset < 20:
            ax1.axvspan(onset, onset + dur, color=CATEGORICAL[3], alpha=0.3, lw=0)
    ax1.plot(t[view], raw[ch][view], color=CATEGORICAL[0], lw=0.8, label="before ICA")
    ax1.plot(t[view], clean[ch][view], color=CATEGORICAL[1], lw=0.8, label="after ICA")
    ax1.set(xlim=(0, 20), xlabel="Time (s)", ylabel=f"{ch} (µV)")
    ax1.set_title(f"{ch}: first 20 s, detected blinks shaded", loc="left")
    ax1.legend(loc="upper right")

    ax2.plot(lag, before, color=CATEGORICAL[0], lw=2, label="before ICA")
    ax2.plot(lag, after, color=CATEGORICAL[1], lw=2, label="after ICA")
    ax2.axvline(0, color="0.6", lw=0.8)
    ax2.set(xlabel="Time from blink peak (ms)", ylabel="µV")
    ax2.set_title(f"Average of {len(peaks)} blinks at {ch}", loc="left")
    ax2.legend(loc="upper right")
    fig.savefig(f"{OUT}/eye_blinks.png", dpi=120)

clean.save(f"{OUT}/blinks_removed.edf")
print(f"Saved {OUT}/eye_blinks.png and {OUT}/blinks_removed.edf")
