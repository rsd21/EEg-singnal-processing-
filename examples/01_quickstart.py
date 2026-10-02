"""Quick start: load any EEG file, inspect it, score every channel, write a report.

Usage:
    python examples/01_quickstart.py                 # uses a simulated recording
    python examples/01_quickstart.py my_file.edf     # or any supported file
"""

import os
import sys

import matplotlib

matplotlib.use("Agg")

import eegproc as ep  # noqa: E402

OUT = "outputs"
os.makedirs(OUT, exist_ok=True)

if len(sys.argv) > 1:
    raw = ep.read_raw(sys.argv[1])                     # EDF, BDF, BrainVision, CSV, MAT, SET, ...
    raw.set_montage("standard_1010")                   # standard positions for 10-10 channel names
else:
    raw = ep.simulate_eeg(duration=120, montage="clinical19", bad_channels="default", eog=True, seed=0)
    print("Simulated recording; injected bad channels:", raw.meta["ground_truth_bads"])

# 1. What is in the file?
print(raw)
print(raw.describe())

# 2. Access data: by channel name, type or time window
cz = raw["Cz"] if "Cz" in raw.ch_names else raw.data[0]
segment = raw.get_data(picks="eeg", tmin=10, tmax=12)   # (n_eeg, 2 s of samples)
print(f"\n2 s EEG segment: {segment.shape}, first channel mean {cz.mean():.2f} µV")

# 3. Which channels perform well? 0-100 score, grade and reasons
quality = raw.assess_quality()
print("\n" + quality.summary())
quality.plot().savefig(f"{OUT}/quality_dashboard.png", bbox_inches="tight")
quality.to_csv(f"{OUT}/channel_quality.csv")

# 4. One self-contained HTML report with everything
ep.generate_report(raw, f"{OUT}/report.html", quality=quality)
print(f"\nWrote {OUT}/quality_dashboard.png, {OUT}/channel_quality.csv and {OUT}/report.html")
