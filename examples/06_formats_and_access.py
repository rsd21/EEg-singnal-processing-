"""Reading, accessing and converting between file formats."""

import os

import numpy as np

import eegproc as ep
from eegproc.io import read_edf_header

OUT = "outputs/formats"
os.makedirs(OUT, exist_ok=True)

raw = ep.simulate_eeg(duration=30, montage="clinical19", task="oddball", eog=True, stim_channel=True, seed=5)

print("Supported formats:")
for row in ep.supported_formats():
    print(f"  {row['extension']:<7} read: {row['read']:<24} write: {row['write'] or '-':<4} {row['description']}")

# Write the same recording in several formats and read each back
for ext in (".edf", ".bdf", ".vhdr", ".csv", ".npz", ".mat"):
    path = os.path.join(OUT, "recording" + ext)
    raw.save(path)
    back = ep.read_raw(path)
    err = np.abs(back.get_data("eeg") - raw.get_data("eeg")).max()
    print(f"{ext:>6}: {back.n_channels} channels, {back.sfreq:g} Hz, {len(back.annotations)} annotations, "
          f"max error {err:.2g} µV")

# EDF header only (fast, no signal data), then a partial read
hdr = read_edf_header(os.path.join(OUT, "recording.edf"))
print("\nEDF header:", hdr["n_signals"], "signals,", hdr["duration"], "s,", hdr["subtype"])
part = ep.read_raw(os.path.join(OUT, "recording.edf"), include=["Cz", "Pz"], tmin=5, tmax=15)
print("Partial read:", part)

# Accessing data
print("\nCz first 5 samples:", np.round(raw["Cz"][:5], 2))
print("EEG channels:", raw.pick_names("eeg"))
print("EOG channels:", raw.pick_names("eog"), "| trigger:", raw.pick_names("stim"))
events, event_id = raw.find_events()
print("Events:", event_id, "first 3:", events[:3].tolist())
print(raw.channel_table()[0])
