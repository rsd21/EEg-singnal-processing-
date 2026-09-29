#!/usr/bin/env python3
"""Example: Detect and analyze eye blinks in EEG data."""

import numpy as np
import matplotlib.pyplot as plt
from eegproc.datasets import simulate_eeg
from eegproc.preprocessing import find_blinks, filter_data
from eegproc.viz.spectra import plot_psd

# Simulate 2 minutes of EEG with realistic blinks (with EOG channels)
print("Simulating EEG data with eye blinks...")
raw = simulate_eeg(duration=120, sfreq=256, montage="32", blinks=True, eog=True, seed=42)

print(f"Data shape: {raw.data.shape}")
print(f"Sampling rate: {raw.sfreq} Hz")
print(f"Channels: {raw.ch_names}")

# Detect blinks using VEOG (vertical EOG)
print("\nDetecting blinks...")
blinks = find_blinks(raw)
print(f"Found {len(blinks)} blinks")
print(f"  Onset times (first 5): {blinks.onset[:5]}")
print(f"  Durations (first 5): {blinks.duration[:5]}")

# Show blink statistics
print(f"\nBlink statistics:")
print(f"  Mean duration: {blinks.duration.mean():.3f} s")
print(f"  Median duration: {np.median(blinks.duration):.3f} s")
print(f"  Blink rate: {len(blinks) / (raw.n_times / raw.sfreq):.2f} blinks/min")

# Analyze EEG during blinks vs non-blinks
print("\nAnalyzing EEG amplitude during blinks...")
eeg_ch = raw.pick_indices("eeg")
blink_samples = []
for onset, duration in zip(blinks.onset, blinks.duration):
    start = int(onset * raw.sfreq)
    end = int((onset + duration) * raw.sfreq)
    blink_samples.extend(range(start, end))

blink_samples = np.array(blink_samples)
non_blink_samples = np.array([i for i in range(raw.n_times) if i not in blink_samples])

blink_amp = np.abs(raw.data[eeg_ch][:, blink_samples]).mean()
non_blink_amp = np.abs(raw.data[eeg_ch][:, non_blink_samples]).mean()

print(f"  Mean EEG amplitude during blinks: {blink_amp:.2f} µV")
print(f"  Mean EEG amplitude outside blinks: {non_blink_amp:.2f} µV")
print(f"  Ratio: {blink_amp / non_blink_amp:.2f}x")

# Filter and plot
print("\nApplying 1-30 Hz bandpass filter...")
filtered = raw.filter(1.0, 30.0)

# Save a quick visualization
print("\nGenerating plots...")
fig, axes = plt.subplots(2, 1, figsize=(12, 8))

# Plot raw VEOG with blink markers
ax = axes[0]
veog = raw["VEOG"]
t = np.arange(len(veog)) / raw.sfreq
ax.plot(t, veog, color="blue", linewidth=0.8, label="VEOG")
for onset, duration in zip(blinks.onset, blinks.duration):
    ax.axvspan(onset, onset + duration, alpha=0.2, color="red")
ax.set_xlim(0, 30)  # First 30 seconds
ax.set_ylabel("VEOG (µV)")
ax.set_title("Vertical EOG with Detected Blinks (red regions)")
ax.legend()

# Plot filtered Cz
ax = axes[1]
cz = filtered["Cz"]
ax.plot(t, cz, color="black", linewidth=0.8)
for onset, duration in zip(blinks.onset, blinks.duration):
    ax.axvspan(onset, onset + duration, alpha=0.2, color="red")
ax.set_xlim(0, 30)
ax.set_xlabel("Time (s)")
ax.set_ylabel("Cz (µV)")
ax.set_title("Filtered EEG (Cz) - Blinks cause large VEOG deflections")

plt.tight_layout()
plt.savefig("/tmp/blink_detection.png", dpi=100, bbox_inches="tight")
print("  Saved: /tmp/blink_detection.png")

print("\n✓ Done! Blinks detected and analyzed.")
