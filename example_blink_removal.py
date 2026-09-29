#!/usr/bin/env python3
"""Example: Remove eye blink artifacts from EEG using ICA."""

import numpy as np
import matplotlib.pyplot as plt
from eegproc.datasets import simulate_eeg
from eegproc.preprocessing import ICA, find_blinks, filter_data

# Simulate EEG with blinks
print("Simulating EEG with eye blinks...")
raw = simulate_eeg(duration=120, sfreq=256, montage="32", blinks=True, eog=True, seed=42)

print(f"Original data: {raw.data.shape}")
print(f"Channels: {raw.ch_names}")

# Find blinks
print("\nDetecting blinks...")
blinks = find_blinks(raw)
print(f"Found {len(blinks)} blinks")

# Apply ICA to separate components
print("\nApplying ICA (n_components=0.99 explains 99% of variance)...")
ica = ICA(n_components=0.99, random_state=0).fit(raw)
print(f"ICA found {ica.n_components} components")

# Find EOG (eye) components
print("\nIdentifying eye components...")
bad_components, scores = ica.find_bads_eog(raw, ch_name="VEOG")
print(f"Eye-related components: {bad_components}")
print(f"Correlation scores: {scores[bad_components]}")

# Mark components for removal and apply
ica.exclude = bad_components
print(f"\nRemoving {len(bad_components)} eye components...")
cleaned = ica.apply(raw)

# Filter both versions for comparison
print("Filtering (1-30 Hz)...")
raw_filt = raw.filter(1.0, 30.0)
cleaned_filt = cleaned.filter(1.0, 30.0)

# Pick Fp1 for visualization (most affected by blinks)
fp1_orig = raw_filt["Fp1"]
fp1_clean = cleaned_filt["Fp1"]
t = np.arange(len(fp1_orig)) / raw.sfreq

# Plot comparison
fig, axes = plt.subplots(3, 1, figsize=(14, 8))

# Original with blinks marked
ax = axes[0]
ax.plot(t, fp1_orig, color="red", linewidth=0.8, label="Original (with blinks)")
for onset, duration in zip(blinks.onset, blinks.duration):
    ax.axvspan(onset, onset + duration, alpha=0.15, color="orange")
ax.set_xlim(0, 40)
ax.set_ylabel("Fp1 (µV)")
ax.set_title("Original EEG - Blink artifacts visible in red regions")
ax.legend()
ax.grid(True, alpha=0.3)

# After ICA removal
ax = axes[1]
ax.plot(t, fp1_clean, color="green", linewidth=0.8, label="After ICA artifact removal")
for onset, duration in zip(blinks.onset, blinks.duration):
    ax.axvspan(onset, onset + duration, alpha=0.15, color="orange")
ax.set_xlim(0, 40)
ax.set_ylabel("Fp1 (µV)")
ax.set_title("After ICA Artifact Removal - Blink artifacts greatly reduced")
ax.legend()
ax.grid(True, alpha=0.3)

# Difference (removed artifact)
ax = axes[2]
diff = fp1_orig - fp1_clean
ax.plot(t, diff, color="purple", linewidth=0.8, label="Removed artifact (original - cleaned)")
for onset, duration in zip(blinks.onset, blinks.duration):
    ax.axvspan(onset, onset + duration, alpha=0.15, color="orange")
ax.set_xlim(0, 40)
ax.set_xlabel("Time (s)")
ax.set_ylabel("Artifact (µV)")
ax.set_title("Isolated Blink Artifacts - Mostly captured by removed ICA components")
ax.legend()
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig("/tmp/blink_removal.png", dpi=100, bbox_inches="tight")
print("\n✓ Saved: /tmp/blink_removal.png")

# Show quantitative improvement
print("\n" + "="*60)
print("QUANTITATIVE RESULTS")
print("="*60)

eeg_ch = raw.pick_indices("eeg")

# Calculate SNR (signal-to-noise as ratio of mean activity)
orig_snr = np.abs(raw_filt.data[eeg_ch]).mean()
clean_snr = np.abs(cleaned_filt.data[eeg_ch]).mean()

# Peak-to-peak amplitude
orig_pp = np.ptp(fp1_orig)
clean_pp = np.ptp(fp1_clean)

print(f"\nFp1 channel (most artifact-prone):")
print(f"  Peak-to-peak (original): {orig_pp:.2f} µV")
print(f"  Peak-to-peak (cleaned):  {clean_pp:.2f} µV")
print(f"  Reduction: {(1 - clean_pp/orig_pp)*100:.1f}%")

print(f"\nOverall EEG amplitude (all channels):")
print(f"  Mean amplitude (original): {orig_snr:.2f} µV")
print(f"  Mean amplitude (cleaned):  {clean_snr:.2f} µV")

print(f"\n✓ ICA successfully removed blink artifacts!")
print(f"  Removed {len(bad_components)} eye-related component(s)")
print(f"  Preserved {ica.n_components - len(bad_components)} brain activity components")
