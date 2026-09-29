# EEG Blink Analysis Guide

Quick reference for detecting and removing eye blink artifacts using the eegproc toolkit.

## Installation

```bash
cd /home/user/EEg-singnal-processing-
pip install -e .
```

## 1. Load Your EEG Data

```python
from eegproc import read_raw

# Supports: EDF, BDF, BrainVision, CSV, MATLAB, EEGLAB, HDF5, XDF
raw = read_raw("your_eeg_file.edf")

print(f"Loaded {raw.n_channels} channels")
print(f"Duration: {raw.n_times / raw.sfreq:.1f} seconds")
print(f"Channels: {raw.ch_names}")
```

## 2. Detect Eye Blinks

```python
from eegproc.preprocessing import find_blinks

# Requires an EOG channel (VEOG, Fp1, or Fp2 by default)
blinks = find_blinks(raw)

print(f"Found {len(blinks)} blinks")
print(f"Mean blink duration: {blinks.duration.mean():.3f} s")
print(f"Blink rate: {len(blinks) / (raw.n_times / raw.sfreq) * 60:.1f} blinks/min")
```

## 3. Remove Blink Artifacts with ICA

```python
from eegproc.preprocessing import ICA

# Extract independent components
ica = ICA(n_components=0.99, random_state=0).fit(raw)

# Find eye-related components
bad_comps, scores = ica.find_bads_eog(raw, ch_name="VEOG")  # or "Fp1", "Fp2"

print(f"Eye components to remove: {bad_comps}")
print(f"Correlation with EOG: {scores[bad_comps]}")

# Remove them
ica.exclude = bad_comps
cleaned = ica.apply(raw)
```

## 4. Assess Data Quality

```python
from eegproc.quality import find_bad_channels, score_channels

# Find bad channels (flat, noisy, high noise, etc.)
bad_ch = find_bad_channels(raw)
print(f"Bad channels: {bad_ch.bads}")

# Score all channels (0-100, higher is better)
scores = score_channels(raw)
for ch in scores.index[:5]:
    print(f"{ch}: {scores[ch]:.1f}")
```

## 5. Filter and Preprocess

```python
# Bandpass filter (1-30 Hz typical for EEG)
filtered = raw.filter(1.0, 30.0, method="iir")

# Notch filter (remove 50/60 Hz line noise)
notched = filtered.notch_filter(50.0)  # Use 60.0 in US/Canada

# Re-reference to average of all EEG channels
raw_avg = notched.set_reference("average")

# Resample to lower sampling rate if needed
downsampled = raw_avg.resample(100.0)  # Downsample to 100 Hz
```

## 6. Visualize Results

```python
from eegproc.viz import plot_raw, plot_psd

# Plot raw signal (first 30 seconds)
fig = plot_raw(filtered, duration=30, n_channels=5, show=True)

# Plot power spectral density
fig = plot_psd(filtered, show=True)
```

## 7. Generate a Full Report

```python
from eegproc.report import make_html_report

report = make_html_report(
    raw=cleaned,
    title="EEG Analysis - Blinks Removed",
    description="ICA-cleaned EEG with blink artifacts removed"
)
report.write_html("eeg_report.html")
```

## CLI Tools

Also available from the command line:

```bash
# Detect blinks
eegproc quality your_file.edf

# Generate HTML report
eegproc report your_file.edf -o report.html

# Plot spectra
eegproc plot your_file.edf spectrum

# Get file info
eegproc info your_file.edf
```

## Example Files

Run the example scripts:

```bash
python example_blink_analysis.py    # Detect and analyze blinks
python example_blink_removal.py     # Remove blinks with ICA
```

## Common Parameters

### ICA
- `n_components=0.99`: Retain 99% of variance (faster)
- `n_components=20`: Fixed 20 components (more control)
- `random_state=0`: For reproducibility

### Filtering
- `method="iir"`: Faster, causal (default)
- `method="fir"`: Zero-phase, higher quality
- `phase="zero"`: Zero-phase distortion (good for ICA input)

### Blink Detection
- Uses `find_blinks(raw)` → correlates with EOG channels
- Returns Annotations with onset times and durations
- Automatically finds VEOG, Fp1, or Fp2

## Next Steps

1. **Preprocessing pipeline:** See `processing_guide.md` for resting-state vs task EEG
2. **Channel quality:** See `channel_quality.md` for detailed scoring metrics
3. **File formats:** See `file_formats.md` for supported readers/writers
4. **API reference:** Generated docs at `docs/` (or `help(raw.filter)`, etc.)
