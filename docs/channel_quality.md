# Channel quality: how the score works

`raw.assess_quality()` (or `eegproc quality FILE`) gives every EEG channel a
score from 0 to 100, a grade, and the reasons it lost points. This page
explains each metric so the numbers can be trusted, questioned and tuned.

## How it is computed

1. The data (at most 30 minutes, spread over the recording) are prepared in
   three versions: unfiltered, high-passed at 1 Hz, and band-passed 1-40 Hz.
2. Twelve metrics are measured per channel (below).
3. Each metric that crosses its "fine" limit costs points along a linear
   ramp up to a maximum. The score is `100 - sum(penalties)`, clipped to
   0-100. A channel that is flat for more than half of the recording scores 0.
4. Grades: **good** ≥ 75, **fair** ≥ 50, **poor** ≥ 25, **bad** < 25.
5. The PREP criteria are also run (`find_bad_channels`). A channel is
   **suggested as bad** when its grade is *bad* or any PREP criterion fires.

The penalties are listed in the report (`rep["Cz"]["penalties"]`) and drawn
in the "Why channels lose points" heatmap, so every score can be traced back.

## Metrics

| Key | Meaning | Normal EEG | Penalty ramp (max) |
|---|---|---|---|
| `flat_fraction` | Share of 1 s windows (1 Hz high-passed) with std < 0.5 µV | 0 | 0 → 50 % (100) |
| `clipping_fraction` | Share of samples in runs of ≥ 3 samples at the channel's own min/max | 0 | 0.05 → 1 % (40) |
| `amplitude_uv` | Robust std (0.7413 × IQR) of the 1-40 Hz signal | 3-40 µV | > 60 µV → 150 µV (40); < 2 µV → 0.5 µV (25) |
| `deviation_z` | Robust z-score of log amplitude against the other channels | \|z\| < 3 | 3 → 8 (40, shared with amplitude) |
| `neighbor_corr` | Mean of the two highest median \|r\| with the 4 nearest electrodes (1 s windows, 1-40 Hz). Without positions: the 4 most correlated channels | 0.6-0.95 | 0.6 → 0.2 (40) |
| `corr_bad_fraction` | Windows where the best \|r\| with any channel is < 0.4 (PREP) | ~0 | 5 → 50 % (30, shared) |
| `hf_noise_z` | PREP noisiness: MAD(> 50 Hz) / MAD(< 50 Hz), robust z across channels | < 3 | 3 → 8 (35) |
| `snr_db` | Mean PSD 1-30 Hz over mean PSD 45-95 Hz (line harmonics excluded) | 15-30 dB | 15 → 3 dB (35, shared) |
| `line_noise_db` | Mains peak over its ±3-8 Hz neighbourhood | < 25 dB | 25 → 45 dB (15) |
| `drift_db` | Robust amplitude of the < 1 Hz part over the 1-40 Hz part; also compared with the median channel | ~0 dB | 12 → 26 dB absolute (20); 6 → 15 dB above median (25) |
| `bad_window_fraction` | Windows where the channel alone has an artifact: peak-to-peak > 200 µV or amplitude z > 5 vs. the other channels in that window | < 2 % | 2 → 30 % (30) |
| `kurtosis` | Excess kurtosis of the channel minus the mean of its neighbours | < 5 | 5 → 20 (25) |
| `instability` | MAD of log10 window amplitude over time | < 0.2 | 0.25 → 0.6 (15) |
| `aperiodic_exponent` | Slope of the 1/f background (2-40 Hz, robust fit) | 1-3 | 0.6 → 0 (10) |
| `alpha_peak_hz`, `alpha_peak_db` | Alpha peak and its height above the 1/f fit | informational | none |

### Why these choices

- **Robust statistics everywhere.** Medians, MADs and IQRs keep one bad
  channel (or one artifact) from hiding another.
- **Neighbours, not all channels.** Scalp potentials are spatially smooth
  (volume conduction), so a working electrode resembles its neighbours. Using
  the *best two* of four neighbours means a good channel is not penalised
  because one of its neighbours is broken.
- **Local kurtosis.** Eye blinks give frontal channels a spiky, high-kurtosis
  signal although the electrodes are fine. Subtracting the neighbours cancels
  activity they share (blinks, real brain transients) and keeps
  single-electrode events (pops, cable movement).
- **Drift in the time domain.** A Welch spectrum with 4 s windows cannot see
  fluctuations slower than 0.25 Hz; the time-domain ratio can.
- **Line noise costs little.** A notch filter removes it, so it matters much
  less than a poor contact.

## Reading the report

```python
rep = raw.assess_quality()
print(rep.summary())            # ranked table with issues
rep.best(5), rep.worst(5)       # channel names
rep.bad_channels                # suggested bads (grade bad or PREP flag)
rep["T8"]                       # every metric, penalty and reason for one channel
rep.by_grade()                  # {'good': [...], 'fair': [...], ...}
rep.to_csv("quality.csv"); rep.to_json("quality.json")
rep.plot()                      # dashboard: ranked bars, sensor grades, score map, penalty heatmap
```

Typical interpretations:

| Pattern | Likely cause | What to do |
|---|---|---|
| flat 100 % | disconnected / broken electrode or reference problem | mark bad, interpolate |
| low `neighbor_corr` + high amplitude | loose electrode, high impedance | re-gel / mark bad |
| low SNR, high `hf_noise_z` | muscle tension (temporal/frontal) or amplifier noise | check electrode; low-pass if gamma not needed |
| high `line_noise_db` on one channel | high impedance on that electrode | notch filter; check contact |
| high `drift_db` | sweat, electrode polarisation, movement | high-pass 0.5-1 Hz; check contact |
| high `kurtosis` / `bad_window_fraction` | electrode pops, cable movement | mark bad or reject affected segments |
| `clipping_fraction` > 0 | amplifier saturation (DC offset too large) | re-gel; the data at those times are lost |

## Tuning

`assess_channel_quality(raw, line_freq=60, window=2.0, flat_threshold=0.2,
ptp_threshold=300, use_prep=False, max_seconds=None)` changes the main
thresholds. For data with very different amplitudes (e.g. intracranial
recordings) the absolute amplitude ramps can be ignored; the relative
criteria (z-scores, neighbour correlation) still apply.

## Validation

`tests/test_quality.py` simulates recordings with eight kinds of faults
(flat, white noise, drift, strong line noise, electrode pops, disconnection,
clipping, intermittent contact) on 19 and 64 channel caps and checks that the
faulty channels are exactly the lowest-ranked ones, and that clean
recordings score *good* throughout.
