# Processing guide

Recommended steps for common kinds of EEG data, with the reasons behind
them. Every step is available as a Python call and as a pipeline step
(`examples/pipeline.yaml`).

## 1. Look before you process

```python
raw = ep.read_raw("file.edf")
raw.set_montage("standard_1010")
print(raw.describe())               # right channels? right units? events present?
quality = raw.assess_quality()
print(quality.summary())
raw.plot(duration=20, quality=quality, display_filter=(0.5, 40))
raw.plot_psd()
```

Check: sampling rate, channel types (EOG/ECG recognised?), amplitudes of
tens of µV, a 1/f spectrum with an alpha peak for eyes-closed segments, the
line-noise peak, and the channels the quality report flags.

## 2. Filtering

| Goal | Typical band |
|---|---|
| Resting-state spectra / connectivity | 1-45 Hz |
| ERPs | 0.1-30 Hz (0.5 Hz high-pass for noisy data or BCI) |
| Motor imagery / SMR BCI | 1-40 Hz, analysis in 8-13 (mu) and 13-30 Hz (beta) |
| ICA fitting | 1 Hz high-pass (the ICA class does this on its own copy) |

- Filters run forwards and backwards (zero phase), so latencies and phases are preserved.
- `method="iir"` (Butterworth, default) is fast and robust; `method="fir"`
  (windowed sinc) has a linear phase and a predictable transition band.
- High-pass filtering above ~0.3 Hz distorts slow ERP components (e.g. the
  P300 and CNV); keep it low for ERP amplitude measurements.
- Remove mains interference with `notch_filter(50)` / `(60)` or `"auto"`.

## 3. Bad channels

```python
res = ep.find_bad_channels(raw)            # PREP criteria
raw.set_bads(sorted(set(res.bads) | set(quality.bad_channels)))
raw = raw.interpolate_bads()               # needs electrode positions
```

Interpolate before average referencing: a bad channel would otherwise leak
into every channel through the average.

## 4. Reference

- **Average reference**: standard for high-density data (≥ 32 channels) and
  topographic analyses.
- **Linked mastoids** (`set_reference(["M1", "M2"])`): common for ERPs.
- **Laplacian** (`laplacian(raw, "spline")`): sharpens local activity; very
  useful for sensorimotor rhythms (C3/C4) and to reduce volume conduction.
- **Bipolar** (`bipolar_reference(raw, "double_banana")`): clinical review.

## 5. Artifacts

- Eye blinks and movements: ICA (`remove_artifacts_ica(raw)` or step by
  step with `ICA.find_bads_eog`). Works best on 1 Hz high-passed data (done
  internally) with enough data (≥ 20 × n_components² samples).
- Heart: `ICA.find_bads_ecg` when an ECG channel exists.
- Muscle: `annotate_muscle(raw)` marks bursts; `ICA.find_bads_muscle` flags
  components with flat/rising spectra.
- Large transients: `annotate_amplitude(raw, peak_to_peak=150)`.
- Segments annotated `BAD*` are skipped by `compute_psd`, `ICA.fit` and
  `Epochs` (`reject_by_annotation=True`).

## 6. Analysis recipes

**Resting state**
```python
spec = clean.compute_psd(picks="eeg", method="welch", window_sec=4)
spec.band_power(relative=True)          # per band per channel
spec.peak_frequency(7, 14)              # individual alpha frequency
spec.aperiodic()                        # 1/f offset and exponent
ep.compute_connectivity(clean, "wpli", band=(8, 13))
ep.rank_channels(clean, "alpha")
```

**ERP**
```python
epochs = ep.Epochs(clean, tmin=-0.2, tmax=0.8, baseline=(None, 0), reject=150)
evoked = epochs["target"].average()
evoked.get_peak(tmin=0.25, tmax=0.5, mode="pos")
ep.rank_channels(epochs, "erp", condition="target")
ep.rank_channels(epochs, "decoding", feature="erp", window=(0.1, 0.6))
```

**Motor imagery / BCI**
```python
epochs = ep.Epochs(clean, tmin=-1, tmax=4, baseline=None, reject=200)
ep.rank_channels(epochs, "erd", baseline=(-1, 0), window=(0.5, 3.5), condition="left")
ep.rank_channels(epochs, "discriminability", window=(0.5, 3.5))
ep.rank_channels(epochs, "decoding", window=(0.5, 3.5))   # CV accuracy + p-value per channel
```

## 7. Statistics notes

- Decoding accuracy is balanced accuracy from repeated stratified k-fold
  cross-validation; the p-value is a binomial test against chance. With few
  trials, chance-level fluctuations are large (Combrisson & Jerbi, 2015), so
  look at the p-value, not only the accuracy.
- ERP SNR uses random sign-flipped averages as the noise estimate; 0 dB
  means no response above noise.
- ERD/ERS is band power relative to baseline, averaged over trials
  (Pfurtscheller & Lopes da Silva, 1999). Epochs should extend at least half
  a second beyond the baseline and analysis windows to avoid filter edge
  effects.

## References

- Bigdely-Shamlo et al. (2015). The PREP pipeline. *Front. Neuroinform.* 9:16.
- Perrin et al. (1989). Spherical splines for scalp potential and current density mapping. *EEG Clin. Neurophysiol.* 72:184-187.
- Hyvärinen & Oja (2000). Independent component analysis: algorithms and applications. *Neural Networks* 13:411-430.
- Vinck et al. (2011). The weighted phase lag index. *NeuroImage* 55:1548-1565.
- Pfurtscheller & Lopes da Silva (1999). Event-related EEG/MEG synchronization and desynchronization. *Clin. Neurophysiol.* 110:1842-1857.
- Ledoit & Wolf (2004). A well-conditioned estimator for large-dimensional covariance matrices. *J. Multivar. Anal.* 88:365-411.
- Kemp & Olivan (2003). European data format 'plus' (EDF+). *Clin. Neurophysiol.* 114:1755-1761.
