# File formats and data access

## Reading

```python
import eegproc as ep

raw = ep.read_raw("file.edf")                 # format from the extension
raw = ep.read_raw("file.dat", fmt="csv")      # force a format
raw = ep.read_raw("signals.npy", sfreq=500)   # arrays without a stored rate
```

`read_raw` picks the reader from the extension, falls back to recognising
the content (EDF/BDF/BrainVision/NumPy/HDF5/MATLAB/XDF signatures, text),
and finally tries MNE-Python if it is installed.

### What every reader guarantees

| Property | Convention |
|---|---|
| Signal array | `raw.data`, shape `(n_channels, n_samples)`, float64 |
| Units | EEG/EOG/ECG/EMG in **µV** whatever the file used (V, mV, µV, nV); other channels keep their unit (`raw.units`) |
| Channel names | cleaned: `"EEG FP1-REF"` → `Fp1`, `"Fc5."` → `FC5`, `T3/T4/T5/T6` → `T7/T8/P7/P8`; bipolar labels such as `Fp1-F7` are left alone; originals in `raw.meta["original_ch_names"]` (disable with `clean_names=False`) |
| Channel types | inferred from labels and units: `eeg`, `eog`, `ecg`, `emg`, `stim`, `resp`, `misc` (`raw.set_channel_types({...})` to correct) |
| Events | `raw.annotations` (onset, duration, description in seconds); trigger channels have type `stim` |
| Metadata | `raw.meta`: file name, format, recording date, subject, original labels, filters... |

### Format notes

**EDF / EDF+ / BDF / BDF+** (`.edf`, `.bdf`, `.rec`)
- implemented from the specification; 16-bit (EDF) and 24-bit (BDF) samples
- EDF+ annotations (time-stamped annotation lists) become `raw.annotations`
- mixed sampling rates: all channels are resampled to the highest rate
  (lower-rate channels such as SpO2 are upsampled)
- BioSemi `Status` channel: kept as a trigger channel, events use the lower 16 bits
- partial reads: `read_raw(path, include=["Cz", "Pz"], tmin=60, tmax=120)`
  decodes only those channels and records; `read_edf_header(path)` reads
  the header without any signal data
- `01.01.85` start dates and `X` patient fields (EDF+ "unknown") are treated as missing

**BrainVision** (`.vhdr` + `.vmrk` + `.eeg`)
- binary `INT_16`, `INT_32`, `UINT_16`, `IEEE_FLOAT_32/64`, multiplexed or
  vectorized; ASCII data files with `SkipLines`/`SkipColumns`/`DecimalSymbol`
- per-channel resolution and unit; markers become annotations
  (`"Stimulus/S  1"`), comments keep their text, extra "New Segment"
  markers become `BAD boundary`

**Text** (`.csv`, `.tsv`, `.txt`, `.dat`, `.asc`)
- delimiter, header row and orientation (channels in columns, or in rows
  with the label first) are detected
- sampling rate from a time column (`time`, `timestamp`, `time (ms)` ...) or
  from comments such as `%Sample Rate = 250 Hz` (OpenBCI) or `# sfreq = 500`
- index columns (`Sample Index`) and non-numeric columns (formatted
  timestamps) are dropped; `marker`/`event`/`trigger` columns become annotations
- `unit="mV"` (or `"V"`) converts values to µV; missing values are interpolated

**MATLAB** (`.mat`) and **EEGLAB** (`.set` + optional `.fdt`)
- the largest numeric matrix is the data (or a field named `data`, `eeg`,
  `signals`, `X` ...), the rate is searched under `fs`, `srate`, `sfreq`,
  `sampling_rate`, ..., labels under `labels`, `ch_names`, `channels`, ...
- v7.3 files are HDF5 and need `h5py`
- EEGLAB: channel locations (X/Y/Z) and events are imported; epoched
  datasets are concatenated with `BAD boundary` markers

**NumPy** (`.npy`, `.npz`), **HDF5** (`.h5`), **XDF** (`.xdf`)
- `.npy` holds only the array: pass `sfreq` (and `ch_names`)
- `.npz`/`.h5` written by eegproc store names, types, units, positions,
  bad channels and annotations
- XDF (needs `pyxdf`): the EEG stream with the most channels is loaded
  (or `stream="name"`), marker streams become annotations

**Through MNE** (needs `mne`): FIF, GDF, Neuroscan CNT, EGI (`.mff`, `.raw`),
Curry, Persyst, Nicolet, eXimia, Nihon Kohden `.eeg` (when no `.vhdr`
exists), SNIRF.

## Writing

```python
raw.save("out.edf")      # EDF+ with annotations (16 bit, range chosen per channel)
raw.save("out.bdf")      # BDF+ (24 bit)
raw.save("out.vhdr")     # BrainVision float32 (+ .vmrk, .eeg)
raw.save("out.csv")      # time column, channels, marker column
raw.save("out.npz")      # everything, lossless
raw.save("out.mat")      # MATLAB
raw.save("out.h5")       # HDF5 (h5py)
raw.save("out_raw.fif")  # FIF (mne)
```

EDF stores 16-bit integers: each channel's range is set from its own
minimum and maximum, so the quantisation step is (max - min) / 65535
(0.006 µV for a ±200 µV channel). Use BDF or BrainVision for 24/32-bit
precision. EDF records must be complete; when the recording length does not
fit a whole number of records, a record size that divides it exactly is
chosen, otherwise the last record is padded.

## Accessing data

```python
raw.ch_names, raw.ch_types, raw.units, raw.sfreq, raw.duration, raw.times
raw["Cz"]                                  # 1D array
raw[["Fz", "Cz"], 0:512]                   # channels x samples
raw.get_data(picks="eeg", tmin=5, tmax=10, exclude_bads=True)
raw.pick_names("eog")                      # names by type
raw.channel_table()                        # per-channel type, unit, position, stats
raw.to_dataframe()                         # pandas, samples x channels (+ time)

raw.pick(["Fz", "Cz", "Pz"])               # new object
raw.drop_channels(["ECG"])
raw.crop(10, 70)
raw.rename_channels({"EXG1": "HEOG"}).set_channel_types({"HEOG": "eog"})  # in place
raw.set_montage("standard_1010")           # or a file: .elc .sfp .loc .ced .csv .tsv
raw.find_events()                          # (events array, {name: code})
```

Picks accept channel names (case-insensitive, raw or cleaned labels),
indices, slices, boolean masks, channel types (`"eeg"`, `"eog"`, ...),
`"data"` (everything but triggers), `"good"` (EEG not marked bad) and lists
mixing these.
