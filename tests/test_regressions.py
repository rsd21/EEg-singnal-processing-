"""Regression tests for bugs found in code review."""

import numpy as np
import pytest

from eegproc import RawEEG, simulate_eeg
from eegproc.analysis import extract_features
from eegproc.core.epochs import Epochs, find_stim_events
from eegproc.preprocessing import annotate_muscle, bipolar_reference, filter_data, interpolate_bads


def test_held_trigger_code_is_not_mistaken_for_an_offset():
    x = np.zeros(1000)
    x[100:600] = 1  # block design: code held for half the recording
    x[700:720] = 2
    assert find_stim_events(x).tolist() == [[100, 0, 1], [700, 0, 2]]
    offset = np.full(1000, 256.0)  # line that never returns to zero
    offset[300:305] = 257
    assert find_stim_events(offset).tolist() == [[300, 0, 257]]


def test_set_montage_can_keep_digitised_positions():
    raw = simulate_eeg(5, montage="standard_1020", seed=0)
    measured = np.array([0.1, 0.1, 0.99]) / np.linalg.norm([0.1, 0.1, 0.99])
    raw.positions["Cz"] = measured
    raw.set_montage("standard_1010", overwrite=False)
    assert np.allclose(raw.positions["Cz"], measured)
    raw.set_montage("standard_1010")
    assert not np.allclose(raw.positions["Cz"], measured)


def test_narrow_fir_band_stop():
    t = np.arange(5000) / 250
    x = np.sin(2 * np.pi * 50 * t) + np.sin(2 * np.pi * 10 * t)
    y = filter_data(x, 250, 52, 48, method="fir")
    mid = slice(1000, -1000)
    assert np.std(y[mid] - np.sin(2 * np.pi * 10 * t[mid])) < 0.05
    with pytest.raises(ValueError, match="wider than the stop band"):
        filter_data(x, 250, 52, 48, method="fir", l_trans=5, h_trans=5)


def test_notch_accepts_an_array_of_frequencies():
    raw = simulate_eeg(10, montage="standard_1020", seed=0)
    out = raw.notch_filter(np.array([50.0, 100.0]))
    assert "notch" in out.history[-1]


def test_muscle_detection_skips_low_sampling_rates():
    raw = RawEEG(np.random.default_rng(0).normal(size=(4, 64 * 30)), 64.0, ["Fz", "Cz", "Pz", "Oz"])
    assert len(annotate_muscle(raw)) == 0


def test_epoch_features_default_to_eeg_channels():
    raw = simulate_eeg(60, montage="standard_1020", eog=True, task="oddball", seed=0)
    epochs = Epochs(raw, tmin=-0.1, tmax=0.5)
    channels = {row["channel"] for row in extract_features(epochs, include=["time"]).rows}
    assert "VEOG" not in channels and "Cz" in channels


def test_interpolation_ignores_other_bad_channels():
    raw = simulate_eeg(60, montage="64", seed=1, blinks=False, bad_channels={})
    clean_c3 = raw["C3"].copy()
    raw.data[raw.channel_index("C3")] = 0.0
    raw.data[raw.channel_index("C1")] = 1e4 * np.random.default_rng(0).normal(size=raw.n_times)
    raw.set_bads(["C3", "C1"])
    fixed = interpolate_bads(raw, bads=["C3"])
    r = np.corrcoef(filter_data(fixed["C3"], raw.sfreq, 1, 30), filter_data(clean_c3, raw.sfreq, 1, 30))[0, 1]
    assert r > 0.8 and fixed.bads == ["C1"]


def test_bipolar_handles_dashes_in_names_and_keeps_bads():
    raw = simulate_eeg(5, montage="standard_1020", seed=0)
    pos = dict(raw.positions)
    raw.rename_channels({n: f"EEG {n}-REF" for n in raw.ch_names})
    raw.positions = {f"EEG {n}-REF": p for n, p in pos.items()}
    raw.set_bads(["EEG F7-REF"])
    bp = bipolar_reference(raw, [("EEG Fp1-REF", "EEG F7-REF"), ("EEG F3-REF", "EEG C3-REF")])
    assert set(bp.positions) == set(bp.ch_names)
    assert bp.bads == ["EEG Fp1-REF-EEG F7-REF"]


def test_quality_of_long_recordings_is_not_hurt_by_subsampling():
    raw = simulate_eeg(600, sfreq=100, montage="standard_1020", seed=2, blinks=False, bad_channels={"O2": "flat"})
    eeg = raw.pick_indices("eeg")
    walk = np.cumsum(np.random.default_rng(0).normal(size=(len(eeg), raw.n_times)), axis=1)
    walk = filter_data(walk, raw.sfreq, None, 0.05)
    raw.data[eeg] += 4000 * walk / walk.std(axis=1, keepdims=True)  # wandering DC offset
    chunked = raw.assess_quality(max_seconds=300)  # 10 separate 30 s chunks
    continuous = raw.crop(150, 450).assess_quality()
    assert chunked.bad_channels == continuous.bad_channels == ["O2"]
    assert abs(np.median(chunked.scores) - np.median(continuous.scores)) < 5


def test_auxiliary_channel_names_are_not_eeg():
    from eegproc.core.channels import infer_channel_type

    assert [infer_channel_type(n) for n in ("EXG1", "Erg2", "Plet", "GSR1", "Temp")] == ["misc"] * 5


def test_bids_channels_tsv_sets_types_and_bads(tmp_path):
    from eegproc.io import read_raw

    raw = simulate_eeg(5, montage="standard_1020", eog=False, seed=0)
    path = raw.save(tmp_path / "sub-01_task-rest_eeg.edf")
    rows = ["name\ttype\tunits\tstatus"] + [f"{n.lower()}\tEEG\tuV\tgood" for n in raw.ch_names]
    rows[1] = f"{raw.ch_names[0]}\tEOG\tuV\tgood"
    rows[2] = f"{raw.ch_names[1]}\tEEG\tuV\tbad"
    (tmp_path / "sub-01_task-rest_channels.tsv").write_text("\n".join(rows) + "\n")
    back = read_raw(path)
    assert back.get_channel_type(raw.ch_names[0]) == "eog" and back.bads == [raw.ch_names[1]]
