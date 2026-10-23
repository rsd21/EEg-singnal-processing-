import os
import shutil
import sys
import types

import numpy as np
import pytest

from eegproc import RawEEG, read_raw, supported_formats, write_raw
from eegproc.io import read_edf_header


def _roundtrip(raw, path, **kw):
    write_raw(raw, path)
    return read_raw(path, **kw)


@pytest.mark.parametrize("ext, tol", [(".edf", 0.01), (".bdf", 1e-4)])
def test_edf_bdf_roundtrip(small_raw, tmp_path, ext, tol):
    back = _roundtrip(small_raw, tmp_path / f"x{ext}")
    assert back.ch_names == small_raw.ch_names
    assert back.sfreq == small_raw.sfreq and back.n_times == small_raw.n_times
    assert np.abs(back.data[:5] - small_raw.data[:5]).max() < tol
    assert np.array_equal(back.data[5], small_raw.data[5])  # trigger codes exact
    assert back.ch_types == small_raw.ch_types
    assert back.annotations.description == small_raw.annotations.description
    assert np.allclose(back.annotations.onset, small_raw.annotations.onset)
    assert np.allclose(back.annotations.duration, small_raw.annotations.duration)
    assert back.meta["format"] == ext[1:]


def test_edf_header_and_partial_read(small_raw, tmp_path):
    path = tmp_path / "x.edf"
    write_raw(small_raw, path)
    hdr = read_edf_header(path)
    assert hdr["n_signals"] == 7 and hdr["subtype"] == "EDF+C"
    part = read_raw(path, include=["Cz", "Pz"], tmin=2.0, tmax=5.0)
    assert part.ch_names == ["Cz", "Pz"] and part.n_times == 3 * 256
    assert np.abs(part.data - small_raw.get_data(["Cz", "Pz"], 2.0, 5.0)).max() < 0.01
    assert part.annotations.description == ["BAD blink"]
    assert part.annotations.onset[0] == pytest.approx(1.25)
    ex = read_raw(path, exclude=["STI"])
    assert "STI" not in ex.ch_names


def test_edf_many_annotations(tmp_path):
    rng = np.random.default_rng(1)
    raw = RawEEG(rng.normal(size=(3, 500)), 100, ["Cz", "Pz", "Oz"])
    for k in range(200):
        raw.annotations.append(k * 0.02, 0.0, f"event number {k} with a long description")
    back = _roundtrip(raw, tmp_path / "many.edf")
    assert len(back.annotations) == 200
    assert back.annotations.description[-1] == "event number 199 with a long description"


def test_edf_odd_length_and_rate(tmp_path):
    rng = np.random.default_rng(2)
    raw = RawEEG(rng.normal(size=(2, 1234)), 250.0, ["Fz", "Cz"])
    back = _roundtrip(raw, tmp_path / "odd.edf")
    assert back.sfreq == 250.0
    assert back.n_times >= 1234 and np.abs(back.data[:, :1234] - raw.data).max() < 1e-3


def test_edf_against_mne(small_raw, tmp_path):
    mne = pytest.importorskip("mne")
    path = tmp_path / "x.edf"
    write_raw(small_raw, path)
    m = mne.io.read_raw_edf(path, preload=True, verbose="error")
    assert m.ch_names == small_raw.ch_names
    assert np.abs(m.get_data()[:5] * 1e6 - small_raw.data[:5]).max() < 0.01
    assert sorted(m.annotations.description) == sorted(small_raw.annotations.description)


def test_read_mne_exported_edf(small_raw, tmp_path):
    mne = pytest.importorskip("mne")
    pytest.importorskip("edfio")
    m = small_raw.pick("eeg").to_mne()
    path = tmp_path / "mne.edf"
    mne.export.export_raw(path, m, fmt="edf", verbose="error")
    back = read_raw(path)
    assert back.ch_names[:5] == ["Fp1", "Fp2", "Cz", "Pz", "O1"]
    n = min(back.n_times, small_raw.n_times)
    assert np.abs(back.data[:5, :n] - small_raw.data[:5, :n]).max() < 0.05
    assert "target" in back.annotations.description


def test_mixed_rate_edf(tmp_path):
    edfio = pytest.importorskip("edfio")
    rng = np.random.default_rng(3)
    sig_fast = rng.normal(0, 10, 2560)
    t = np.arange(20) / 2.0
    sig_slow = 95 + np.sin(t)
    edf = edfio.Edf([edfio.EdfSignal(sig_fast, sampling_frequency=256, label="Cz", physical_dimension="uV"),
                     edfio.EdfSignal(sig_slow, sampling_frequency=2, label="SpO2", physical_dimension="%")])
    path = tmp_path / "mixed.edf"
    edf.write(path)
    raw = read_raw(path)
    assert raw.sfreq == 256 and raw.n_times == 2560
    assert raw.ch_types == ["eeg", "misc"]
    assert np.abs(raw["Cz"] - sig_fast).max() < 0.01
    assert 93 < raw["SpO2"].mean() < 97


def test_brainvision_roundtrip(small_raw, tmp_path):
    back = _roundtrip(small_raw, tmp_path / "bv.vhdr")
    assert back.ch_names == small_raw.ch_names
    assert np.abs(back.data - small_raw.data).max() < 1e-3
    assert "target" in back.annotations.description and "BAD blink" in back.annotations.description
    # reading through the .eeg or .vmrk path also works
    assert read_raw(tmp_path / "bv.eeg").n_times == small_raw.n_times


def test_brainvision_from_pybv(small_raw, tmp_path):
    pybv = pytest.importorskip("pybv")
    data = small_raw.data[:5] * 1e-6
    pybv.write_brainvision(data=data, sfreq=256, ch_names=small_raw.ch_names[:5], fname_base="pb",
                           folder_out=str(tmp_path), events=np.array([[256, 1], [512, 2]]), overwrite=True)
    back = read_raw(tmp_path / "pb.vhdr")
    assert np.abs(back.data - small_raw.data[:5]).max() < 0.01
    assert len(back.annotations.select(regexp="Stimulus")) == 2


def test_brainvision_ascii(tmp_path):
    (tmp_path / "a.vhdr").write_text(
        "Brain Vision Data Exchange Header File Version 1.0\n[Common Infos]\nDataFile=a.dat\n"
        "MarkerFile=a.vmrk\nDataFormat=ASCII\nDataOrientation=MULTIPLEXED\nNumberOfChannels=2\n"
        "SamplingInterval=4000\n[ASCII Infos]\nDecimalSymbol=.\nSkipLines=1\nSkipColumns=0\n"
        "[Channel Infos]\nCh1=Cz,,0.5,µV\nCh2=Pz,,1,mV\n")
    (tmp_path / "a.dat").write_text("Cz Pz\n1 0.001\n2 0.002\n3 0.003\n4 0.004\n")
    (tmp_path / "a.vmrk").write_text("[Marker Infos]\nMk1=Stimulus,S  1,2,1,0\n")
    raw = read_raw(tmp_path / "a.vhdr")
    assert raw.sfreq == 250
    assert np.allclose(raw["Cz"], [0.5, 1.0, 1.5, 2.0])
    assert np.allclose(raw["Pz"], [1, 2, 3, 4])  # mV -> µV
    assert raw.annotations.description == ["Stimulus/S  1"] and raw.annotations.onset[0] == pytest.approx(0.004)


def test_csv_with_time_column(tmp_path):
    t = np.arange(500) / 250.0
    x = np.sin(2 * np.pi * 10 * t) * 20
    path = tmp_path / "rec.csv"
    path.write_text("time,Fp1,Cz,marker\n" + "\n".join(
        f"{ti:.6f},{xi:.4f},{-xi:.4f},{'go' if k == 100 else ''}" for k, (ti, xi) in enumerate(zip(t, x))))
    raw = read_raw(path)
    assert raw.sfreq == pytest.approx(250.0)
    assert raw.ch_names == ["Fp1", "Cz"]
    assert np.allclose(raw["Fp1"], x, atol=1e-3)
    assert raw.annotations.description == ["go"] and raw.annotations.onset[0] == pytest.approx(0.4)


def test_text_formats(tmp_path):
    rng = np.random.default_rng(0)
    x = rng.normal(size=(300, 3))
    tsv = tmp_path / "rec.tsv"
    np.savetxt(tsv, x, delimiter="\t")
    with pytest.raises(ValueError, match="sampling rate"):
        read_raw(tsv)
    raw = read_raw(tsv, sfreq=100)
    assert raw.n_channels == 3 and raw.n_times == 300 and np.allclose(raw.data, x.T)
    # OpenBCI-style export: % comments with the rate, index column and a text timestamp column
    ob = tmp_path / "OpenBCI-RAW.txt"
    lines = ["%OpenBCI Raw EXG Data", "%Number of channels = 2", "%Sample Rate = 250 Hz",
             "Sample Index, EXG Channel 0, EXG Channel 1, Timestamp (Formatted)"]
    lines += [f"{k}, {x[k, 0]:.5f}, {x[k, 1]:.5f}, 2024-01-01 10:00:{k % 60:02d}.000" for k in range(300)]
    ob.write_text("\n".join(lines))
    raw = read_raw(ob)
    assert raw.sfreq == 250 and raw.ch_names == ["EXG Channel 0", "EXG Channel 1"]
    # channels stored as rows with labels in the first column
    rows = tmp_path / "rows.csv"
    rows.write_text("\n".join(f"{n}," + ",".join(f"{v:.5f}" for v in x[:, k])
                              for k, n in enumerate(["Fz", "Cz", "Pz"])))
    raw = read_raw(rows, sfreq=100)
    assert raw.ch_names == ["Fz", "Cz", "Pz"] and np.allclose(raw.data, x.T, atol=1e-4)


def test_text_roundtrip_and_units(small_raw, tmp_path):
    back = _roundtrip(small_raw, tmp_path / "x.csv")
    assert back.sfreq == small_raw.sfreq and back.ch_names == small_raw.ch_names
    assert np.abs(back.data - small_raw.data).max() < 1e-3
    path = tmp_path / "volts.csv"
    path.write_text("Cz,Pz\n" + "\n".join(f"{v * 1e-6},{-v * 1e-6}" for v in range(10)))
    raw = read_raw(path, sfreq=10, unit="V")
    assert np.allclose(raw["Cz"], np.arange(10))


def test_numpy_formats(small_raw, tmp_path):
    raw = small_raw.copy().set_bads(["Pz"])
    back = _roundtrip(raw, tmp_path / "x.npz")
    assert back.ch_names == raw.ch_names and back.ch_types == raw.ch_types and back.bads == ["Pz"]
    assert np.allclose(back.data, raw.data) and set(back.positions) == set(raw.positions)
    assert back.annotations.description == raw.annotations.description
    np.save(tmp_path / "x.npy", raw.data[:5].T)  # samples x channels
    arr = read_raw(tmp_path / "x.npy", sfreq=256, ch_names=raw.ch_names[:5])
    assert arr.n_channels == 5 and np.allclose(arr.data, raw.data[:5])
    with pytest.raises(ValueError):
        read_raw(tmp_path / "x.npy")


def test_matlab_formats(small_raw, tmp_path):
    back = _roundtrip(small_raw, tmp_path / "x.mat")
    assert back.ch_names == small_raw.ch_names and back.sfreq == 256
    assert np.allclose(back.data, small_raw.data)
    from scipy.io import savemat

    savemat(tmp_path / "custom.mat", {"EEGdata": {"signals": small_raw.data[:5].T, "Fs": 256.0,
                                                  "labels": np.array(small_raw.ch_names[:5], dtype=object)}})
    raw = read_raw(tmp_path / "custom.mat")
    assert raw.sfreq == 256 and raw.ch_names == small_raw.ch_names[:5]
    assert np.allclose(raw.data, small_raw.data[:5])


def test_eeglab_set(small_raw, tmp_path):
    mne = pytest.importorskip("mne")
    pytest.importorskip("eeglabio")
    m = small_raw.pick("eeg").to_mne()
    path = tmp_path / "x.set"
    mne.export.export_raw(path, m, fmt="eeglab", verbose="error")
    raw = read_raw(path)
    assert raw.ch_names == ["Fp1", "Fp2", "Cz", "Pz", "O1"]
    assert np.abs(raw.data - small_raw.data[:5]).max() < 1e-3
    assert "target" in raw.annotations.description
    assert len(raw.positions) == 5


def test_hdf5_roundtrip(small_raw, tmp_path):
    pytest.importorskip("h5py")
    back = _roundtrip(small_raw, tmp_path / "x.h5")
    assert back.ch_names == small_raw.ch_names and np.allclose(back.data, small_raw.data)
    assert back.ch_types == small_raw.ch_types


def test_fif_via_mne(small_raw, tmp_path):
    pytest.importorskip("mne")
    back = _roundtrip(small_raw, tmp_path / "x_raw.fif")
    assert back.ch_names == small_raw.ch_names
    assert np.abs(back.data[:5] - small_raw.data[:5]).max() < 1e-3
    assert len(back.positions) == 5


def test_xdf_with_fake_pyxdf(monkeypatch, tmp_path):
    rng = np.random.default_rng(0)
    ts = rng.normal(size=(1000, 3))
    stamps = 100.0 + np.arange(1000) / 500.0
    eeg_stream = {"info": {"name": ["amp"], "type": ["EEG"], "nominal_srate": ["500"],
                           "desc": [{"channels": [{"channel": [{"label": ["Fz"]}, {"label": ["Cz"]},
                                                                {"label": ["Pz"]}]}]}]},
                  "time_series": ts, "time_stamps": stamps}
    marker_stream = {"info": {"name": ["markers"], "type": ["Markers"], "nominal_srate": ["0"]},
                     "time_series": [["start"], ["stop"]], "time_stamps": np.array([100.5, 101.0])}
    fake = types.SimpleNamespace(load_xdf=lambda path: ([eeg_stream, marker_stream], {}))
    monkeypatch.setitem(sys.modules, "pyxdf", fake)
    path = tmp_path / "rec.xdf"
    path.write_bytes(b"XDF:")
    raw = read_raw(path)
    assert raw.sfreq == 500 and raw.ch_names == ["Fz", "Cz", "Pz"]
    assert np.allclose(raw.data, ts.T)
    assert raw.annotations.description == ["start", "stop"]
    assert raw.annotations.onset.tolist() == pytest.approx([0.5, 1.0])


def test_sniff_unknown_extension(small_raw, tmp_path):
    write_raw(small_raw, tmp_path / "x.edf")
    shutil.copy(tmp_path / "x.edf", tmp_path / "recording.data_file")
    raw = read_raw(tmp_path / "recording.data_file")
    assert raw.meta["format"] == "edf"


def test_errors_and_listing(small_raw, tmp_path):
    with pytest.raises(FileNotFoundError):
        read_raw(tmp_path / "missing.edf")
    with pytest.raises(ValueError):
        write_raw(small_raw, tmp_path / "x.unknown")
    rows = supported_formats()
    exts = {r["extension"] for r in rows}
    assert {".edf", ".bdf", ".vhdr", ".csv", ".mat", ".set", ".npz", ".fif", ".xdf"} <= exts
    assert write_raw(small_raw, tmp_path / "noext", fmt="npz").endswith(".npz")
    assert os.path.exists(tmp_path / "noext.npz")


@pytest.mark.parametrize("ext", [".edf", ".bdf", ".vhdr", ".csv", ".tsv", ".npz", ".mat", ".h5", ".fif"])
def test_every_writer_preserves_channels_and_annotations(ext, tmp_path):
    if ext == ".h5":
        pytest.importorskip("h5py")
    if ext == ".fif":
        pytest.importorskip("mne")
    from eegproc.datasets import simulate_eeg

    raw = simulate_eeg(20, 128, "clinical19", task="oddball", eog=True, stim_channel=True, seed=5)
    back = _roundtrip(raw, tmp_path / f"rec{ext}")
    assert back.ch_names == raw.ch_names and back.ch_types == raw.ch_types
    assert back.annotations.description == raw.annotations.description
    assert np.abs(back.get_data("eeg") - raw.get_data("eeg")).max() < 0.01


def test_eeglab_epoched_struct(tmp_path):
    from scipy.io import savemat

    rng = np.random.default_rng(0)
    nbchan, pnts, trials = 3, 100, 4
    data = rng.normal(size=(nbchan, pnts, trials))
    chanlocs = np.array([{"labels": n, "X": x, "Y": y, "Z": z}
                         for n, (x, y, z) in zip(["Fz", "Cz", "Pz"], [(0.7, 0, 0.7), (0, 0, 1), (-0.7, 0, 0.7)])],
                        dtype=object)
    events = np.array([{"type": "stim", "latency": 51.0}, {"type": "stim", "latency": 151.0}], dtype=object)
    savemat(tmp_path / "ep.set", {"EEG": {"srate": 100.0, "nbchan": nbchan, "pnts": pnts, "trials": trials,
                                          "data": data, "chanlocs": chanlocs, "event": events}})
    raw = read_raw(tmp_path / "ep.set")
    assert raw.n_times == pnts * trials and raw.ch_names == ["Fz", "Cz", "Pz"]
    assert np.allclose(raw.data[:, pnts:2 * pnts], data[:, :, 1])
    assert raw.annotations.count() == {"BAD boundary": 3, "stim": 2}
    assert raw.positions["Fz"][1] > 0  # EEGLAB +X (nose) -> our +y
