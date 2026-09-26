import numpy as np
import pytest

from eegproc import Annotations, Epochs, Montage, RawEEG
from eegproc.core.annotations import auto_event_id
from eegproc.core.channels import clean_channel_name, clean_channel_names, infer_channel_type, voltage_scale_to_uv
from eegproc.core.epochs import find_events, find_stim_events, make_fixed_length_epochs
from eegproc.core.montage import fit_sphere, project_to_2d, standard_name, unproject_from_2d


# ------------------------------------------------------------------ channels
@pytest.mark.parametrize("label, expected", [
    ("EEG Fp1-REF", "Fp1"), ("FP1", "Fp1"), ("CZ", "Cz"), ("Fc5.", "FC5"), ("Cz..", "Cz"),
    ("T3", "T7"), ("C3-A2", "C3"), ("Fp1-F7", "Fp1-F7"), ("EEG Oz-LE", "Oz"), ("afz", "AFz"),
    ("ECG", "ECG"), ("  Pz ", "Pz"),
])
def test_clean_channel_name(label, expected):
    assert clean_channel_name(label) == expected


def test_clean_names_unique():
    assert clean_channel_names(["FP1", "Fp1", "EEG FP1-REF"]) == ["Fp1", "Fp1-1", "Fp1-2"]


@pytest.mark.parametrize("label, kind", [
    ("Fp1", "eeg"), ("EEG C3-REF", "eeg"), ("HEOG", "eog"), ("EOG left", "eog"), ("ECG", "ecg"),
    ("EKG1", "ecg"), ("EMG chin", "emg"), ("Status", "stim"), ("STI 014", "stim"), ("Resp", "resp"),
    ("SpO2", "misc"), ("Accel X", "misc"), ("EXG Channel 3", "eeg"),
])
def test_infer_channel_type(label, kind):
    assert infer_channel_type(label) == kind


def test_voltage_units():
    assert voltage_scale_to_uv("V") == 1e6
    assert voltage_scale_to_uv("mV") == 1e3
    assert voltage_scale_to_uv("µV") == 1.0
    assert voltage_scale_to_uv("uV") == 1.0
    assert voltage_scale_to_uv("degC") is None


# ------------------------------------------------------------------- montage
def test_standard_montage_geometry():
    m = Montage.standard("standard_1010")
    assert np.allclose(m.get("Cz"), [0, 0, 1])
    for name in m.ch_names:
        assert np.isclose(np.linalg.norm(m.get(name)), 1.0)
    fp1, fp2 = m.get("Fp1"), m.get("Fp2")
    assert np.isclose(fp1[0], -fp2[0]) and np.isclose(fp1[1], fp2[1])
    assert m.get("C3")[0] < 0 < m.get("C4")[0]          # left / right
    assert m.get("Fz")[1] > 0 > m.get("Pz")[1]          # front / back
    assert np.allclose(m.get("T3"), m.get("T7"))        # alias
    assert len(Montage.standard("standard_1020")) == 23
    assert standard_name("fcz") == "FCz"


def test_projection_roundtrip():
    m = Montage.standard("standard_1010")
    xyz = m.as_array()
    xy = project_to_2d(xyz)
    assert np.allclose(unproject_from_2d(xy), xyz, atol=1e-9)
    r = np.hypot(*project_to_2d(m.get("T7"))[0])
    assert np.isclose(r, 0.8)


def test_fit_sphere():
    rng = np.random.default_rng(0)
    pts = rng.normal(size=(50, 3))
    pts = pts / np.linalg.norm(pts, axis=1, keepdims=True) * 9.5 + [1, 2, 3]
    c, r = fit_sphere(pts)
    assert np.allclose(c, [1, 2, 3], atol=1e-6) and np.isclose(r, 9.5)


def test_montage_from_files(tmp_path):
    std = Montage.standard("standard_1010")
    names = ["Fp1", "Fp2", "Cz", "Pz", "O1", "T7"]
    csv = tmp_path / "pos.csv"
    csv.write_text("name,x,y,z\n" + "\n".join(f"{n},{9 * p[0]},{9 * p[1]},{9 * p[2]}"
                                             for n, p in ((n, std.get(n)) for n in names)))
    m = Montage.from_file(csv)
    assert all(np.allclose(m.get(n), std.get(n), atol=1e-6) for n in names)
    elc = tmp_path / "pos.elc"
    elc.write_text("NumberPositions= 3\nUnitPosition mm\nPositions\n" +
                   "\n".join(f"{n}: {90 * std.get(n)[0]} {90 * std.get(n)[1]} {90 * std.get(n)[2]}"
                             for n in names[:3]) + "\n")
    m2 = Montage.from_file(elc)
    assert set(m2.ch_names) == set(names[:3])
    loc = tmp_path / "pos.loc"
    loc.write_text("1\t-18\t0.511\tFp1\n2\t18\t0.511\tFp2\n3\t0\t0\tCz\n4\t-90\t0.25\tC3\n")
    m3 = Montage.from_file(loc)
    assert np.allclose(m3.get("Cz"), [0, 0, 1])
    assert m3.get("Fp1")[0] < 0 < m3.get("Fp2")[0] and m3.get("Fp1")[1] > 0
    assert m3.get("C3")[0] < 0


# ------------------------------------------------------------------- RawEEG
def test_raw_validation():
    with pytest.raises(ValueError):
        RawEEG(np.zeros((2, 10)), 100, ["a", "a"])
    with pytest.raises(ValueError):
        RawEEG(np.zeros((2, 10)), 0)
    with pytest.raises(ValueError):
        RawEEG(np.zeros((2, 10)), 100, ch_types=["eeg", "banana"])
    raw = RawEEG(np.zeros(10), 100)
    assert raw.n_channels == 1 and raw.ch_names == ["Ch1"]


def test_picks(small_raw):
    raw = small_raw
    assert raw.pick_indices("eeg") == [0, 1, 2, 3, 4]
    assert raw.pick_indices("stim") == [5]
    assert raw.pick_indices("data") == [0, 1, 2, 3, 4]
    assert raw.pick_indices(["Cz", 0, "stim"]) == [2, 0, 5]
    assert raw.pick_indices("cz") == [2]
    assert raw.pick_indices("EEG FP1-REF") == [0]
    assert raw.pick_indices(slice(1, 3)) == [1, 2]
    raw2 = raw.copy().set_bads(["Pz"])
    assert raw2.pick_indices("good") == [0, 1, 2, 4]
    assert raw2.pick_indices("eeg", exclude_bads=True) == [0, 1, 2, 4]
    with pytest.raises(KeyError):
        raw.pick_indices("XYZ")


def test_access_and_slicing(small_raw):
    raw = small_raw
    assert raw["Cz"].shape == (raw.n_times,)
    assert raw[["Fp1", "Cz"], 0:100].shape == (2, 100)
    seg = raw.get_data(["Pz"], tmin=1.0, tmax=2.0)
    assert seg.shape == (1, 256)
    assert np.array_equal(seg[0], raw.data[3, 256:512])
    assert raw.duration == pytest.approx(raw.n_times / 256)


def test_copy_independent(small_raw):
    raw2 = small_raw.copy()
    raw2.data[0, 0] = 1e6
    raw2.annotations.append(0, 0, "x")
    assert small_raw.data[0, 0] != 1e6
    assert len(small_raw.annotations) == 2


def test_crop_shifts_annotations(small_raw):
    c = small_raw.crop(1.0, 5.0)
    assert c.duration == pytest.approx(4.0)
    assert c.annotations.onset[0] == pytest.approx(0.5)
    assert "crop" in c.history[-1]
    with pytest.raises(ValueError):
        small_raw.crop(5, 1)


def test_pick_drop_rename_types(small_raw):
    raw = small_raw.pick(["Cz", "Pz"])
    assert raw.ch_names == ["Cz", "Pz"] and set(raw.positions) == {"Cz", "Pz"}
    raw = small_raw.drop_channels("STI")
    assert "STI" not in raw.ch_names
    raw = small_raw.copy().rename_channels({"Cz": "CZ_new"})
    assert "CZ_new" in raw.ch_names and "CZ_new" in raw.positions
    raw.set_channel_types({"Fp1": "eog"})
    assert raw.get_channel_type("Fp1") == "eog"


def test_append_and_add_channels(small_raw):
    both = small_raw.append(small_raw)
    assert both.n_times == 2 * small_raw.n_times
    assert "BAD boundary" in both.annotations.description
    added = small_raw.add_channels(np.ones((1, small_raw.n_times)), ["EXTRA"], ["misc"])
    assert added.ch_names[-1] == "EXTRA" and added.ch_types[-1] == "misc"


def test_describe_summary(small_raw):
    text = small_raw.describe()
    assert "Sampling" in text and "Fp1" in text
    s = small_raw.summary()
    assert s["n_channels"] == 6 and s["channel_types"] == {"eeg": 5, "stim": 1}
    pytest.importorskip("pandas")
    assert small_raw.to_dataframe().shape == (small_raw.n_times, 7)


def test_set_montage(small_raw):
    raw = small_raw.copy()
    raw.positions = {}
    raw.set_montage("standard_1020")
    assert raw.has_positions("eeg")
    raw2 = RawEEG(np.zeros((2, 10)), 100, ["Cz", "Weird"])
    with pytest.raises(ValueError):
        raw2.set_montage("standard_1020", on_missing="raise")


# -------------------------------------------------------------- annotations
def test_annotations_events():
    ann = Annotations([1.0, 2.0, 3.0, 4.0], 0, ["S  1", "S  2", "target", "BAD x"])
    events, eid = ann.to_events(100)
    assert eid == {"S  1": 1, "S  2": 2, "target": 3}
    assert events[:, 0].tolist() == [100, 200, 300]
    assert auto_event_id(["left", "right"]) == {"left": 1, "right": 2}
    back = Annotations.from_events(events, 100, {1: "a"})
    assert back.description[0] == "a" and back.onset[1] == pytest.approx(2.0)


def test_annotations_crop_and_mask():
    ann = Annotations([0.5, 2.0, 9.0], [1.0, 1.0, 0.0], ["BAD_a", "ev", "late"])
    c = ann.crop(1.0, 5.0)
    assert c.description == ["BAD_a", "ev"]
    assert c.onset.tolist() == [0.0, 1.0] and c.duration[0] == pytest.approx(0.5)
    mask = ann.bad_mask(1000, 100)
    assert mask[60] and not mask[250]
    assert ann.count() == {"BAD_a": 1, "ev": 1, "late": 1}


# ------------------------------------------------------------------- epochs
def test_find_stim_events():
    x = np.zeros(100)
    x[10:13] = 1
    x[40:45] = 2
    x[45:50] = 3
    ev = find_stim_events(x)
    assert ev[:, 0].tolist() == [10, 40, 45] and ev[:, 2].tolist() == [1, 2, 3]
    y = np.full(100, 255.0)
    y[20:22] = 1
    assert find_stim_events(y)[:, 2].tolist() == [1]  # non-zero idle level ignored


def test_find_events_sources(small_raw):
    ev, eid = find_events(small_raw)
    assert eid == {"target": 1}                          # annotations win
    ev, eid = find_events(small_raw, source="stim")
    assert ev[:, 0].tolist() == [300, 1000] and set(eid.values()) == {3, 7}


def test_epochs(small_raw):
    events = np.array([[512, 0, 1], [1024, 0, 2], [2000, 0, 1], [10, 0, 1]])
    ep = Epochs(small_raw, events, {"a": 1, "b": 2}, tmin=-0.1, tmax=0.4, baseline=(None, 0),
                reject_by_annotation=False)
    assert ep.n_epochs == 3 and ep.drop_log.count("OUT_OF_BOUNDS") == 1
    assert ep.data.shape[1] == 5  # data channels only
    pre = ep.times <= 0
    assert np.allclose(ep.data[:, :, pre].mean(axis=-1), 0, atol=1e-9)
    assert len(ep["a"]) == 2 and len(ep[["a", "b"]]) == 3
    evk = ep.average()
    assert evk.data.shape == (5, len(ep.times)) and evk.nave == 3
    ch, lat, amp = evk.get_peak()
    assert ch in ep.ch_names
    rej = Epochs(small_raw, events, {"a": 1, "b": 2}, tmin=-0.1, tmax=0.4, reject=1.0,
                 reject_by_annotation=False)
    assert rej.n_epochs == 0 and rej.n_dropped == 4


def test_epochs_reject_by_annotation(small_raw):
    events = np.array([[int(3.4 * 256), 0, 1], [int(6 * 256), 0, 1]])
    ep = Epochs(small_raw, events, {"a": 1}, tmin=0, tmax=0.2)
    assert ep.n_epochs == 1 and "BAD_ANNOTATION" in ep.drop_log


def test_fixed_length_epochs_and_from_array(small_raw):
    ep = make_fixed_length_epochs(small_raw, 2.0)
    assert ep.data.shape[2] == 512 and ep.n_epochs == int(small_raw.duration // 2) - 1  # BAD blink drops one
    arr = Epochs.from_array(np.zeros((4, 3, 50)), 100, ["a", "b", "c"], labels=[1, 2, 1, 2])
    assert arr.event_id == {"1": 1, "2": 2} and len(arr["1"]) == 2
