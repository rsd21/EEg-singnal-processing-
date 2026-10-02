import numpy as np
import pytest

from eegproc import RawEEG
from eegproc._spline import csd_matrix, interpolation_matrix
from eegproc.core.montage import Montage
from eegproc.preprocessing import (ICA, annotate_amplitude, annotate_muscle, bipolar_reference, detect_line_noise,
                                   fastica, filter_data, find_bad_channels, find_blinks, interpolate_bads,
                                   laplacian, notch_filter_data, remove_artifacts_ica, resample_raw,
                                   set_reference)

FS = 250.0
T = np.arange(int(10 * FS)) / FS


def _amp(x, f):
    """Amplitude of frequency f in x (least squares)."""
    m = np.column_stack([np.sin(2 * np.pi * f * T), np.cos(2 * np.pi * f * T)])
    coef, *_ = np.linalg.lstsq(m, x, rcond=None)
    return np.hypot(*coef)


@pytest.mark.parametrize("method", ["iir", "fir"])
def test_bandpass_attenuation(method):
    x = np.sin(2 * np.pi * 2 * T) + np.sin(2 * np.pi * 10 * T) + np.sin(2 * np.pi * 60 * T)
    y = filter_data(x, FS, 8.0, 13.0, method=method)
    mid = slice(500, -500)
    T_mid = T[mid]

    def amp(sig, f):
        m = np.column_stack([np.sin(2 * np.pi * f * T_mid), np.cos(2 * np.pi * f * T_mid)])
        return np.hypot(*np.linalg.lstsq(m, sig[mid], rcond=None)[0])

    assert amp(y, 10) == pytest.approx(1.0, abs=0.05)
    assert amp(y, 2) < 0.02 and amp(y, 60) < 0.02


def test_highpass_lowpass_bandstop_causal():
    x = np.sin(2 * np.pi * 0.2 * T) + np.sin(2 * np.pi * 20 * T)
    assert _amp(filter_data(x, FS, 2.0, None), 0.2) < 0.05
    assert _amp(filter_data(x, FS, None, 5.0), 20) < 0.01
    y = filter_data(np.sin(2 * np.pi * 10 * T), FS, 12.0, 8.0)  # l > h -> band-stop
    assert _amp(y, 10) < 0.1
    causal = filter_data(x, FS, None, 5.0, phase="causal")
    assert causal.shape == x.shape
    with pytest.raises(ValueError):
        filter_data(x, FS, 1.0, 5.0, method="magic")


def test_zero_phase_keeps_latency():
    x = np.zeros(2000)
    x[1000] = 1.0
    y = filter_data(x, FS, 1.0, 30.0)
    assert abs(int(np.argmax(np.abs(y))) - 1000) <= 1


def test_notch_and_line_detection():
    rng = np.random.default_rng(0)
    x = rng.normal(size=(3, T.size)) + 5 * np.sin(2 * np.pi * 50 * T) + 2 * np.sin(2 * np.pi * 100 * T)
    assert detect_line_noise(x, FS) == 50.0
    y = notch_filter_data(x, FS, 50.0)
    assert _amp(y[0], 50) < 0.2 and _amp(y[0], 100) < 0.2
    assert detect_line_noise(rng.normal(size=(3, T.size)), FS) is None


def test_filter_raw_skips_stim(small_raw):
    out = small_raw.filter(1.0, 30.0)
    assert np.array_equal(out["STI"], small_raw["STI"])
    assert out.meta["highpass"] == 1.0 and out.meta["lowpass"] == 30.0
    assert not np.allclose(out["Cz"], small_raw["Cz"])
    n = small_raw.notch_filter(50)
    assert "notch" in n.history[-1]


def test_resample_preserves_signal_and_events():
    x = np.sin(2 * np.pi * 5 * T)
    stim = np.zeros_like(T)
    stim[1000:1003] = 4
    raw = RawEEG(np.vstack([x, stim]), FS, ["Cz", "STI"], ["eeg", "stim"])
    r = resample_raw(raw, 100.0)
    assert r.sfreq == 100.0 and r.n_times == 1000
    t2 = np.arange(1000) / 100
    assert np.abs(r["Cz"][50:-50] - np.sin(2 * np.pi * 5 * t2)[50:-50]).max() < 0.02
    onset = np.flatnonzero(r["STI"])
    assert onset[0] == 400 and r["STI"][onset[0]] == 4
    assert r.meta["lowpass"] == 50.0


def test_average_and_channel_reference(clean_raw):
    avg = set_reference(clean_raw, "average")
    eeg = avg.pick_indices("eeg")
    assert np.allclose(avg.data[eeg].mean(axis=0), 0, atol=1e-9)
    assert np.array_equal(avg["VEOG"], clean_raw["VEOG"])  # only EEG changes
    cz = set_reference(clean_raw, "Cz")
    assert np.allclose(cz["Cz"], 0)
    linked = set_reference(clean_raw, ["P7", "P8"], drop_ref=True)
    assert "P7" not in linked.ch_names and linked.meta["reference"] == "P7+P8"


def test_bipolar(clean_raw):
    bp = bipolar_reference(clean_raw, "double_banana")
    assert "Fp1-F7" in bp.ch_names and bp.n_channels == 18
    assert np.allclose(bp["Fp1-F7"], clean_raw["Fp1"] - clean_raw["F7"])
    with pytest.raises(ValueError):
        bipolar_reference(clean_raw, [("X1", "X2")])


def test_laplacians(clean_raw):
    h = laplacian(clean_raw, "hjorth")
    s = laplacian(clean_raw, "spline")
    assert h.data.shape == clean_raw.data.shape and s.units[0] == "uV/r2"
    # a spatially constant potential has zero surface Laplacian
    pos = clean_raw.get_positions("eeg")
    csd = csd_matrix(pos) @ np.ones(len(pos))
    assert np.abs(csd).max() < 1e-6
    no_pos = RawEEG(np.zeros((6, 100)), 100, [f"E{i}" for i in range(6)])
    with pytest.raises(ValueError):
        laplacian(no_pos)


def test_spline_interpolation_accuracy():
    m = Montage.standard("standard_1010")
    names = [n for n in m.ch_names if n not in ("A1", "A2", "M1", "M2")]
    field = lambda p: p[:, 2] * 10 + p[:, 1] * 5  # smooth potential  # noqa: E731
    target = ["Cz", "P3", "F4"]
    src = [n for n in names if n not in target]
    mat = interpolation_matrix(m.as_array(src), m.as_array(target))
    est = mat @ field(m.as_array(src))
    assert np.allclose(est, field(m.as_array(target)), atol=0.2)
    assert np.allclose(mat.sum(axis=1), 1.0, atol=1e-6)  # constants are reproduced


def test_interpolate_bads_restores_channel(clean_raw):
    raw = clean_raw.copy()
    raw.data[raw.channel_index("C3")] = 0.0
    raw.set_bads(["C3"])
    fixed = interpolate_bads(raw)
    r = np.corrcoef(filter_data(fixed["C3"], 256, 1, 30), filter_data(clean_raw["C3"], 256, 1, 30))[0, 1]
    assert r > 0.8 and fixed.bads == []
    raw.positions.pop("C3")
    with pytest.raises(ValueError):
        interpolate_bads(raw)


def test_find_bad_channels(bad_raw, clean_raw):
    res = find_bad_channels(bad_raw, random_state=0)
    for ch in ("T8", "O2", "P4"):
        assert ch in res.bads, res.by_method
    assert "flat" in res.reasons()["O2"]
    assert find_bad_channels(clean_raw).bads == []
    assert "channel" in res.summary()


def test_ransac_on_dense_montage():
    from eegproc.datasets import simulate_eeg

    raw = simulate_eeg(60, 128, "64", bad_channels={"CP3": "disconnected"}, seed=3, blinks=False)
    res = find_bad_channels(raw, methods=("ransac",), random_state=0)
    assert res.bads == ["CP3"]


def test_amplitude_and_muscle_annotations(clean_raw):
    raw = clean_raw.copy()
    raw.data[raw.channel_index("Cz"), 2560:2600] += 400.0
    ann = annotate_amplitude(raw, peak_to_peak=250)
    assert any(o <= 10.1 <= o + d for o, d in zip(ann.onset, ann.duration))
    rng = np.random.default_rng(0)
    burst = slice(int(30 * 256), int(32 * 256))
    raw.data[raw.pick_indices("eeg"), burst] += 30 * rng.normal(size=(19, 512))
    m = annotate_muscle(raw)
    assert len(m) >= 1 and any(o < 32 and o + d > 30 for o, d in zip(m.onset, m.duration))


def test_find_blinks(clean_raw):
    blinks = find_blinks(clean_raw)
    assert 5 <= len(blinks) <= 40


def test_fastica_recovers_sources():
    rng = np.random.default_rng(0)
    n = 5000
    s = np.vstack([np.sign(np.sin(np.linspace(0, 80, n))), rng.laplace(size=n), rng.uniform(-1, 1, n)])
    a = rng.normal(size=(3, 3))
    x = a @ s
    x = x - x.mean(axis=1, keepdims=True)
    evals, evecs = np.linalg.eigh(np.cov(x))
    white = (evecs / np.sqrt(evals)).T
    w, _, ok = fastica(white @ x, random_state=0)
    est = w @ white @ x
    corr = np.abs(np.corrcoef(np.vstack([s, est]))[:3, 3:])
    assert ok and np.all(corr.max(axis=1) > 0.95)


def test_ica_removes_blinks(clean_raw):
    ica = ICA(n_components=0.99, random_state=0).fit(clean_raw)
    bad, scores = ica.find_bads_eog(clean_raw)
    assert len(bad) >= 1 and scores[bad[0]] > 0.8
    ica.exclude = bad
    clean = ica.apply(clean_raw)
    blinks = find_blinks(clean_raw, channel="VEOG")
    pk = ((blinks.onset + blinks.duration / 2) * 256).astype(int)
    before = np.abs(filter_data(clean_raw["Fp1"], 256, 1, 10)[pk]).mean()
    after = np.abs(filter_data(clean["Fp1"], 256, 1, 10)[pk]).mean()
    assert after < 0.3 * before
    r = np.corrcoef(filter_data(clean["O1"], 256, 1, 30), filter_data(clean_raw["O1"], 256, 1, 30))[0, 1]
    assert r > 0.95  # posterior channels barely touched
    table = ica.component_table(clean_raw)
    assert table[bad[0]]["label"] == "eye"
    cleaned, ica2 = remove_artifacts_ica(clean_raw)
    assert ica2.exclude and "ICA removed" in cleaned.history[-1]
