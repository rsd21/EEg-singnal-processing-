import numpy as np
import pytest

from eegproc import RawEEG, compute_connectivity, compute_psd, extract_features, morlet_power
from eegproc.analysis import (fit_aperiodic, higuchi_fd, hjorth, permutation_entropy, sample_entropy,
                              spectrogram, summarize_bands)
from eegproc.analysis.spectral import Spectrum

FS = 256.0


def _pink(n, exponent, rng):
    f = np.fft.rfftfreq(n, 1 / FS)
    spec = (rng.normal(size=f.size) + 1j * rng.normal(size=f.size)) / np.sqrt(np.maximum(f, 1e-3) ** exponent)
    spec[0] = 0
    return np.fft.irfft(spec, n)


@pytest.mark.parametrize("method", ["welch", "multitaper", "periodogram"])
def test_psd_peak_and_parseval(method):
    t = np.arange(int(20 * FS)) / FS
    x = 3 * np.sin(2 * np.pi * 10 * t) + np.random.default_rng(0).normal(0, 1, t.size)
    spec = compute_psd(x[None], sfreq=FS, method=method)
    assert spec.freqs[np.argmax(spec.data[0])] == pytest.approx(10, abs=0.3)
    total = np.trapezoid(spec.data[0], spec.freqs) if hasattr(np, "trapezoid") else np.trapz(spec.data[0], spec.freqs)
    assert total == pytest.approx(x.var(), rel=0.1)
    bp = spec.band_power({"alpha": (8, 12)})["alpha"][0]
    assert bp == pytest.approx(4.5 + 4 / (FS / 2) * 1, rel=0.15)  # A^2/2 + noise in band


def test_spectrum_measures():
    rng = np.random.default_rng(1)
    n = int(60 * FS)
    x = _pink(n, 1.5, rng)
    spec = compute_psd(x[None], sfreq=FS, window_sec=4)
    _, expo = fit_aperiodic(spec.freqs, spec.data, 2, 40)
    assert expo[0] == pytest.approx(1.5, abs=0.2)
    white = compute_psd(rng.normal(size=(1, n)), sfreq=FS)
    t = np.arange(n) / FS
    tone = compute_psd(np.sin(2 * np.pi * 10 * t)[None] + 0.01 * rng.normal(size=n), sfreq=FS)
    assert white.spectral_entropy()[0] > 0.95 > tone.spectral_entropy()[0]
    assert tone.peak_frequency(7, 14)[0] == pytest.approx(10, abs=0.3)
    assert tone.spectral_edge(0.95)[0] == pytest.approx(10, abs=1.0)
    rel = tone.band_power(relative=True)
    assert rel["alpha"][0] > 0.95
    assert "leading_channel" in summarize_bands(tone)


def test_psd_of_raw_skips_bad_segments(clean_raw):
    raw = clean_raw.copy()
    raw.data[:, 2560:5120] += 1000.0 * np.sin(2 * np.pi * 20 * np.arange(2560) / 256)
    raw.annotations.append(10.0, 10.0, "BAD_artifact")
    spec = raw.compute_psd(picks="eeg")
    assert isinstance(spec, Spectrum) and spec.ch_names == raw.pick_names("eeg")
    beta = spec.band_power({"b": (19, 21)})["b"]
    ref = clean_raw.compute_psd(picks="eeg").band_power({"b": (19, 21)})["b"]
    assert np.all(beta < 3 * ref)


def test_morlet_and_spectrogram():
    t = np.arange(int(4 * FS)) / FS
    x = 2 * np.sin(2 * np.pi * 10 * t)
    p = morlet_power(x, FS, [10.0, 30.0])
    assert np.median(p[0, 256:-256]) == pytest.approx(4.0, rel=0.05)
    assert np.median(p[1, 256:-256]) < 0.01
    times, freqs, sxx = spectrogram(x, FS, window_sec=1, fmax=40)
    assert freqs[np.argmax(sxx.mean(axis=1))] == pytest.approx(10, abs=1)


def test_time_and_complexity_features():
    rng = np.random.default_rng(2)
    t = np.arange(2000) / FS
    sine = np.sin(2 * np.pi * 8 * t)
    noise = rng.normal(size=2000)
    act, mob, comp = hjorth(np.vstack([sine, noise]))
    assert mob[0] == pytest.approx(2 * np.pi * 8 / FS, rel=0.02)
    assert permutation_entropy(noise) > 0.99 and permutation_entropy(np.arange(100.0)) == 0
    assert sample_entropy(sine, max_samples=1000) < sample_entropy(noise, max_samples=1000)
    assert higuchi_fd(noise) == pytest.approx(2.0, abs=0.1)
    assert higuchi_fd(sine) < 1.2


def test_extract_features(clean_raw):
    table = extract_features(clean_raw, include=("time", "spectral", "complexity"))
    assert len(table) == 19
    for col in ("rms", "hjorth_mobility", "rel_alpha", "peak_alpha_freq", "aperiodic_exponent",
                "sample_entropy", "higuchi_fd"):
        assert col in table.columns
    arr = table.to_array()
    assert arr.shape[0] == 19 and np.isfinite(arr).all()
    o1 = next(r for r in table.rows if r["channel"] == "O1")
    fz = next(r for r in table.rows if r["channel"] == "Fz")
    assert o1["rel_alpha"] > fz["rel_alpha"]  # posterior alpha


def test_connectivity_lagged_coupling():
    rng = np.random.default_rng(3)
    n = int(60 * FS)
    t = np.arange(n) / FS
    common = np.sin(2 * np.pi * 10 * t + np.cumsum(rng.normal(0, 0.05, n)))
    lag = int(0.02 * FS)
    a = common + 0.5 * rng.normal(size=n)
    b = np.roll(common, lag) + 0.5 * rng.normal(size=n)
    c = rng.normal(size=n)
    raw = RawEEG(np.vstack([a, b, c]), FS, ["A", "B", "C"])
    for method in ("coherence", "imcoh", "plv", "pli", "wpli", "correlation", "aec"):
        con = compute_connectivity(raw, method=method, band=(8, 12))
        m = con.matrix
        assert np.allclose(m, m.T, atol=1e-9), method
        if method not in ("correlation", "aec"):
            assert abs(m[0, 1]) > 2 * abs(m[0, 2]), method
    wpli = compute_connectivity(raw, method="wpli", band=(8, 12))
    assert wpli.matrix[0, 1] > 0.8 and wpli.node_strength()[2] < wpli.node_strength()[0]
    assert wpli.strongest(1)[0]["channel_a"] == "A"
    with pytest.raises(ValueError):
        compute_connectivity(raw, method="magic")
