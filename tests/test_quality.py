import json

import numpy as np
import pytest

from eegproc import Epochs, RawEEG, assess_channel_quality, rank_channels
from eegproc.quality import (ChannelRanking, ShrinkageLDA, band_activity, cross_val_accuracy, decoding_accuracy,
                             discriminability, dominant_bands, erd_ers, erp_snr, fisher_score)
from eegproc.quality.ranking import ledoit_wolf


def test_clean_recording_is_all_good(clean_raw):
    rep = assess_channel_quality(clean_raw)
    assert rep.by_grade()["bad"] == [] and rep.by_grade()["poor"] == []
    assert rep.overall_score > 90
    assert rep.bad_channels == []
    assert rep.info["line_freq"] == 50.0


def test_injected_bad_channels_rank_last(bad_raw, injected):
    rep = assess_channel_quality(bad_raw)
    worst = set(rep.worst(len(injected)))
    assert worst == set(injected)
    for ch in ("T8", "O2"):
        assert rep[ch]["grade"] == "bad" and rep[ch]["suggest_bad"]
    assert rep["O2"]["score"] == 0 and "flat" in rep["O2"]["reasons"][0]
    assert any("line noise" in r for r in rep["P4"]["reasons"])
    assert any("drift" in r for r in rep["F7"]["reasons"])
    assert any("pops" in r or "artifacts" in r for r in rep["C3"]["reasons"])
    good = [c for c in rep.ch_names if c not in injected]
    assert all(rep[c]["score"] >= 75 for c in good)


def test_all_bad_kinds_detected():
    from eegproc.datasets import BAD_KINDS, simulate_eeg

    chans = ["F3", "F4", "C3", "C4", "P3", "P4", "O1", "O2"]
    bads = dict(zip(chans, BAD_KINDS))
    raw = simulate_eeg(90, 256, "64", bad_channels=bads, seed=4, blinks=False)
    rep = assess_channel_quality(raw)
    for ch, kind in bads.items():
        assert rep[ch]["score"] < 75, (ch, kind, rep[ch])
    assert set(rep.worst(len(bads))) == set(bads)


def test_report_outputs(bad_raw, tmp_path):
    rep = assess_channel_quality(bad_raw)
    text = rep.summary()
    assert "Suggested bad" in text and "O2" in text
    rep.to_csv(tmp_path / "q.csv")
    rep.to_json(tmp_path / "q.json")
    data = json.loads((tmp_path / "q.json").read_text())
    assert data["channels"][0]["rank"] == 1 and len(data["channels"]) == 19
    if __import__("importlib").util.find_spec("pandas"):
        assert rep.to_dataframe().shape[0] == 19
    assert "<ChannelQualityReport" in repr(rep)


def test_quality_without_positions_and_few_channels():
    from eegproc.datasets import simulate_eeg

    raw = simulate_eeg(60, 200, "clinical19", bad_channels={"Cz": "noisy"}, seed=5, blinks=False)
    anon = RawEEG(raw.data, raw.sfreq, [f"E{i}" for i in range(raw.n_channels)], ["eeg"] * raw.n_channels)
    rep = assess_channel_quality(anon)
    assert rep.worst(1) == [f"E{raw.ch_names.index('Cz')}"]
    two = RawEEG(raw.data[:2], raw.sfreq, ["A", "B"])
    rep2 = assess_channel_quality(two)
    assert len(rep2.rows) == 2
    low = RawEEG(raw.data[:, : 60 * 200], 100.0, raw.ch_names)  # 100 Hz: no >45 Hz band beyond nyquist-ish
    assert len(assess_channel_quality(low).rows) == 19


def test_band_rankings(clean_raw):
    acts = band_activity(clean_raw)
    assert set(acts) == {"delta", "theta", "alpha", "beta", "gamma"}
    assert acts["alpha"].best in ("O1", "O2", "P3", "P4", "Pz")
    assert acts["delta"].best in ("Fp1", "Fp2")  # blinks live in the delta band
    assert rank_channels(clean_raw, "band:8-12").top(1)[0] in ("O1", "O2", "P3", "P4", "Pz")
    dom = dominant_bands(clean_raw)
    assert {d["channel"] for d in dom} == set(clean_raw.pick_names("eeg"))


def test_quality_and_snr_rankings(bad_raw):
    q = rank_channels(bad_raw, "quality")
    assert q.bottom(1)[0] == "O2"
    s = rank_channels(bad_raw, "snr")
    assert s.bottom(2)[0] in ("O2", "T8")
    c = rank_channels(bad_raw.pick(bad_raw.pick_names("eeg")[:8]), "connectivity")
    assert isinstance(c, ChannelRanking) and len(c.ch_names) == 8
    with pytest.raises(ValueError):
        rank_channels(bad_raw, "nonsense")


def test_motor_imagery_rankings(mi_raw):
    ep = Epochs(mi_raw, tmin=-1.0, tmax=4.0, baseline=None)
    assert set(ep.event_id) == {"left", "right"}
    dec = decoding_accuracy(ep, window=(0.5, 3.5), n_repeats=2)
    assert set(dec.top(2)) == {"C3", "C4"}
    assert dec.value("C3") > 0.8 and dec.details["chance_level"] == 0.5
    assert dec.details["all_channels_accuracy"] > 0.85
    disc = discriminability(ep, window=(0.5, 3.5))
    assert set(disc.top(2)) == {"C3", "C4"}
    left = erd_ers(ep, baseline=(-1, 0), window=(0.5, 3.5), condition="left")
    right = erd_ers(ep, baseline=(-1, 0), window=(0.5, 3.5), condition="right")
    assert left.best == "C4" and right.best == "C3"
    pct = dict(zip(left.ch_names, left.details["erd_ers_pct"]))
    assert pct["C4"] < -40  # desynchronisation
    assert rank_channels(ep, "erd", baseline=(-1, 0), window=(0.5, 3.5)).top(2)


def test_oddball_erp_ranking(oddball_raw):
    raw = oddball_raw.filter(0.5, 30)
    ep = Epochs(raw, tmin=-0.2, tmax=0.8, reject=150)
    snr = erp_snr(ep, condition="target")
    assert snr.best in ("Pz", "CP1", "CP2", "P3", "P4")
    lat = dict(zip(snr.ch_names, snr.details["peak_latency_s"]))
    assert 0.25 < lat["Pz"] < 0.45
    dec = rank_channels(ep, "decoding", feature="erp", window=(0.1, 0.6), n_repeats=1)
    assert dec.best in ("Pz", "CP1", "CP2", "P3", "P4")


def test_lda_and_statistics():
    rng = np.random.default_rng(0)
    x0 = rng.normal(0, 1, (60, 3))
    x1 = rng.normal(0, 1, (60, 3)) + [3, 0, 0]
    x = np.vstack([x0, x1])
    y = np.repeat([0, 1], 60)
    assert (ShrinkageLDA().fit(x, y).predict(x) == y).mean() > 0.9
    assert cross_val_accuracy(x, y) > 0.9
    assert 0.3 < cross_val_accuracy(x, rng.permutation(y)) < 0.7
    fs = fisher_score(x, y)
    assert np.argmax(fs) == 0
    cov = ledoit_wolf(x - x.mean(axis=0))
    assert np.all(np.linalg.eigvalsh(cov) > 0)


def test_ranking_object():
    r = ChannelRanking(["a", "b", "c"], np.array([1.0, 3.0, np.nan]), "m", details={"x": [1, 2, 3]})
    assert r.top(2) == ["b", "a"] and r.bottom(1) == ["c"]
    low = ChannelRanking(["a", "b"], np.array([1.0, 3.0]), "m", higher_is_better=False)
    assert low.best == "a"
    assert "rank" in r.summary() and r.ranked()[0]["x"] == 2
