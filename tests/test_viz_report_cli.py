import json

import numpy as np
import pytest

from eegproc import Epochs, ICA, Pipeline, RawEEG, generate_report, rank_channels, read_raw, viz
from eegproc.analysis import compute_connectivity
from eegproc.cli import main
from eegproc.quality import erd_ers


# -------------------------------------------------------------------- plots
def test_plots_smoke(bad_raw, tmp_path):
    rep = bad_raw.assess_quality()
    figs = [
        viz.plot_raw(bad_raw, duration=5, quality=rep, display_filter=(0.5, None)),
        viz.plot_psd(bad_raw.copy().set_bads(["O2"])),
        viz.plot_band_power(bad_raw),
        viz.plot_band_topomaps(bad_raw),
        viz.plot_spectrogram(bad_raw, "O1"),
        viz.plot_sensors(bad_raw.copy().set_bads(["T8"])),
        viz.plot_quality_report(rep),
        viz.plot_quality_bars(rep),
        viz.plot_quality_heatmap(rep),
        viz.plot_quality_topomap(rep, metric="snr_db"),
        viz.plot_ranking(rank_channels(bad_raw, "alpha")),
        viz.plot_connectivity(compute_connectivity(bad_raw, "coherence", (8, 12))),
    ]
    for k, fig in enumerate(figs):
        fig.savefig(tmp_path / f"fig{k}.png")
    assert len(list(tmp_path.glob("*.png"))) == len(figs)


def test_topomap_inputs():
    names = ["Fp1", "Fp2", "C3", "C4", "O1", "O2", "Cz"]
    fig = viz.plot_topomap(np.arange(7.0) - 3, names, show_names=True)
    assert fig is not None
    with pytest.raises(ValueError):
        viz.plot_topomap([1, 2], ["X", "Y"])


def test_erp_and_ica_plots(oddball_raw, mi_raw):
    ep = Epochs(oddball_raw.filter(0.5, 30), tmin=-0.2, tmax=0.8)
    evk = ep["target"].average()
    viz.plot_evoked(evk, picks=["Pz"])
    viz.plot_evoked_topomaps(evk)
    viz.plot_evoked_topomaps(evk, times=[0.1, 0.35])
    ep_mi = Epochs(mi_raw, tmin=-1, tmax=4, baseline=None)
    viz.plot_erds_timecourse(ep_mi, ["C3", "C4"])
    viz.plot_ranking(erd_ers(ep_mi, baseline=(-1, 0), window=(0.5, 3.5), condition="left"))
    ica = ICA(n_components=8, random_state=0).fit(oddball_raw.crop(0, 60))
    ica.plot_components()


# ------------------------------------------------------------------- report
def test_report_oddball(bad_raw, tmp_path):
    raw = bad_raw.copy()
    raw.annotations.append(np.arange(5, 80, 1.5), 0.0, "tone")
    path = generate_report(raw, tmp_path / "r.html")
    html = open(path, encoding="utf-8").read()
    for section in ("Overview", "Channel quality", "Spectral activity", "Events and responses", "Methods"):
        assert section in html
    assert "O2" in html and "data:image/png;base64" in html
    assert "notch filter" in html


def test_report_motor_imagery_and_no_positions(mi_raw, tmp_path):
    path = generate_report(mi_raw, tmp_path / "mi.html", epoch_window=(-1.0, 4.0), connectivity=False)
    html = open(path, encoding="utf-8").read()
    assert "Mu/alpha" in html and "best separate" in html
    # channels without known positions: no scalp maps, but the rest of the report still works
    anon = RawEEG(mi_raw.data[:8, : 30 * 256], 256, [f"E{i}" for i in range(8)], ["eeg"] * 8)
    html = open(generate_report(anon, tmp_path / "anon.html"), encoding="utf-8").read()
    assert "Channel quality" in html and "band topomaps" not in html


# ----------------------------------------------------------------- pipeline
def test_pipeline_from_config(bad_raw, tmp_path):
    cfg = {"steps": [{"filter": {"l_freq": 1.0, "h_freq": 40.0}}, {"notch": {"freqs": "auto"}},
                     {"detect_bad_channels": {}}, {"interpolate_bad_channels": {}},
                     {"rereference": {"ref": "average"}}, {"resample": {"sfreq": 128}}],
           "output": str(tmp_path / "clean.npz")}
    pipe = Pipeline.from_config(cfg)
    out = pipe.run(bad_raw)
    assert out.sfreq == 128 and out.bads == []
    assert [e["step"] for e in pipe.log] == ["filter", "notch", "detect_bad_channels", "interpolate_bad_channels",
                                             "rereference", "resample"]
    assert (tmp_path / "clean.npz").exists()
    (tmp_path / "p.json").write_text(json.dumps(cfg))
    assert len(Pipeline.from_config(tmp_path / "p.json").steps) == 6
    yaml = pytest.importorskip("yaml")
    (tmp_path / "p.yaml").write_text(yaml.safe_dump(cfg))
    assert Pipeline.from_config(tmp_path / "p.yaml").to_config()["steps"][0] == {"filter": {"l_freq": 1.0, "h_freq": 40.0}}
    with pytest.raises(ValueError):
        Pipeline([{"unknown_step": {}}])


def test_pipeline_ica_step(clean_raw):
    out = Pipeline().add("filter", l_freq=1.0, h_freq=40.0).add("ica", n_components=0.99).run(clean_raw)
    assert "ica_excluded" in out.meta and out.meta["ica_excluded"]


# ---------------------------------------------------------------------- CLI
def test_cli_end_to_end(tmp_path, capsys):
    edf = str(tmp_path / "sim.edf")
    assert main(["simulate", edf, "--duration", "60", "--task", "oddball", "--bad-channels"]) == 0
    assert main(["info", edf, "--brief"]) == 0
    assert "Sampling" in capsys.readouterr().out
    assert main(["info", edf, "--json"]) == 0
    assert json.loads(capsys.readouterr().out)["n_channels"] == 21
    assert main(["quality", edf, "--csv", str(tmp_path / "q.csv"), "--json", str(tmp_path / "q.json"),
                 "--plot", str(tmp_path / "q.png")]) == 0
    out = capsys.readouterr().out
    assert "Suggested bad" in out and (tmp_path / "q.png").exists()
    assert main(["rank", edf, "--by", "alpha", "--top", "3"]) == 0
    assert main(["rank", edf, "--by", "erp", "--reject", "200", "--plot", str(tmp_path / "erp.png")]) == 0
    assert main(["convert", edf, str(tmp_path / "sim.vhdr"), "--picks", "eeg"]) == 0
    assert read_raw(tmp_path / "sim.vhdr").n_channels == 19
    cfg = tmp_path / "pipe.json"
    cfg.write_text(json.dumps({"steps": [{"filter": {"l_freq": 1, "h_freq": 40}}, {"detect_bad_channels": {}}]}))
    assert main(["process", edf, "--config", str(cfg), "-o", str(tmp_path / "clean.edf")]) == 0
    assert read_raw(tmp_path / "clean.edf").meta["format"] == "edf"
    assert main(["plot", edf, "--kind", "psd", "-o", str(tmp_path / "psd.png")]) == 0
    assert main(["plot", edf, "--kind", "quality", "-o", str(tmp_path / "qq.png")]) == 0
    assert main(["features", edf, "-o", str(tmp_path / "f.csv")]) == 0
    assert main(["report", edf, "-o", str(tmp_path / "rep.html"), "--no-connectivity"]) == 0
    assert (tmp_path / "rep.html").stat().st_size > 10000
    assert main(["formats"]) == 0
    assert main(["info", str(tmp_path / "missing.edf")]) == 1
