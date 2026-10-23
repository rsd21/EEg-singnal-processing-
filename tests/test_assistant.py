"""The assistant's tools, and its conversation loop against a scripted fake API (no network)."""

import json

import matplotlib
import pytest

matplotlib.use("Agg")

from eegproc.assistant import AnalysisSession  # noqa: E402
from eegproc.assistant.session import validate_input  # noqa: E402


@pytest.fixture
def session(tmp_path):
    s = AnalysisSession(tmp_path / "out", confirm=lambda q: True)
    assert not s.call("load_recording", {"path": "demo:blinks"}).is_error
    return s


def test_tool_definitions_are_valid_and_stable():
    defs = AnalysisSession.tool_definitions()
    names = [d["name"] for d in defs]
    assert len(names) == len(set(names)) and "run_python" in names
    assert "run_python" not in [d["name"] for d in AnalysisSession.tool_definitions(allow_python=False)]
    for d in defs:
        schema = d["input_schema"]
        assert schema["type"] == "object" and schema["additionalProperties"] is False
        assert set(schema["required"]) <= set(schema["properties"]) and d["description"]
    assert json.dumps(defs) == json.dumps(AnalysisSession.tool_definitions())


def test_validate_input():
    schema = {"properties": {"a": {"type": "number"}, "b": {"type": "array", "items": {"type": "string"}},
                             "c": {"type": "string", "enum": ["x", "y"]}}, "required": ["a"]}
    assert validate_input(schema, {"a": 1.5, "b": ["F3"], "c": "x"}) == []
    problems = validate_input(schema, {"b": [1], "c": "z", "d": 0})
    assert len(problems) == 4
    assert validate_input(schema, {"a": True}) and validate_input(schema, "nope")


def test_overview_hides_patient_fields(tmp_path):
    from eegproc.datasets import simulate_eeg

    raw = simulate_eeg(10, montage="standard_1020", seed=0)
    raw.meta["subject"] = {"name": "Jane_Doe", "id": "MRN123"}
    path = raw.save(tmp_path / "rec.edf")
    s = AnalysisSession(tmp_path / "out")
    text = s.call("load_recording", {"path": path}).text
    assert "Jane" not in text and "MRN123" not in text and "not shown" in text


def test_processing_undo_and_errors(session):
    out = session.call("filter_data", {"l_freq": 1.0, "h_freq": 40.0})
    assert not out.is_error and session.steps == ["band-pass 1-40 Hz"]
    before = session.raw
    session.call("set_reference", {"kind": "average"})
    assert session.raw is not before
    assert "Undid" in session.call("undo_last_step", {}).text and session.raw is before
    assert session.call("filter_data", {}).is_error
    assert session.call("filter_data", {"l_freq": "1"}).is_error
    assert session.call("mark_bad_channels", {"channels": ["Nope"]}).is_error
    assert session.call("compute_erp", {}).is_error  # no epochs yet
    assert session.call("no_such_tool", {}).is_error


def test_blinks_quality_and_ica(session):
    blinks = session.call("detect_blinks", {"add_annotations": True})
    assert "blinks/min" in blinks.text and blinks.figures
    assert session.raw.annotations.count().get("blink", 0) > 10
    epochs = session.call("make_epochs", {"event_names": ["blink"], "tmin": -0.5, "tmax": 0.5})
    assert "blink" in epochs.text and session.epochs.n_epochs > 10
    quality = session.call("assess_quality", {})
    assert "Overall score" in quality.text and quality.figures[0].exists()
    ica = session.call("run_ica", {})
    assert "Suggested for removal -> eye" in ica.text
    assert not session.call("remove_ica_components", {"components": [0]}).is_error
    assert session.ica is None and session.epochs is None  # cleared by the data change


def test_outputs_stay_in_output_folder(session, tmp_path):
    assert not session.call("save_data", {"filename": "../../escape.edf"}).is_error
    assert (tmp_path / "out" / "escape.edf").exists() and not (tmp_path / "escape.edf").exists()
    assert session.call("save_data", {"filename": "data.exe"}).is_error


def test_run_python_needs_approval(tmp_path):
    asked = []
    s = AnalysisSession(tmp_path, confirm=lambda q: asked.append(q) or False)
    s.call("load_recording", {"path": "demo:blinks"})
    out = s.call("run_python", {"code": "print('hi')", "purpose": "say hi"})
    assert "declined" in out.text and "print('hi')" in asked[0]
    s.confirm = lambda q: True
    out = s.call("run_python", {"code": "print(raw.n_channels)\nnew_raw = raw.crop(0, 30)", "purpose": "crop"})
    assert out.text.startswith("34") and s.raw.duration == pytest.approx(30, abs=0.01)
    assert s.call("run_python", {"code": "1/0", "purpose": "fail"}).is_error
