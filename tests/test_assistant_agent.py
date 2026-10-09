"""The assistant's conversation loop, driven through the real Anthropic SDK against a scripted
fake API (no network, no API key)."""

import json

import pytest

anthropic = pytest.importorskip("anthropic")
httpx2 = pytest.importorskip("httpx2")


def _sse(blocks, stop_reason):
    """Server-sent events for one streamed message made of the given content blocks."""
    events = [("message_start", {"type": "message_start", "message": {
        "id": "msg_1", "type": "message", "role": "assistant", "model": "claude-opus-5-5", "content": [],
        "stop_reason": None, "stop_sequence": None,
        "usage": {"input_tokens": 100, "output_tokens": 1, "cache_read_input_tokens": 50,
                  "cache_creation_input_tokens": 10}}})]
    for i, block in enumerate(blocks):
        if block["type"] == "thinking":
            start, deltas = {"type": "thinking", "thinking": "", "signature": ""}, [
                {"type": "thinking_delta", "thinking": block["thinking"]},
                {"type": "signature_delta", "signature": "sig"}]
        elif block["type"] == "text":
            start, deltas = {"type": "text", "text": ""}, [{"type": "text_delta", "text": block["text"]}]
        else:
            start = {"type": "tool_use", "id": block["id"], "name": block["name"], "input": {}}
            deltas = [{"type": "input_json_delta", "partial_json": json.dumps(block["input"])}]
        events.append(("content_block_start", {"type": "content_block_start", "index": i, "content_block": start}))
        events += [("content_block_delta", {"type": "content_block_delta", "index": i, "delta": d}) for d in deltas]
        events.append(("content_block_stop", {"type": "content_block_stop", "index": i}))
    events.append(("message_delta", {"type": "message_delta", "delta": {"stop_reason": stop_reason,
                                                                         "stop_sequence": None},
                                     "usage": {"output_tokens": 20}}))
    events.append(("message_stop", {"type": "message_stop"}))
    return "".join(f"event: {name}\ndata: {json.dumps(data)}\n\n" for name, data in events)


class ScriptedAPI:
    def __init__(self, replies):
        self.replies = list(replies)
        self.requests = []

    def __call__(self, request):
        self.requests.append({"headers": dict(request.headers), "body": json.loads(request.content)})
        return httpx2.Response(200, headers={"content-type": "text/event-stream"},
                               content=_sse(*self.replies.pop(0)).encode())


class QuietUI:
    notebook = False

    def __init__(self):
        self.printed, self.figures, self.notices = [], [], []

    def begin(self, kind):
        pass

    def text(self, chunk):
        self.printed.append(chunk)

    progress = text

    def tool(self, name, args):
        pass

    def tool_result(self, out):
        self.figures += out.figures

    def notice(self, message):
        self.notices.append(message)

    def confirm(self, question):
        return True


def _assistant(api, tmp_path, **kwargs):
    from eegproc.assistant import EEGAssistant

    client = anthropic.Anthropic(api_key="test", max_retries=0,
                                 http_client=anthropic.DefaultHttpxClient(transport=httpx2.MockTransport(api)))
    return EEGAssistant("demo:blinks", client=client, ui=QuietUI(), output_dir=tmp_path / "out", **kwargs)


def test_agent_runs_tools_and_sends_results(tmp_path):
    api = ScriptedAPI([
        ([{"type": "thinking", "thinking": "Checking blinks first."},
          {"type": "text", "text": "Let me look at the blinks."},
          {"type": "tool_use", "id": "toolu_1", "name": "detect_blinks", "input": {}},
          {"type": "tool_use", "id": "toolu_2", "name": "filter_data", "input": {"l_freq": "bad"}}], "tool_use"),
        ([{"type": "text", "text": "You blink about 14 times per minute. What is your study about?"}], "end_turn"),
    ])
    a = _assistant(api, tmp_path)
    a.send(a._opening_message())

    first, second = api.requests
    body, headers = first["body"], first["headers"]
    assert body["model"] == "claude-opus-5-5" and body["fallbacks"] == "default"
    assert body["thinking"] == {"type": "adaptive", "display": "updates",
                                "block_binding": {"prefix_mismatch_behavior": "drop_block"}}
    assert body["output_config"] == {"effort": "medium"} and body["cache_control"] == {"type": "ephemeral"}
    assert body["system"][-1]["cache_control"] == {"type": "ephemeral"} and body["stream"] is True
    assert all(t["eager_input_streaming"] for t in body["tools"])
    for beta in ("thinking-display-updates-2026-08-18", "server-side-fallback-2026-07-01"):
        assert beta in headers["anthropic-beta"]
    assert "summary of it" in body["messages"][0]["content"] and "VEOG" in body["messages"][0]["content"]

    # the assistant turn is replayed unchanged (thinking with its signature), then the tool results
    msgs = second["body"]["messages"]
    assert [m["role"] for m in msgs] == ["user", "assistant", "user"]
    assert msgs[1]["content"][0] == {"type": "thinking", "thinking": "Checking blinks first.", "signature": "sig"}
    results = msgs[2]["content"]
    assert [r["tool_use_id"] for r in results] == ["toolu_1", "toolu_2"]
    assert "blinks/min" in results[0]["content"][0]["text"] and results[0]["content"][1]["type"] == "image"
    assert results[1]["is_error"] is True and "must be number" in results[1]["content"][0]["text"]
    assert len(a.messages) == 4 and a.messages[-1]["role"] == "assistant"
    assert "Checking blinks first." in "".join(a.ui.printed) and a.ui.figures
    assert a.usage["input"] == 200 and a.usage["cache_read"] == 100


def test_agent_refusal_is_not_added_to_history(tmp_path):
    api = ScriptedAPI([([], "refusal")])
    a = _assistant(api, tmp_path)
    a.send("hello")
    assert a.messages == [{"role": "user", "content": "hello"}] and "declined" in a.ui.notices[0]


def test_agent_reports_api_errors(tmp_path):
    def handler(request):
        return httpx2.Response(401, json={"type": "error", "error": {"type": "authentication_error",
                                                                     "message": "bad key"}})
    a = _assistant(handler, tmp_path)
    a.send("hello")
    assert "API key" in a.ui.notices[0]


def test_echo_blocks_drops_declined_attempt():
    from types import SimpleNamespace as B

    from eegproc.assistant.agent import echo_blocks

    content = [B(type="thinking"), B(type="text"), B(type="tool_use"), B(type="fallback"),
               B(type="thinking"), B(type="tool_use")]
    assert [b.type for b in echo_blocks(content)] == ["text", "thinking", "tool_use"]
    assert echo_blocks(content[:2]) == content[:2]


def test_interrupted_tool_calls_are_answered(tmp_path):
    api = ScriptedAPI([])
    a = _assistant(api, tmp_path)
    from types import SimpleNamespace as B

    a.messages = [{"role": "user", "content": "hi"},
                  {"role": "assistant", "content": [B(type="tool_use", id="toolu_9")]}]
    a._close_open_tool_calls()
    assert a.messages[-1]["content"][0]["tool_use_id"] == "toolu_9"
    assert len(a.messages) == 3


def test_figures_stop_being_sent_past_the_budget(tmp_path, monkeypatch):
    import eegproc.assistant.agent as agent

    monkeypatch.setattr(agent, "IMAGE_BUDGET_BYTES", 1000)
    api = ScriptedAPI([
        ([{"type": "tool_use", "id": "toolu_1", "name": "detect_blinks", "input": {}}], "tool_use"),
        ([{"type": "text", "text": "Done."}], "end_turn"),
    ])
    a = _assistant(api, tmp_path)
    a.session.call("load_recording", {"path": "demo:blinks"})
    a.send("detect blinks")
    result = api.requests[1]["body"]["messages"][-1]["content"][0]
    assert [c["type"] for c in result["content"]] == ["text"]
    assert "image budget" in result["content"][0]["text"] and "size limit" in a.ui.notices[0]
    assert a.ui.figures  # the user still sees the figure


def test_missing_api_key_gives_instructions(tmp_path, monkeypatch):
    from eegproc.assistant import EEGAssistant

    for var in ("ANTHROPIC_API_KEY", "ANTHROPIC_AUTH_TOKEN", "ANTHROPIC_PROFILE", "ANTHROPIC_CONFIG_DIR",
                "ANTHROPIC_FEDERATION_RULE_ID"):
        monkeypatch.delenv(var, raising=False)
    monkeypatch.setenv("HOME", str(tmp_path))  # no saved `ant auth login` profile
    monkeypatch.setenv("APPDATA", str(tmp_path))
    with pytest.raises(RuntimeError, match="ANTHROPIC_API_KEY"):
        EEGAssistant(ui=QuietUI(), output_dir=tmp_path / "out")
