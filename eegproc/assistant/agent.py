"""The conversational loop: Claude plans and explains, the session's tools do the processing."""

from __future__ import annotations

import base64
import json
import sys
from pathlib import Path

try:
    import anthropic
except ImportError as err:  # pragma: no cover - exercised only without the extra installed
    raise ImportError('The EEG assistant needs the Anthropic SDK: pip install "eegproc[assistant]"') from err

from .prompt import PYTHON_NOTE, SYSTEM_PROMPT
from .session import AnalysisSession, ToolOutput

DEFAULT_MODEL = "claude-opus-5-5"
BETAS = ["thinking-display-updates-2026-08-18", "thinking-binding-controls-2026-08-01",
         "server-side-fallback-2026-07-01"]
EXIT_WORDS = {"quit", "exit", "q", "bye"}
# Every request resends the whole conversation, figures included; the API caps a request at 32 MB.
IMAGE_BUDGET_BYTES = 20_000_000


def _in_notebook() -> bool:
    if "google.colab" in sys.modules:
        return True
    try:
        from IPython import get_ipython
    except ImportError:
        return False
    shell = get_ipython()
    return shell is not None and "IPKernelApp" in shell.config


class ConsoleUI:
    """Prints the conversation; works in a terminal, Jupyter and Google Colab."""

    DIM, RESET = "\033[2m", "\033[0m"

    def __init__(self, notebook: bool | None = None):
        self.notebook = _in_notebook() if notebook is None else notebook
        self._pending: str | None = None

    def begin(self, kind: str) -> None:
        self._pending = kind

    def _start(self, kind: str) -> None:
        if self._pending == kind:
            print("\n\n" if kind == "text" else f"\n{self.DIM}  ", end="")
            self._pending = None

    def text(self, chunk: str) -> None:
        self._start("text")
        print(chunk, end="", flush=True)

    def progress(self, chunk: str) -> None:
        self._start("thinking")
        print(f"{self.DIM}{chunk}{self.RESET}", end="", flush=True)

    def tool(self, name: str, args: dict) -> None:
        shown = ", ".join(f"{k}={json.dumps(v)}" for k, v in args.items() if k != "code")
        print(f"\n{self.DIM}  -> {name}({shown[:120]}){self.RESET}", flush=True)

    def tool_result(self, out: ToolOutput) -> None:
        first = out.text.strip().splitlines()[0] if out.text.strip() else ""
        print(f"{self.DIM}     {'error: ' if out.is_error else ''}{first[:110]}{self.RESET}", flush=True)
        for path in out.figures:
            self.figure(path)

    def figure(self, path: Path) -> None:
        if self.notebook:
            from IPython.display import Image, display

            display(Image(filename=str(path)))
        else:
            print(f"{self.DIM}     figure: {path}{self.RESET}")

    def notice(self, message: str) -> None:
        print(f"\n[{message}]", flush=True)

    def ask(self) -> str:
        return input("\n\nyou> ")

    def confirm(self, question: str) -> bool:
        print(f"\n{question}")
        return input("[y/N] ").strip().lower() in ("y", "yes")


def echo_blocks(content: list) -> list:
    """Blocks to keep in history; after a mid-output fallback, drop the declined attempt's blocks."""
    cut = max((i for i, b in enumerate(content) if b.type == "fallback"), default=None)
    if cut is None:
        return list(content)
    earlier = [b for b in content[:cut]
               if b.type not in ("thinking", "redacted_thinking", "tool_use", "server_tool_use")]
    return earlier + list(content[cut + 1:])


class EEGAssistant:
    """Chat with Claude about an EEG recording; it runs the analysis through eegproc tools.

    Parameters
    ----------
    recording : path to an EEG file (or ``'demo:blinks'`` etc.), loaded before the chat starts.
    model, effort : Claude model and effort level (``low``/``medium``/``high``/``xhigh``/``max``).
    output_dir : where figures, saved data, reports and feature tables go.
    allow_python : offer the run_python tool (each run still needs the user's approval).
    send_figures : also show the figures to Claude so it can read them.
    client : an ``anthropic.Anthropic`` client; by default one is created from ``ANTHROPIC_API_KEY``.
    """

    def __init__(self, recording: str | Path | None = None, *, model: str = DEFAULT_MODEL,
                 effort: str = "medium", output_dir: str | Path = "eeg_assistant_output",
                 allow_python: bool = True, send_figures: bool = True, client=None, ui=None,
                 max_tool_rounds: int = 40):
        if client is None:
            try:
                client = anthropic.Anthropic()
            except anthropic.CredentialsError as err:
                raise RuntimeError(f"Could not load Anthropic credentials: {err}") from err
            if not (client.api_key or client.auth_token or client.credentials):
                raise RuntimeError("No Anthropic API key found. Get one at console.anthropic.com and set it: "
                                   "export ANTHROPIC_API_KEY=sk-ant-... (Windows PowerShell: "
                                   "$env:ANTHROPIC_API_KEY=\"sk-ant-...\"; Colab: the Secrets panel).")
        self.client = client
        self.ui = ui or ConsoleUI()
        self.model = model
        self.effort = effort
        self.send_figures = send_figures
        self.max_tool_rounds = max_tool_rounds
        self.session = AnalysisSession(output_dir, confirm=self.ui.confirm)
        self.tools = [{**t, "eager_input_streaming": True}
                      for t in self.session.tool_definitions(allow_python)]
        prompt = SYSTEM_PROMPT + (PYTHON_NOTE if allow_python else "")
        self.system = [{"type": "text", "text": prompt, "cache_control": {"type": "ephemeral"}}]
        self.messages: list[dict] = []
        self.usage = {"input": 0, "output": 0, "cache_read": 0, "cache_write": 0}
        self._image_bytes = 0
        self.recording = recording

    # ------------------------------------------------------------ public API
    def chat(self) -> None:
        """Interactive loop until the user types quit."""
        self.ui.notice(f"EEG assistant ({self.model}). Type 'quit' to stop. "
                       f"Files go to {self.session.output_dir.resolve()}")
        self.send(self._opening_message())
        while True:
            try:
                text = self.ui.ask().strip()
            except (EOFError, KeyboardInterrupt):
                break
            if text.lower() in EXIT_WORDS:
                break
            if text:
                self.send(text)
        u = self.usage
        self.ui.notice(f"Session ended. Tokens: {u['input']} input, {u['cache_read']} cache read, "
                       f"{u['cache_write']} cache write, {u['output']} output.")

    def send(self, text: str) -> None:
        """Send one user message and run Claude's tool calls until it replies."""
        self.messages.append({"role": "user", "content": text})
        try:
            self._run_turns()
        except anthropic.AuthenticationError:
            self.ui.notice("The API key was rejected. Set ANTHROPIC_API_KEY to a valid key.")
        except anthropic.RateLimitError:
            self.ui.notice("Rate limited by the API. Wait a minute, then send your message again.")
        except anthropic.APIStatusError as err:
            self.ui.notice(f"API error {err.status_code}: {err.message}")
        except anthropic.APIConnectionError:
            self.ui.notice("Could not reach the API. Check the internet connection and try again.")
        except KeyboardInterrupt:
            self._close_open_tool_calls()
            self.ui.notice("Interrupted.")

    # ------------------------------------------------------------ internals
    def _opening_message(self) -> str:
        if self.recording is None:
            return "Hello. I'd like help analysing EEG data. I haven't loaded a recording yet."
        out = self.session.call("load_recording", {"path": str(self.recording)})
        if out.is_error:
            self.ui.notice(out.text)
            return f"Hello. I tried to load {self.recording} but it failed: {out.text}"
        return ("Hello. I've loaded my recording; this is the toolkit's summary of it:\n\n"
                f"{out.text}\n\nPlease help me analyse it.")

    def _close_open_tool_calls(self) -> None:
        """Answer tool calls that never ran, so the conversation stays valid."""
        last = self.messages[-1]
        if last["role"] != "assistant":
            return
        ids = [b.id for b in last["content"] if b.type == "tool_use"]
        if ids:
            self.messages.append({"role": "user", "content": [
                {"type": "tool_result", "tool_use_id": i, "is_error": True,
                 "content": "Interrupted by the user before this ran."} for i in ids]})

    def _request(self) -> dict:
        return dict(
            model=self.model, max_tokens=64000, system=self.system, tools=self.tools,
            messages=self.messages, betas=BETAS, fallbacks="default",
            thinking={"type": "adaptive", "display": "updates",
                      "block_binding": {"prefix_mismatch_behavior": "drop_block"}},
            output_config={"effort": self.effort}, cache_control={"type": "ephemeral"},
        )

    def _stream_once(self):
        for attempt in range(3):
            try:
                with self.client.beta.messages.stream(**self._request()) as stream:
                    for event in stream:
                        if event.type == "content_block_start" and event.content_block.type in ("text", "thinking"):
                            self.ui.begin(event.content_block.type)
                        elif event.type == "text":
                            self.ui.text(event.text)
                        elif event.type == "thinking":
                            self.ui.progress(event.thinking)
                    return stream.get_final_message()
            except ValueError:
                # Tool-call JSON the SDK could not parse at all; no tool_use id to answer, so ask again.
                if attempt == 2:
                    raise
                self.ui.notice("The reply was garbled; asking again.")

    def _count(self, usage) -> None:
        self.usage["input"] += usage.input_tokens or 0
        self.usage["output"] += usage.output_tokens or 0
        self.usage["cache_read"] += usage.cache_read_input_tokens or 0
        self.usage["cache_write"] += usage.cache_creation_input_tokens or 0

    def _run_turns(self) -> None:
        for _ in range(self.max_tool_rounds):
            response = self._stream_once()
            self._count(response.usage)
            if response.stop_reason == "refusal":
                category = getattr(response.stop_details, "category", None)
                self.ui.notice(f"Claude declined this request ({category or 'no category'}). "
                               "Try rephrasing it.")
                return
            blocks = echo_blocks(response.content)
            self.messages.append({"role": "assistant", "content": blocks})
            calls = [b for b in blocks if b.type == "tool_use"]
            if not calls:
                if response.stop_reason == "max_tokens":
                    self.ui.notice("The reply hit the length limit and was cut off.")
                return
            self.messages.append({"role": "user", "content": [self._run_tool(b, response.stop_reason)
                                                              for b in calls]})
        self.ui.notice(f"Stopped after {self.max_tool_rounds} tool rounds; send a message to continue.")

    def _run_tool(self, block, stop_reason: str) -> dict:
        if stop_reason == "max_tokens":
            out = ToolOutput("Not run: the tool input was cut off by the length limit. Try again "
                             "with a shorter input.", is_error=True)
        else:
            self.ui.tool(block.name, block.input if isinstance(block.input, dict) else {})
            out = self.session.call(block.name, block.input)
            self.ui.tool_result(out)
        text = out.text
        if out.figures:
            text += "\n[Shown to the user: " + ", ".join(p.name for p in out.figures) + "]"
        images = []
        for path in out.figures if self.send_figures else []:
            data = base64.standard_b64encode(Path(path).read_bytes()).decode()
            if self._image_bytes + len(data) > IMAGE_BUDGET_BYTES:
                text += f"\n[{path.name} was not sent to you: the session's image budget is used up]"
                if self._image_bytes <= IMAGE_BUDGET_BYTES:
                    self.ui.notice("Figures are no longer sent to Claude (size limit); it still sees "
                                   "the numbers, and you still see the figures.")
                    self._image_bytes = IMAGE_BUDGET_BYTES + 1
                continue
            self._image_bytes += len(data)
            images.append({"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": data}})
        content: list[dict] = [{"type": "text", "text": text}, *images]
        result = {"type": "tool_result", "tool_use_id": block.id, "content": content}
        if out.is_error:
            result["is_error"] = True
        return result


def chat(recording: str | Path | None = None, **kwargs) -> EEGAssistant:
    """Start an interactive session: ``eegproc.assistant.chat("recording.edf")``."""
    assistant = EEGAssistant(recording, **kwargs)
    assistant.chat()
    return assistant
