"""Conversational EEG analysis assistant.

``AnalysisSession`` (the tools) works with the core install; ``EEGAssistant`` and
``chat`` talk to Claude and need ``pip install "eegproc[assistant]"``.
"""

from .session import AnalysisSession, ToolOutput

__all__ = ["AnalysisSession", "ToolOutput", "EEGAssistant", "chat"]


def __getattr__(name):
    if name in ("EEGAssistant", "chat"):
        from . import agent

        return getattr(agent, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
