"""Public API for the shared, provider-neutral agent runtime."""

from .drafts import AnswerDraft, CritiqueDraft
from .ports import ModelAdapter, ToolExecutor
from .run_store import AtomicJsonRunStore
from .runtime import AgentRuntime, RuntimeExecutionError

__version__ = "0.1.0"

__all__ = [
    "AgentRuntime",
    "AnswerDraft",
    "AtomicJsonRunStore",
    "CritiqueDraft",
    "ModelAdapter",
    "RuntimeExecutionError",
    "ToolExecutor",
]
