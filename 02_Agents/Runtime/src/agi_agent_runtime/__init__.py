"""Public API for the shared, provider-neutral agent runtime."""

from .drafts import AnswerDraft, CritiqueDraft
from .budgets import BudgetExceeded, BudgetGovernor, BudgetLedger
from .fast_path import FastPathResult, FastPathRunner
from .ports import ModelAdapter, ToolExecutor
from .precheck import HostPrecheckError, HostPrecheckResult, run_host_precheck
from .routing import BudgetPolicy, ReasoningTier, RoutingPolicy, get_routing_policy
from .run_store import AtomicJsonRunStore
from .runtime import AgentRuntime, RuntimeExecutionError

__version__ = "0.1.0"

__all__ = [
    "AgentRuntime",
    "AnswerDraft",
    "AtomicJsonRunStore",
    "BudgetExceeded",
    "BudgetGovernor",
    "BudgetLedger",
    "BudgetPolicy",
    "CritiqueDraft",
    "FastPathResult",
    "FastPathRunner",
    "HostPrecheckError",
    "HostPrecheckResult",
    "ModelAdapter",
    "ReasoningTier",
    "RuntimeExecutionError",
    "RoutingPolicy",
    "ToolExecutor",
    "get_routing_policy",
    "run_host_precheck",
]
