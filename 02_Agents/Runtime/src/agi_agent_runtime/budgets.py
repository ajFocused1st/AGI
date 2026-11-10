"""Runtime budget accounting for routed model and tool attempts."""

from __future__ import annotations

from dataclasses import dataclass, field
from time import perf_counter
from typing import Any

from agi_contracts import ModelUsage

from .routing import BudgetPolicy, RoutingPolicy


class BudgetExceeded(RuntimeError):
    """Raised when a routed run would exceed a host-owned budget."""


@dataclass
class BudgetEvent:
    kind: str
    phase: str
    accepted: bool
    reason: str | None = None
    input_tokens: int | None = None
    output_tokens: int | None = None
    total_tokens: int | None = None

    def as_dict(self) -> dict[str, Any]:
        return {
            "kind": self.kind,
            "phase": self.phase,
            "accepted": self.accepted,
            "reason": self.reason,
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "total_tokens": self.total_tokens,
        }


@dataclass
class BudgetLedger:
    policy: RoutingPolicy
    model_attempts: int = 0
    tool_attempts: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    total_tokens: int = 0
    events: list[BudgetEvent] = field(default_factory=list)

    def as_dict(self) -> dict[str, Any]:
        return {
            "tier": self.policy.tier.value,
            "model_attempts": self.model_attempts,
            "tool_attempts": self.tool_attempts,
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "total_tokens": self.total_tokens,
            "events": [event.as_dict() for event in self.events],
        }


class BudgetGovernor:
    """Counts every attempted call and rejects work before a hard limit is crossed."""

    def __init__(self, policy: RoutingPolicy) -> None:
        self.policy = policy
        self.ledger = BudgetLedger(policy=policy)
        self._started_clock = perf_counter()

    def reserve_model_call(self, phase: str) -> None:
        self._raise_if_time_exceeded(phase)
        next_count = self.ledger.model_attempts + 1
        if next_count > self.policy.budget.max_model_calls:
            reason = _reason("model call", next_count, self.policy.budget.max_model_calls)
            self.ledger.events.append(BudgetEvent("model_call", phase, False, reason))
            raise BudgetExceeded(reason)
        self.ledger.model_attempts = next_count
        self.ledger.events.append(BudgetEvent("model_call", phase, True))

    def record_model_usage(self, phase: str, usage: ModelUsage) -> None:
        input_tokens = usage.input_tokens or 0
        output_tokens = usage.output_tokens or 0
        total_tokens = usage.total_tokens
        if total_tokens is None:
            total_tokens = input_tokens + output_tokens
        self.ledger.input_tokens += input_tokens
        self.ledger.output_tokens += output_tokens
        self.ledger.total_tokens += total_tokens
        self.ledger.events.append(
            BudgetEvent(
                "model_usage",
                phase,
                True,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                total_tokens=total_tokens,
            )
        )
        self._raise_if_token_budget_exceeded(self.policy.budget, phase)

    def reserve_tool_call(self, phase: str) -> None:
        self._raise_if_time_exceeded(phase)
        next_count = self.ledger.tool_attempts + 1
        if next_count > self.policy.budget.max_tool_calls:
            reason = _reason("tool call", next_count, self.policy.budget.max_tool_calls)
            self.ledger.events.append(BudgetEvent("tool_call", phase, False, reason))
            raise BudgetExceeded(reason)
        self.ledger.tool_attempts = next_count
        self.ledger.events.append(BudgetEvent("tool_call", phase, True))

    def _raise_if_token_budget_exceeded(self, budget: BudgetPolicy, phase: str) -> None:
        failures = []
        if budget.max_input_tokens is not None and self.ledger.input_tokens > budget.max_input_tokens:
            failures.append(
                _reason("input token", self.ledger.input_tokens, budget.max_input_tokens)
            )
        if self.ledger.output_tokens > budget.max_output_tokens:
            failures.append(
                _reason("output token", self.ledger.output_tokens, budget.max_output_tokens)
            )
        if budget.max_total_tokens is not None and self.ledger.total_tokens > budget.max_total_tokens:
            failures.append(
                _reason("total token", self.ledger.total_tokens, budget.max_total_tokens)
            )
        if failures:
            reason = "; ".join(failures)
            self.ledger.events.append(BudgetEvent("budget_exceeded", phase, False, reason))
            raise BudgetExceeded(reason)

    def _raise_if_time_exceeded(self, phase: str) -> None:
        elapsed = perf_counter() - self._started_clock
        if elapsed <= self.policy.budget.max_wallclock_seconds:
            return
        reason = (
            f"wallclock budget exceeded: {elapsed:.3f}s > "
            f"{self.policy.budget.max_wallclock_seconds:.3f}s"
        )
        self.ledger.events.append(BudgetEvent("budget_exceeded", phase, False, reason))
        raise BudgetExceeded(reason)


def _reason(label: str, observed: int, limit: int) -> str:
    return f"{label} budget exceeded: {observed} > {limit}"
