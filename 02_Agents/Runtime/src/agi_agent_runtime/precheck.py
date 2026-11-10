"""Host-side checks that run before model or tool attempts."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

from agi_contracts import AgentProfile, ToolDefinition

from .routing import RoutingPolicy


class HostPrecheckError(ValueError):
    """Raised when a run configuration cannot satisfy its routing contract."""


@dataclass(frozen=True)
class HostPrecheckResult:
    policy_tier: str
    model: str
    allowed_tool_count: int
    available_tool_count: int
    warnings: tuple[str, ...] = ()

    def as_dict(self) -> dict[str, object]:
        return {
            "policy_tier": self.policy_tier,
            "model": self.model,
            "allowed_tool_count": self.allowed_tool_count,
            "available_tool_count": self.available_tool_count,
            "warnings": list(self.warnings),
        }


def run_host_precheck(
    *,
    policy: RoutingPolicy,
    profile: AgentProfile,
    tools: Mapping[str, ToolDefinition],
) -> HostPrecheckResult:
    """Validate host-owned configuration before any model call begins."""

    missing_tools = sorted(set(profile.allowed_tools) - set(tools))
    if missing_tools:
        raise HostPrecheckError(
            "profile allows tools that are not registered: " + ", ".join(missing_tools)
        )
    if policy.allow_tools and policy.budget.max_tool_calls < 1:
        raise HostPrecheckError("tool-enabled routing policy must allow at least one tool call")
    if not policy.allow_tools and policy.budget.max_tool_calls != 0:
        raise HostPrecheckError("tool-disabled routing policy must set max_tool_calls to 0")
    if not policy.allow_critique and policy.allow_revision:
        warnings = ("revision is enabled without critique; revisions must be rule-triggered",)
    else:
        warnings = ()
    if not policy.allow_tools and profile.allowed_tools:
        warnings = warnings + ("profile tools are configured but disabled by routing policy",)
    return HostPrecheckResult(
        policy_tier=policy.tier.value,
        model=profile.model,
        allowed_tool_count=len(profile.allowed_tools),
        available_tool_count=len(tools),
        warnings=warnings,
    )
