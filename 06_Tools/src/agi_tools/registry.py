"""Permission-aware tool registration and dispatch."""

from __future__ import annotations

import inspect
from collections.abc import Awaitable, Callable, Collection, Mapping
from dataclasses import dataclass
from datetime import UTC, datetime
from types import MappingProxyType
from typing import Protocol, TypeAlias

from agi_contracts import (
    PermissionDecision,
    RiskLevel,
    ToolDefinition,
    ToolRequest,
    ToolResult,
)

from .base import BaseTool


class ApprovalPolicyLike(Protocol):
    """The approval settings consumed from a shared agent profile."""

    default_requires_human_approval: bool
    tools_requiring_approval: Collection[str]
    risk_levels_requiring_approval: Collection[RiskLevel]


class AgentProfileLike(Protocol):
    """The subset of an agent profile used by the permission boundary."""

    permissions: Collection[str]
    allowed_tools: Collection[str]
    approval_policy: ApprovalPolicyLike


@dataclass(frozen=True, slots=True)
class _ApprovalContext:
    default_required: bool = False
    tools_requiring_approval: frozenset[str] = frozenset()
    risk_levels_requiring_approval: frozenset[RiskLevel] = frozenset()

    def requires_approval(self, definition: ToolDefinition) -> bool:
        return bool(
            definition.human_approval_required
            or self.default_required
            or definition.name in self.tools_requiring_approval
            or definition.risk_level in self.risk_levels_requiring_approval
        )


DecisionRecorder: TypeAlias = Callable[[PermissionDecision], None | Awaitable[None]]


class ToolRegistry:
    """An allowlist registry that makes a permission decision before dispatch."""

    def __init__(self, *, decision_recorder: DecisionRecorder | None = None) -> None:
        self._tools: dict[str, BaseTool] = {}
        self._decisions: list[PermissionDecision] = []
        self._decision_recorder = decision_recorder

    @property
    def decision_log(self) -> tuple[PermissionDecision, ...]:
        """Return an immutable snapshot of decisions in execution order."""

        return tuple(self._decisions)

    @property
    def tools(self) -> Mapping[str, ToolDefinition]:
        """Return definitions only; executable tool objects remain private."""

        return MappingProxyType(
            {name: tool.definition for name, tool in sorted(self._tools.items())}
        )

    def register(self, tool: BaseTool) -> None:
        """Register one tool, rejecting ambiguous duplicate names."""

        if not isinstance(tool, BaseTool):
            raise TypeError("tool must inherit BaseTool")
        if not tool.name or tool.name.strip() != tool.name:
            raise ValueError("tool name must be a non-empty normalized string")
        if tool.name in self._tools:
            raise ValueError(f"tool is already registered: {tool.name}")
        # Constructing the shared contract validates schemas and governance metadata
        # at registration rather than on the first production request.
        _ = tool.definition
        self._tools[tool.name] = tool

    async def execute(
        self,
        request: ToolRequest,
        *,
        agent_profile: AgentProfileLike | Mapping[str, object] | None = None,
        granted_permissions: Collection[str] | None = None,
        human_approved: bool = False,
    ) -> tuple[PermissionDecision, ToolResult]:
        """Decide, record, and only then execute an allowlisted tool.

        ``agent_profile`` and ``granted_permissions`` are mutually exclusive. The
        latter is convenient for tightly scoped runtimes and deterministic tests.
        Permission strings are matched exactly; wildcard grants are not supported.
        """

        if agent_profile is not None and granted_permissions is not None:
            raise ValueError("provide an agent profile or explicit permissions, not both")

        permissions, allowed_tools, approval_context = self._permission_context(
            agent_profile=agent_profile,
            granted_permissions=granted_permissions,
        )
        tool = self._tools.get(request.tool_name)

        if tool is None:
            decision = self._decision(
                request=request,
                allowed=False,
                reason="tool is not registered",
                permissions_checked=(),
                human_approval_required=True,
            )
        else:
            required = tuple(tool.definition.permissions_required)
            missing = sorted(set(required).difference(permissions))
            approval_required = approval_context.requires_approval(tool.definition)
            if allowed_tools is not None and tool.name not in allowed_tools:
                decision = self._decision(
                    request=request,
                    allowed=False,
                    reason="tool is not allowed by the agent profile",
                    permissions_checked=required,
                    human_approval_required=approval_required,
                )
            elif missing:
                decision = self._decision(
                    request=request,
                    allowed=False,
                    reason=f"missing required permissions: {', '.join(missing)}",
                    permissions_checked=required,
                    human_approval_required=approval_required,
                )
            elif approval_required and not human_approved:
                decision = self._decision(
                    request=request,
                    allowed=False,
                    reason="human approval is required",
                    permissions_checked=required,
                    human_approval_required=True,
                )
            else:
                decision = self._decision(
                    request=request,
                    allowed=True,
                    reason="required permissions are granted",
                    permissions_checked=required,
                    human_approval_required=approval_required,
                )

        recorder_succeeded = await self._record_decision(decision)
        if not recorder_succeeded and decision.allowed:
            # Audit recording is part of the security boundary. Fail closed if the
            # configured recorder cannot persist an otherwise-allowed decision.
            decision = self._decision(
                request=request,
                allowed=False,
                reason="permission decision could not be recorded",
                permissions_checked=tuple(decision.permissions_checked),
                human_approval_required=decision.human_approval_required,
            )
            self._decisions[-1] = decision

        if not decision.allowed or tool is None:
            return decision, self._denied_result(request, decision.reason)

        result = await tool._execute_request(request)
        return decision, result

    async def _record_decision(self, decision: PermissionDecision) -> bool:
        self._decisions.append(decision)
        if self._decision_recorder is None:
            return True
        try:
            maybe_awaitable = self._decision_recorder(decision)
            if inspect.isawaitable(maybe_awaitable):
                await maybe_awaitable
        except Exception:
            return False
        return True

    @staticmethod
    def _permission_context(
        *,
        agent_profile: AgentProfileLike | Mapping[str, object] | None,
        granted_permissions: Collection[str] | None,
    ) -> tuple[frozenset[str], frozenset[str] | None, _ApprovalContext]:
        if granted_permissions is not None:
            return (
                ToolRegistry._normalize_permissions(granted_permissions),
                None,
                _ApprovalContext(),
            )
        if agent_profile is None:
            return frozenset(), None, _ApprovalContext()

        if isinstance(agent_profile, Mapping):
            raw_permissions = agent_profile.get("permissions", ())
            raw_allowed_tools = agent_profile.get("allowed_tools", ())
            raw_policy = agent_profile.get("approval_policy")
        else:
            raw_permissions = getattr(agent_profile, "permissions", ())
            raw_allowed_tools = getattr(agent_profile, "allowed_tools", ())
            raw_policy = getattr(agent_profile, "approval_policy", None)

        if isinstance(raw_permissions, str) or not isinstance(raw_permissions, Collection):
            raise TypeError("agent profile permissions must be a collection of strings")
        if isinstance(raw_allowed_tools, str) or not isinstance(raw_allowed_tools, Collection):
            raise TypeError("agent profile allowed tools must be a collection of strings")
        return (
            ToolRegistry._normalize_permissions(raw_permissions),
            ToolRegistry._normalize_names(raw_allowed_tools, label="allowed tools"),
            ToolRegistry._approval_context(raw_policy),
        )

    @staticmethod
    def _approval_context(
        raw_policy: ApprovalPolicyLike | Mapping[str, object] | None,
    ) -> _ApprovalContext:
        if raw_policy is None:
            return _ApprovalContext()
        if isinstance(raw_policy, Mapping):
            default_required = bool(raw_policy.get("default_requires_human_approval", False))
            raw_tools = raw_policy.get("tools_requiring_approval", ())
            raw_risk_levels = raw_policy.get("risk_levels_requiring_approval", ())
        else:
            default_required = bool(getattr(raw_policy, "default_requires_human_approval", False))
            raw_tools = getattr(raw_policy, "tools_requiring_approval", ())
            raw_risk_levels = getattr(raw_policy, "risk_levels_requiring_approval", ())

        if isinstance(raw_tools, str) or not isinstance(raw_tools, Collection):
            raise TypeError("approval policy tools must be a collection of strings")
        if any(not isinstance(tool_name, str) or not tool_name for tool_name in raw_tools):
            raise TypeError("approval policy tools must contain non-empty strings")
        if isinstance(raw_risk_levels, str) or not isinstance(raw_risk_levels, Collection):
            raise TypeError("approval policy risk levels must be a collection")
        try:
            risk_levels = frozenset(RiskLevel(level) for level in raw_risk_levels)
        except (TypeError, ValueError) as exc:
            raise TypeError("approval policy contains an invalid risk level") from exc
        return _ApprovalContext(
            default_required=default_required,
            tools_requiring_approval=frozenset(raw_tools),
            risk_levels_requiring_approval=risk_levels,
        )

    @staticmethod
    def _normalize_permissions(permissions: Collection[str]) -> frozenset[str]:
        return ToolRegistry._normalize_names(permissions, label="permissions")

    @staticmethod
    def _normalize_names(values: Collection[str], *, label: str) -> frozenset[str]:
        if any(not isinstance(value, str) or not value for value in values):
            raise TypeError(f"{label} must contain non-empty strings")
        return frozenset(values)

    @staticmethod
    def _decision(
        *,
        request: ToolRequest,
        allowed: bool,
        reason: str,
        permissions_checked: Collection[str],
        human_approval_required: bool,
    ) -> PermissionDecision:
        return PermissionDecision(
            request_id=request.request_id,
            tool_name=request.tool_name,
            allowed=allowed,
            reason=reason,
            permissions_checked=list(permissions_checked),
            human_approval_required=human_approval_required,
            decided_at=datetime.now(UTC),
        )

    @staticmethod
    def _denied_result(request: ToolRequest, reason: str) -> ToolResult:
        now = datetime.now(UTC)
        return ToolResult(
            request_id=request.request_id,
            tool_name=request.tool_name,
            success=False,
            output={},
            error=f"tool request denied: {reason}",
            started_at=now,
            completed_at=now,
            duration_ms=0.0,
        )
