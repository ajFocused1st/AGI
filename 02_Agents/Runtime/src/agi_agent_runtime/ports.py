"""Structural ports that keep orchestration independent of providers and tools."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Protocol

from agi_contracts import (
    AgentProfile,
    ModelRequest,
    ModelResponse,
    PermissionDecision,
    ToolDefinition,
    ToolRequest,
    ToolResult,
)


class ModelAdapter(Protocol):
    async def generate(self, request: ModelRequest) -> ModelResponse: ...


class ToolExecutor(Protocol):
    @property
    def tools(self) -> Mapping[str, ToolDefinition]: ...

    async def execute(
        self,
        request: ToolRequest,
        *,
        agent_profile: AgentProfile,
        human_approved: bool = False,
    ) -> tuple[PermissionDecision, ToolResult]: ...
