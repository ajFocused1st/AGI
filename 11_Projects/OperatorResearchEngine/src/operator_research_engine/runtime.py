"""Thin project facade over the shared agent runtime."""

from __future__ import annotations

from agi_agent_runtime import AgentRuntime
from agi_contracts import AgentProfile, AgentRequest, AgentResponse, TaskRequirements


class OperatorResearchEngine:
    def __init__(self, *, runtime: AgentRuntime, profile: AgentProfile) -> None:
        self._runtime = runtime
        self._profile = profile

    async def research(
        self, goal: str, *, requirements: TaskRequirements | None = None
    ) -> AgentResponse:
        request = AgentRequest(goal=goal, agent_profile_id=self._profile.id)
        return await self._runtime.run(request, self._profile, requirements=requirements)
