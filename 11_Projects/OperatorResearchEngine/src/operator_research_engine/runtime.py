"""Thin project facade over the shared agent runtime."""

from __future__ import annotations

from agi_agent_runtime import AgentRuntime
from agi_agent_runtime.requirement_defaults import infer_task_requirements
from agi_contracts import AgentProfile, AgentRequest, AgentResponse, TaskRequirements


class OperatorResearchEngine:
    def __init__(self, *, runtime: AgentRuntime, profile: AgentProfile) -> None:
        self._runtime = runtime
        self._profile = profile

    async def research(
        self, goal: str, *, requirements: TaskRequirements | None = None
    ) -> AgentResponse:
        request = AgentRequest(goal=goal, agent_profile_id=self._profile.id)
        if requirements is None:
            # Never run ungoverned: infer conservative host-side requirements.
            usable_tools = (
                self._profile.allowed_tools if self._runtime.routing_policy.allow_tools else []
            )
            requirements = infer_task_requirements(goal, usable_tools)
        return await self._runtime.run(request, self._profile, requirements=requirements)
