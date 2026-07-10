from __future__ import annotations

import unittest

from agi_contracts import AgentProfile, RiskLevel, ToolRequest
from agi_tools import BaseTool, ToolRegistry
from pydantic import BaseModel, ConfigDict


class EmptyInput(BaseModel):
    model_config = ConfigDict(extra="forbid")


class CountOutput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    calls: int


class CountingTool(BaseTool[EmptyInput, CountOutput]):
    name = "counting"
    description = "A test-only invocation counter."
    risk_level = RiskLevel.LOW
    permissions_required = ("counting.execute",)
    human_approval_required = False
    timeout_seconds = 1.0
    input_model = EmptyInput
    output_model = CountOutput

    def __init__(self, events: list[str] | None = None) -> None:
        self.calls = 0
        self.events = events if events is not None else []

    async def run(self, tool_input: EmptyInput) -> CountOutput:
        self.events.append("run")
        self.calls += 1
        return CountOutput(calls=self.calls)


class RegistryTests(unittest.IsolatedAsyncioTestCase):
    async def test_missing_permission_is_logged_and_never_invoked(self) -> None:
        tool = CountingTool()
        registry = ToolRegistry()
        registry.register(tool)
        request = ToolRequest(tool_name="counting", arguments={})

        decision, result = await registry.execute(request, granted_permissions=set())

        self.assertFalse(decision.allowed)
        self.assertFalse(result.success)
        self.assertEqual(tool.calls, 0)
        self.assertEqual(registry.decision_log, (decision,))

    async def test_unknown_tool_is_denied_without_dispatch(self) -> None:
        registry = ToolRegistry()
        request = ToolRequest(tool_name="delete_file", arguments={"path": "fixture.md"})

        decision, result = await registry.execute(request, granted_permissions=set())

        self.assertFalse(decision.allowed)
        self.assertTrue(decision.human_approval_required)
        self.assertFalse(result.success)
        self.assertIn("not registered", decision.reason)

    async def test_decision_recorder_runs_before_tool(self) -> None:
        events: list[str] = []

        def recorder(_decision) -> None:
            events.append("decision")

        tool = CountingTool(events)
        registry = ToolRegistry(decision_recorder=recorder)
        registry.register(tool)
        request = ToolRequest(tool_name="counting", arguments={})

        decision, result = await registry.execute(
            request,
            granted_permissions={"counting.execute"},
        )

        self.assertTrue(decision.allowed)
        self.assertTrue(result.success)
        self.assertEqual(events, ["decision", "run"])

    async def test_audit_recorder_failure_denies_before_tool(self) -> None:
        def broken_recorder(_decision) -> None:
            raise OSError("simulated logging failure")

        tool = CountingTool()
        registry = ToolRegistry(decision_recorder=broken_recorder)
        registry.register(tool)
        request = ToolRequest(tool_name="counting", arguments={})

        decision, result = await registry.execute(
            request,
            granted_permissions={"counting.execute"},
        )

        self.assertFalse(decision.allowed)
        self.assertFalse(result.success)
        self.assertEqual(tool.calls, 0)
        self.assertIn("could not be recorded", decision.reason)

    async def test_profile_tool_allowlist_and_approval_policy_are_enforced(self) -> None:
        tool = CountingTool()
        registry = ToolRegistry()
        registry.register(tool)
        request = ToolRequest(tool_name="counting", arguments={})
        base_profile = {
            "permissions": ["counting.execute"],
            "allowed_tools": [],
            "approval_policy": {
                "default_requires_human_approval": False,
                "tools_requiring_approval": ["counting"],
                "risk_levels_requiring_approval": [],
            },
        }

        not_allowlisted, _ = await registry.execute(
            request,
            agent_profile=base_profile,
        )
        base_profile["allowed_tools"] = ["counting"]
        not_approved, _ = await registry.execute(request, agent_profile=base_profile)
        allowed, result = await registry.execute(
            request,
            agent_profile=base_profile,
            human_approved=True,
        )

        self.assertFalse(not_allowlisted.allowed)
        self.assertIn("not allowed", not_allowlisted.reason)
        self.assertFalse(not_approved.allowed)
        self.assertTrue(allowed.allowed)
        self.assertTrue(result.success)
        self.assertEqual(tool.calls, 1)

    async def test_shared_agent_profile_contract_is_accepted(self) -> None:
        tool = CountingTool()
        registry = ToolRegistry()
        registry.register(tool)
        profile = AgentProfile(
            id="test_agent",
            description="Test profile for governed tool dispatch.",
            model="test-model",
            allowed_tools=["counting"],
            permissions=["counting.execute"],
        )

        decision, result = await registry.execute(
            ToolRequest(tool_name="counting", arguments={}),
            agent_profile=profile,
        )

        self.assertTrue(decision.allowed)
        self.assertTrue(result.success)
        self.assertEqual(tool.calls, 1)

    def test_duplicate_registration_is_rejected(self) -> None:
        registry = ToolRegistry()
        registry.register(CountingTool())

        with self.assertRaises(ValueError):
            registry.register(CountingTool())


if __name__ == "__main__":
    unittest.main()
