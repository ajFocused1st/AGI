from __future__ import annotations

import unittest

from agi_agent_runtime import HostPrecheckError, ReasoningTier, get_routing_policy, run_host_precheck
from agi_contracts import AgentProfile
from agi_tools import Calculator


class HostPrecheckTests(unittest.TestCase):
    def test_accepts_matching_profile_policy_and_tools(self) -> None:
        calculator = Calculator()
        result = run_host_precheck(
            policy=get_routing_policy(ReasoningTier.STANDARD),
            profile=AgentProfile(
                id="researcher",
                description="Test",
                model="scripted",
                allowed_tools=[calculator.definition.name],
            ),
            tools={calculator.definition.name: calculator.definition},
        )

        self.assertEqual(result.policy_tier, "standard")
        self.assertEqual(result.allowed_tool_count, 1)

    def test_rejects_missing_allowed_tool(self) -> None:
        with self.assertRaises(HostPrecheckError):
            run_host_precheck(
                policy=get_routing_policy(ReasoningTier.STANDARD),
                profile=AgentProfile(
                    id="researcher",
                    description="Test",
                    model="scripted",
                    allowed_tools=["read_only_file_search"],
                ),
                tools={},
            )

    def test_fast_policy_warns_when_profile_tools_are_configured(self) -> None:
        calculator = Calculator()
        result = run_host_precheck(
            policy=get_routing_policy(ReasoningTier.FAST),
            profile=AgentProfile(
                id="researcher",
                description="Test",
                model="scripted",
                allowed_tools=[calculator.definition.name],
            ),
            tools={calculator.definition.name: calculator.definition},
        )

        self.assertIn("disabled by routing policy", result.warnings[-1])


if __name__ == "__main__":
    unittest.main()
