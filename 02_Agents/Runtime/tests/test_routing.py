from __future__ import annotations

import unittest

from agi_agent_runtime import (
    BudgetExceeded,
    BudgetGovernor,
    ReasoningTier,
    get_routing_policy,
)
from agi_contracts import ModelUsage


class RoutingPolicyTests(unittest.TestCase):
    def test_fast_policy_is_one_direct_call_without_tools(self) -> None:
        policy = get_routing_policy(ReasoningTier.FAST)

        self.assertEqual(policy.budget.max_model_calls, 1)
        self.assertEqual(policy.budget.max_tool_calls, 0)
        self.assertFalse(policy.allow_tools)
        self.assertFalse(policy.allow_retrieval)
        self.assertFalse(policy.allow_critique)
        self.assertFalse(policy.allow_revision)
        self.assertFalse(policy.native_thinking)

    def test_budget_governor_counts_attempts_before_rejecting(self) -> None:
        governor = BudgetGovernor(get_routing_policy(ReasoningTier.FAST))

        governor.reserve_model_call("fast_answer")

        with self.assertRaises(BudgetExceeded):
            governor.reserve_model_call("hidden_router")

        self.assertEqual(governor.ledger.model_attempts, 1)
        self.assertFalse(governor.ledger.events[-1].accepted)
        self.assertIn("model call budget exceeded", governor.ledger.events[-1].reason)

    def test_budget_governor_rejects_excess_token_usage(self) -> None:
        governor = BudgetGovernor(get_routing_policy(ReasoningTier.FAST))
        governor.reserve_model_call("fast_answer")

        with self.assertRaises(BudgetExceeded):
            governor.record_model_usage(
                "fast_answer",
                ModelUsage(input_tokens=4096, output_tokens=513, total_tokens=4609),
            )

        self.assertEqual(governor.ledger.output_tokens, 513)
        self.assertFalse(governor.ledger.events[-1].accepted)

    def test_fast_policy_rejects_tool_calls(self) -> None:
        governor = BudgetGovernor(get_routing_policy(ReasoningTier.FAST))

        with self.assertRaises(BudgetExceeded):
            governor.reserve_tool_call("calculator")

        self.assertEqual(governor.ledger.tool_attempts, 0)
        self.assertFalse(governor.ledger.events[-1].accepted)


if __name__ == "__main__":
    unittest.main()
