"""Regression guards for defects confirmed in correction-plan step 1, fixed in steps 3-6.

Each test asserts the correct behavior. They were marked expectedFailure until their fix
landed; the resulting "unexpected success" forced removing the marker.
"""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from agi_agent_runtime import AgentRuntime, AtomicJsonRunStore, RuntimeExecutionError
from agi_agent_runtime.evidence import build_evidence_bundle
from agi_agent_runtime.routing import BudgetPolicy, ReasoningTier, RoutingPolicy
from agi_contracts import AgentProfile, AgentRequest, StepStatus, ToolRequest
from agi_evaluation import RuleBasedEvaluator
from agi_ollama import ScriptedModelAdapter, ScriptedReply
from agi_tools import (
    Calculator,
    ReadOnlyFileReader,
    ReadOnlyFileReaderConfig,
    SearchRoot,
    ToolRegistry,
)

ANSWER = ScriptedReply(content="", structured_output={"answer": "done"})


def _plan(*tool_calls: dict) -> ScriptedReply:
    return ScriptedReply(
        content="",
        structured_output={
            "steps": [{"step_id": 1, "description": "Do the step"}],
            "tool_calls": list(tool_calls),
        },
    )


def _profile(tools: list[str], permissions: list[str], roots: list[str] | None = None) -> AgentProfile:
    return AgentProfile(
        id="researcher",
        description="defect probe",
        model="scripted",
        allowed_tools=tools,
        permissions=permissions,
        allowed_document_roots=roots or [],
        max_steps=4,
        max_tool_calls=4,
    )


class KnownDefectTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self) -> None:
        self._temporary = tempfile.TemporaryDirectory()
        self.logs = Path(self._temporary.name) / "logs"

    def tearDown(self) -> None:
        self._temporary.cleanup()

    def _runtime(self, adapter: ScriptedModelAdapter, registry: ToolRegistry, policy) -> AgentRuntime:
        return AgentRuntime(
            model_adapter=adapter,
            tool_executor=registry,
            evaluator=RuleBasedEvaluator(),
            run_store=AtomicJsonRunStore(self.logs),
            enable_model_critique=False,
            routing_policy=policy,
        )

    async def test_fast_tier_completes_a_simple_question(self) -> None:
        # Defect: FAST allows 1 model call but the runtime always needs plan + answer.
        registry = ToolRegistry()
        registry.register(Calculator())
        adapter = ScriptedModelAdapter([ANSWER])
        runtime = self._runtime(adapter, registry, ReasoningTier.FAST)
        request = AgentRequest(goal="What is the capital of France?", agent_profile_id="researcher")
        try:
            response = await runtime.run(
                request, _profile(["calculator"], ["calculator.execute"])
            )
        except RuntimeExecutionError as exc:
            self.fail(f"FAST tier run failed instead of completing: {exc.__cause__!r}")
        self.assertEqual(response.status.value, "completed")
        self.assertEqual(len(adapter.requests), 1, "FAST must make exactly one model call")

    async def test_file_reader_output_becomes_citable_evidence(self) -> None:
        # Defect: evidence bundle only accepts calculator and file-search results.
        with tempfile.TemporaryDirectory() as documents:
            Path(documents, "note.md").write_text("[N1] The probe value is 42.\n", encoding="utf-8")
            root = SearchRoot(label="documents1", path=Path(documents))
            registry = ToolRegistry()
            registry.register(ReadOnlyFileReader(ReadOnlyFileReaderConfig(roots=(root,))))
            request = ToolRequest(
                tool_name="read_only_file_reader",
                arguments={"root_label": "documents1", "relative_path": "note.md"},
            )
            _, result = await registry.execute(
                request,
                agent_profile=_profile(
                    ["read_only_file_reader"], ["filesystem.read.readonly"], [documents]
                ),
                human_approved=False,
            )
        self.assertTrue(result.success, result.error)
        bundle = build_evidence_bundle([request], [result])
        self.assertGreaterEqual(bundle.evidence_count, 1)

    async def test_step_with_failed_tool_is_not_marked_completed(self) -> None:
        # Defect: every plan step is set to COMPLETED regardless of tool outcomes.
        registry = ToolRegistry()
        registry.register(Calculator())
        adapter = ScriptedModelAdapter(
            [
                _plan({"tool_name": "calculator", "arguments": {"expression": "1 / 0"}, "related_step_id": 1}),
                ANSWER,
            ]
        )
        runtime = self._runtime(adapter, registry, ReasoningTier.STANDARD)
        request = AgentRequest(goal="Compute 1 divided by 0.", agent_profile_id="researcher")
        response = await runtime.run(request, _profile(["calculator"], ["calculator.execute"]))
        record_path = self.logs / str(response.run_id) / "run.json"
        from agi_contracts import RunRecord

        record = RunRecord.model_validate_json(record_path.read_text(encoding="utf-8"))
        self.assertFalse(record.tool_results[0].success, "precondition: tool must fail")
        self.assertNotEqual(record.plan.steps[0].status, StepStatus.COMPLETED)

    async def test_over_budget_prompt_is_rejected_before_the_model_call(self) -> None:
        # Defect: token budgets are checked only after the model has been called and paid for.
        policy = RoutingPolicy(
            tier=ReasoningTier.STANDARD,
            budget=BudgetPolicy(
                max_model_calls=5,
                max_tool_calls=4,
                max_input_tokens=200,
                max_output_tokens=200,
                max_total_tokens=400,
                max_wallclock_seconds=60.0,
            ),
            allow_tools=True,
            allow_retrieval=True,
            allow_critique=False,
            allow_revision=True,
            native_thinking=False,
            description="tiny input budget probe",
        )
        registry = ToolRegistry()
        registry.register(Calculator())
        adapter = ScriptedModelAdapter([_plan(), ANSWER])
        runtime = self._runtime(adapter, registry, policy)
        request = AgentRequest(goal="Explain the plan. " * 200, agent_profile_id="researcher")
        # Also covers: adapters that report no usage (None) are counted as 0 tokens today,
        # so the budget is never enforced. A host-side estimate must apply regardless.
        failed = False
        try:
            await runtime.run(request, _profile(["calculator"], ["calculator.execute"]))
        except RuntimeExecutionError:
            failed = True
        self.assertEqual(len(adapter.requests), 0, "model was called with an over-budget prompt")
        self.assertTrue(failed, "over-budget run must fail safely")


class UsageEstimateTests(unittest.TestCase):
    def test_missing_provider_usage_is_estimated_not_zero(self) -> None:
        from agi_agent_runtime.budgets import BudgetGovernor
        from agi_agent_runtime.routing import get_routing_policy
        from agi_contracts import ModelMessage, ModelRequest, ModelResponse, ModelUsage

        request = ModelRequest(
            model="m", messages=[ModelMessage(role="user", content="x" * 400)], max_tokens=64
        )
        response = ModelResponse(
            request_id=request.request_id, model="m", content="y" * 40, usage=ModelUsage()
        )
        governor = BudgetGovernor(get_routing_policy(ReasoningTier.STANDARD))
        governor.reserve_model_call("answer", request)
        governor.record_model_usage("answer", ModelUsage(), request=request, response=response)
        ledger = governor.ledger.as_dict()
        self.assertEqual((ledger["input_tokens"], ledger["output_tokens"]), (100, 10))
        self.assertTrue(ledger["events"][-1]["estimated"])

    def test_reported_provider_usage_is_used_as_is(self) -> None:
        from agi_agent_runtime.budgets import BudgetGovernor
        from agi_agent_runtime.routing import get_routing_policy
        from agi_contracts import ModelUsage

        governor = BudgetGovernor(get_routing_policy(ReasoningTier.STANDARD))
        governor.reserve_model_call("answer")
        governor.record_model_usage(
            "answer", ModelUsage(input_tokens=7, output_tokens=3, total_tokens=10)
        )
        self.assertEqual(governor.ledger.total_tokens, 10)
        self.assertNotIn("estimated", governor.ledger.as_dict()["events"][-1])


class RoutingPolicyFlagTests(unittest.IsolatedAsyncioTestCase):
    async def test_revision_disabled_policy_makes_no_correction_calls(self) -> None:
        policy = RoutingPolicy(
            tier=ReasoningTier.STANDARD,
            budget=BudgetPolicy(
                max_model_calls=5, max_tool_calls=4, max_output_tokens=4096, max_wallclock_seconds=60.0
            ),
            allow_tools=True,
            allow_retrieval=True,
            allow_critique=False,
            allow_revision=False,
            native_thinking=False,
            description="no revision probe",
        )
        registry = ToolRegistry()
        registry.register(Calculator())
        adapter = ScriptedModelAdapter([_plan(), ANSWER])  # plan omits the required tool
        with tempfile.TemporaryDirectory() as temporary:
            runtime = AgentRuntime(
                model_adapter=adapter,
                tool_executor=registry,
                evaluator=RuleBasedEvaluator(),
                run_store=AtomicJsonRunStore(Path(temporary) / "logs"),
                enable_model_critique=False,
                routing_policy=policy,
            )
            from agi_contracts import TaskRequirements

            await runtime.run(
                AgentRequest(goal="Compute 2 + 2.", agent_profile_id="researcher"),
                _profile(["calculator"], ["calculator.execute"]),
                requirements=TaskRequirements(required_capabilities=["calculator"]),
            )
        phases = [request.metadata["phase"] for request in adapter.requests]
        self.assertEqual(phases, ["plan", "answer"])


class DenialTimestampRegressionTests(unittest.TestCase):
    """Fixed in step 3: denial results could predate their decision (~0.1% of records)."""

    def test_denial_results_never_predate_their_decision(self) -> None:
        from agi_agent_runtime.drafts import ToolCallDraft
        from agi_contracts import PlanRequirementResult

        blocked = PlanRequirementResult(
            valid=False, forbidden_capabilities_proposed=["delete_file"]
        )
        proposal = [ToolCallDraft(tool_name="delete_file", arguments={"path": "x.md"})]
        for _ in range(5000):
            for _, results, decisions in (
                AgentRuntime._policy_denials({"calculator"}),
                AgentRuntime._blocked_plan_proposals(proposal, blocked),
            ):
                self.assertLessEqual(decisions[0].decided_at, results[0].started_at)


class ExceptionMessageRedactionTests(unittest.TestCase):
    def test_messages_are_bounded_and_redacted(self) -> None:
        from agi_agent_runtime.runtime import _safe_exception_message

        message = _safe_exception_message(
            ValueError(
                "schema failed at tool_calls.0.purpose; Authorization: Bearer abc123secret "
                "key=rpa_ABCDEFGHIJKLMNOP blob=" + "x" * 64 + " " + "y" * 400
            )
        )
        self.assertIn("schema failed at tool_calls.0.purpose", message)
        for leaked in ("abc123secret", "rpa_ABCDEFGHIJKLMNOP", "x" * 64):
            self.assertNotIn(leaked, message)
        self.assertLessEqual(len(message), 300)


if __name__ == "__main__":
    unittest.main()
