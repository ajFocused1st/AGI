from __future__ import annotations

import unittest
from datetime import UTC, datetime
from uuid import uuid4

from agi_contracts import (
    AgentRequest,
    Citation,
    EvidenceReference,
    PermissionDecision,
    StepStatus,
    TaskPlan,
    TaskStep,
    ToolRequest,
    ToolResult,
)
from agi_evaluation import EvaluationExpectations, RuleBasedEvaluator


class RuleBasedEvaluatorTests(unittest.TestCase):
    def test_grounded_tool_run_passes(self) -> None:
        now = datetime.now(UTC)
        run_id = uuid4()
        request = AgentRequest(goal="Who owns Atlas?")
        step = TaskStep(
            description="Search the corpus",
            status=StepStatus.COMPLETED,
            tool_name="read_only_file_search",
            tool_arguments={"query": "owner"},
        )
        plan = TaskPlan(goal=request.goal, steps=[step])
        tool_request = ToolRequest(
            tool_name="read_only_file_search",
            arguments={"query": "owner"},
            step_id=step.step_id,
        )
        decision = PermissionDecision(
            request_id=tool_request.request_id,
            tool_name=tool_request.tool_name,
            allowed=True,
            reason="allowed",
            permissions_checked=["filesystem.search.readonly"],
            decided_at=now,
        )
        result = ToolResult(
            request_id=tool_request.request_id,
            tool_name=tool_request.tool_name,
            success=True,
            output={},
            started_at=now,
            completed_at=now,
            duration_ms=0,
        )
        evidence = EvidenceReference(
            source_id="corpus:atlas.md",
            source_path="atlas.md",
            locator="line 6",
            excerpt="[A4] Mina Chen owns the Atlas review.",
        )
        citation = Citation(evidence_id=evidence.evidence_id, label="A4")

        evaluation = RuleBasedEvaluator().evaluate(
            run_id=run_id,
            request=request,
            plan=plan,
            answer="Mina Chen owns the review.",
            evidence=[evidence],
            citations=[citation],
            tool_requests=[tool_request],
            tool_results=[result],
            permission_decisions=[decision],
            expectations=EvaluationExpectations(
                required_answer_substrings=("Mina Chen",),
                required_evidence_markers=("A4",),
                expected_tools=("read_only_file_search",),
                require_citations=True,
            ),
            unsupported_claims=[],
        )

        self.assertEqual(evaluation.verdict.value, "passed")
        self.assertTrue(
            next(
                check
                for check in evaluation.rule_based_checks
                if check.name == "permission_before_execution"
            ).passed
        )

    def test_permission_bypass_is_critical_failure(self) -> None:
        now = datetime.now(UTC)
        request = AgentRequest(goal="Do something")
        step = TaskStep(description="Run", status=StepStatus.COMPLETED)
        plan = TaskPlan(goal=request.goal, steps=[step])
        tool_request = ToolRequest(tool_name="calculator", arguments={"expression": "1+1"})
        result = ToolResult(
            request_id=tool_request.request_id,
            tool_name="calculator",
            success=True,
            output={"value": "2"},
            started_at=now,
            completed_at=now,
            duration_ms=0,
        )
        evaluation = RuleBasedEvaluator().evaluate(
            run_id=uuid4(),
            request=request,
            plan=plan,
            answer="2",
            evidence=[],
            citations=[],
            tool_requests=[tool_request],
            tool_results=[result],
            permission_decisions=[],
            unsupported_claims=[],
        )
        self.assertEqual(evaluation.verdict.value, "failed")


if __name__ == "__main__":
    unittest.main()
