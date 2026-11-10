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

    def test_cited_calculator_evidence_must_match_answer_number(self) -> None:
        now = datetime.now(UTC)
        request = AgentRequest(goal="Calculate 37 * 14")
        plan = TaskPlan(
            goal=request.goal,
            steps=[TaskStep(description="Calculate", status=StepStatus.COMPLETED)],
        )
        tool_request = ToolRequest(
            tool_name="calculator", arguments={"expression": "37 * 14"}
        )
        decision = PermissionDecision(
            request_id=tool_request.request_id,
            tool_name=tool_request.tool_name,
            allowed=True,
            reason="allowed",
            decided_at=now,
        )
        result = ToolResult(
            request_id=tool_request.request_id,
            tool_name=tool_request.tool_name,
            success=True,
            output={"expression": "37 * 14", "value": 518},
            started_at=now,
            completed_at=now,
            duration_ms=0,
        )
        evidence = EvidenceReference(
            source_id="tool:calculator",
            source_path="tool:calculator",
            excerpt="37 * 14 = 518",
            metadata={
                "retrieval_tool": "calculator",
                "expression": "37 * 14",
                "value": 518,
            },
        )
        citation = Citation(evidence_id=evidence.evidence_id)

        bad = RuleBasedEvaluator().evaluate(
            run_id=uuid4(),
            request=request,
            plan=plan,
            answer="37 multiplied by 14 is 519.",
            evidence=[evidence],
            citations=[citation],
            tool_requests=[tool_request],
            tool_results=[result],
            permission_decisions=[decision],
            unsupported_claims=[],
        )
        good = RuleBasedEvaluator().evaluate(
            run_id=uuid4(),
            request=request,
            plan=plan,
            answer="37 multiplied by 14 is 518.",
            evidence=[evidence],
            citations=[citation],
            tool_requests=[tool_request],
            tool_results=[result],
            permission_decisions=[decision],
            unsupported_claims=[],
        )

        self.assertEqual(bad.verdict.value, "failed")
        self.assertFalse(
            next(
                check
                for check in bad.rule_based_checks
                if check.name == "calculator_evidence_consistency"
            ).passed
        )
        self.assertEqual(good.verdict.value, "passed")

    def test_cited_document_evidence_must_support_concrete_claim_terms(self) -> None:
        now = datetime.now(UTC)
        run_id = uuid4()
        request = AgentRequest(goal="Who owns Atlas?")
        plan = TaskPlan(
            goal=request.goal,
            steps=[TaskStep(description="Search", status=StepStatus.COMPLETED)],
        )
        tool_request = ToolRequest(
            tool_name="read_only_file_search", arguments={"query": "Atlas owner"}
        )
        decision = PermissionDecision(
            request_id=tool_request.request_id,
            tool_name=tool_request.tool_name,
            allowed=True,
            reason="allowed",
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
            source_id="documents:atlas.md",
            source_path="atlas.md",
            locator="line 6",
            excerpt="[A4] Mina Chen owns the Atlas review.",
            metadata={"retrieval_tool": "read_only_file_search"},
        )
        citation = Citation(evidence_id=evidence.evidence_id)

        good = RuleBasedEvaluator().evaluate(
            run_id=run_id,
            request=request,
            plan=plan,
            answer="Mina Chen owns the Atlas review.",
            evidence=[evidence],
            citations=[citation],
            tool_requests=[tool_request],
            tool_results=[result],
            permission_decisions=[decision],
            unsupported_claims=[],
        )
        bad = RuleBasedEvaluator().evaluate(
            run_id=run_id,
            request=request,
            plan=plan,
            answer="Nora Patel owns the Atlas review.",
            evidence=[evidence],
            citations=[citation],
            tool_requests=[tool_request],
            tool_results=[result],
            permission_decisions=[decision],
            unsupported_claims=[],
        )

        self.assertEqual(good.verdict.value, "passed")
        self.assertEqual(bad.verdict.value, "failed")
        self.assertFalse(
            next(
                check
                for check in bad.rule_based_checks
                if check.name == "document_evidence_support"
            ).passed
        )

    def test_missing_unsupported_claim_analysis_is_not_a_failure(self) -> None:
        result = RuleBasedEvaluator().evaluate(
            run_id=uuid4(),
            request=AgentRequest(goal="Say hello"),
            plan=TaskPlan(
                goal="Say hello",
                steps=[TaskStep(description="Answer", status=StepStatus.COMPLETED)],
            ),
            answer="Hello.",
            evidence=[],
            citations=[],
            tool_requests=[],
            tool_results=[],
            permission_decisions=[],
            unsupported_claims=None,
        )

        unsupported = next(
            check
            for check in result.rule_based_checks
            if check.name == "unsupported_claims"
        )
        self.assertTrue(unsupported.passed)
        self.assertEqual(result.verdict.value, "passed")

    def test_document_support_ignores_rendered_citation_labels(self) -> None:
        now = datetime.now(UTC)
        evidence = EvidenceReference(
            source_id="documents:atlas.md",
            source_path="atlas.md",
            locator="line 3",
            excerpt="[A1] Project Atlas launched on 2025-02-14.",
            metadata={"retrieval_tool": "read_only_file_search"},
        )
        citation = Citation(evidence_id=evidence.evidence_id)
        tool_request = ToolRequest(
            tool_name="read_only_file_search",
            arguments={"query": "Project Atlas launch date"},
        )

        result = RuleBasedEvaluator().evaluate(
            run_id=uuid4(),
            request=AgentRequest(goal="When did Atlas launch?"),
            plan=TaskPlan(
                goal="When did Atlas launch?",
                steps=[TaskStep(description="Search", status=StepStatus.COMPLETED)],
            ),
            answer="Project Atlas launched on 2025-02-14. [atlas.md (lines 3-3)]",
            evidence=[evidence],
            citations=[citation],
            tool_requests=[tool_request],
            tool_results=[
                ToolResult(
                    request_id=tool_request.request_id,
                    tool_name="read_only_file_search",
                    success=True,
                    output={},
                    started_at=now,
                    completed_at=now,
                    duration_ms=0,
                )
            ],
            permission_decisions=[
                PermissionDecision(
                    request_id=tool_request.request_id,
                    tool_name="read_only_file_search",
                    allowed=True,
                    reason="allowed",
                    decided_at=now,
                )
            ],
            unsupported_claims=[],
        )

        support = next(
            check
            for check in result.rule_based_checks
            if check.name == "document_evidence_support"
        )
        self.assertTrue(support.passed)
        self.assertEqual(result.verdict.value, "passed")


if __name__ == "__main__":
    unittest.main()
