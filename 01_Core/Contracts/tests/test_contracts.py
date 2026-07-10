from __future__ import annotations

import unittest
from datetime import datetime, timedelta, timezone
from uuid import uuid4

from pydantic import ValidationError

from agi_contracts import (
    AgentProfile,
    AgentRequest,
    AgentResponse,
    Citation,
    EvaluationResult,
    EvaluationScores,
    EvaluationVerdict,
    EvidenceReference,
    MessageRole,
    ModelMessage,
    ModelRequest,
    ModelResponse,
    PermissionDecision,
    RiskLevel,
    RunRecord,
    RunStatus,
    TaskPlan,
    TaskStep,
    ToolDefinition,
    ToolRequest,
    ToolResult,
)


class ContractTests(unittest.TestCase):
    def test_complete_run_record_round_trip(self) -> None:
        request = AgentRequest(goal="What does the local source say?")
        profile = AgentProfile(
            id="researcher",
            description="Grounded local research agent.",
            model="qwen2.5:3b",
            allowed_tools=["read_only_file_search"],
            permissions=["filesystem.search.readonly"],
            allowed_document_roots=[r"C:\AGI\11_Projects\OperatorResearchEngine\docs"],
        )
        step = TaskStep(
            description="Search the local corpus.", tool_name="read_only_file_search"
        )
        plan = TaskPlan(goal=request.goal, steps=[step])
        tool_request = ToolRequest(
            tool_name="read_only_file_search",
            arguments={"query": "local source"},
            step_id=step.step_id,
        )
        permission = PermissionDecision(
            request_id=tool_request.request_id,
            tool_name=tool_request.tool_name,
            allowed=True,
            reason="Profile grants read-only search.",
            permissions_checked=["filesystem.search.readonly"],
        )
        started = datetime.now(timezone.utc)
        evidence = EvidenceReference(
            source_id="fixture",
            source_path="docs/fixture.md",
            locator="L1-L2",
            excerpt="A grounded statement.",
            content_sha256="a" * 64,
        )
        result = ToolResult(
            request_id=tool_request.request_id,
            tool_name=tool_request.tool_name,
            success=True,
            output={"matches": 1},
            evidence=[evidence],
            started_at=started,
            completed_at=started + timedelta(milliseconds=5),
            duration_ms=5,
        )
        run_id = uuid4()
        evaluation = EvaluationResult(
            run_id=run_id,
            scores=EvaluationScores(
                task_completion=1,
                factual_grounding=1,
                plan_adherence=1,
                tool_correctness=1,
                unsupported_claims=1,
                output_usefulness=1,
            ),
            verdict=EvaluationVerdict.PASSED,
        )
        response = AgentResponse(
            request_id=request.request_id,
            run_id=run_id,
            status=RunStatus.COMPLETED,
            answer="The source contains a grounded statement.",
            evidence=[evidence],
            citations=[Citation(evidence_id=evidence.evidence_id)],
            evaluation=evaluation,
        )
        record = RunRecord(
            run_id=run_id,
            request=request,
            profile=profile,
            profile_hash="b" * 64,
            model_id=profile.model,
            plan=plan,
            permission_decisions=[permission],
            tool_requests=[tool_request],
            tool_results=[result],
            response=response,
            evaluation=evaluation,
            status=RunStatus.COMPLETED,
            started_at=started,
            completed_at=started + timedelta(milliseconds=10),
            duration_ms=10,
        )

        restored = RunRecord.model_validate_json(record.model_dump_json())
        self.assertEqual(restored, record)
        self.assertEqual(restored.request.goal, request.goal)

    def test_tool_definition_exposes_required_manifest(self) -> None:
        definition = ToolDefinition(
            name="calculator",
            description="Evaluate bounded arithmetic expressions.",
            input_schema={"type": "object"},
            output_schema={"type": "object"},
            risk_level=RiskLevel.LOW,
            permissions_required=["calculator.execute"],
            human_approval_required=False,
            timeout_seconds=2,
        )
        self.assertEqual(definition.timeout_seconds, 2)

    def test_model_contracts_support_structured_output(self) -> None:
        request = ModelRequest(
            model="qwen2.5:3b",
            messages=[ModelMessage(role=MessageRole.USER, content="Return JSON.")],
            response_schema={"type": "object"},
        )
        response = ModelResponse(
            request_id=request.request_id,
            model=request.model,
            structured_output={"answer": "ok"},
        )
        self.assertEqual(response.structured_output, {"answer": "ok"})

    def test_extra_fields_are_forbidden(self) -> None:
        with self.assertRaises(ValidationError):
            AgentRequest(goal="Valid goal", unexpected=True)

    def test_naive_timestamps_are_rejected(self) -> None:
        with self.assertRaises(ValidationError):
            AgentRequest(goal="Valid goal", created_at=datetime.now())

    def test_document_roots_must_be_local_absolute_paths(self) -> None:
        with self.assertRaises(ValidationError):
            AgentProfile(
                id="researcher",
                description="Researcher.",
                model="qwen2.5:3b",
                allowed_document_roots=["relative/docs"],
            )
        with self.assertRaises(ValidationError):
            AgentProfile(
                id="researcher",
                description="Researcher.",
                model="qwen2.5:3b",
                allowed_document_roots=[r"\\server\share"],
            )

    def test_invalid_timing_is_rejected(self) -> None:
        started = datetime.now(timezone.utc)
        with self.assertRaises(ValidationError):
            ToolResult(
                request_id=uuid4(),
                tool_name="calculator",
                success=False,
                error="Timed out.",
                started_at=started,
                completed_at=started - timedelta(milliseconds=1),
                duration_ms=1,
            )


if __name__ == "__main__":
    unittest.main()
