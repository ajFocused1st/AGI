from __future__ import annotations

import unittest
from datetime import UTC, datetime
from uuid import uuid4

from agi_agent_runtime.requirements import (
    validate_plan_requirements,
    validate_response_requirements,
)
from agi_contracts import (
    Citation,
    EvidenceReference,
    RiskLevel,
    TaskPlan,
    TaskRequirements,
    TaskStep,
    ToolDefinition,
)
from pydantic import ValidationError


class RequirementValidatorTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tools = {
            "read_only_file_search": ToolDefinition(
                name="read_only_file_search",
                description="Search approved documents",
                input_schema={"type": "object"},
                output_schema={"type": "object"},
                risk_level=RiskLevel.LOW,
            ),
            "calculator": ToolDefinition(
                name="calculator",
                description="Calculate arithmetic",
                input_schema={"type": "object"},
                output_schema={"type": "object"},
                risk_level=RiskLevel.LOW,
            ),
        }

    def test_plan_satisfies_required_capability(self) -> None:
        result = validate_plan_requirements(
            TaskRequirements(required_capabilities=["read_only_file_search"]),
            _plan("read_only_file_search"),
            self.tools,
        )
        self.assertTrue(result.valid)

    def test_missing_required_capability(self) -> None:
        result = validate_plan_requirements(
            TaskRequirements(required_capabilities=["read_only_file_search"]),
            _plan(None),
            self.tools,
        )
        self.assertEqual(result.missing_capabilities, ["read_only_file_search"])

    def test_forbidden_and_unknown_capabilities(self) -> None:
        forbidden = validate_plan_requirements(
            TaskRequirements(forbidden_capabilities=["calculator"]),
            _plan("calculator"),
            self.tools,
        )
        unknown = validate_plan_requirements(
            TaskRequirements(), _plan("invented_tool"), self.tools
        )
        self.assertEqual(forbidden.forbidden_capabilities_proposed, ["calculator"])
        self.assertEqual(unknown.unknown_capabilities_proposed, ["invented_tool"])

    def test_contract_rejects_conflict_and_invalid_citation_minimum(self) -> None:
        with self.assertRaises(ValidationError):
            TaskRequirements(
                required_capabilities=["calculator"],
                forbidden_capabilities=["calculator"],
            )
        with self.assertRaises(ValidationError):
            TaskRequirements(requires_citations=True, minimum_citation_count=0)
        immutable = TaskRequirements()
        with self.assertRaises(ValidationError):
            immutable.requires_grounding = True

    def test_maximum_tool_calls(self) -> None:
        plan = TaskPlan(
            goal="calculate",
            steps=[
                TaskStep(description="one", tool_name="calculator"),
                TaskStep(description="two", tool_name="calculator"),
            ],
        )
        result = validate_plan_requirements(
            TaskRequirements(maximum_tool_calls=1), plan, self.tools
        )
        self.assertFalse(result.valid)
        self.assertIn("exceed maximum", result.validation_messages[0])

    def test_response_citation_resolution_and_count(self) -> None:
        evidence = _evidence()
        requirements = TaskRequirements(
            requires_grounding=True,
            requires_citations=True,
            minimum_citation_count=1,
        )
        valid = validate_response_requirements(
            requirements,
            answer="Supported answer.",
            evidence=[evidence],
            citations=[Citation(evidence_id=evidence.evidence_id)],
        )
        missing = validate_response_requirements(
            requirements, answer="Unsupported.", evidence=[evidence], citations=[]
        )
        unresolved = validate_response_requirements(
            requirements,
            answer="Bad citation.",
            evidence=[evidence],
            citations=[Citation(evidence_id=uuid4())],
        )
        self.assertTrue(valid.valid)
        self.assertFalse(missing.citation_requirement_satisfied)
        self.assertFalse(unresolved.citation_requirement_satisfied)

    def test_uncertainty_statement(self) -> None:
        requirements = TaskRequirements(requires_uncertainty_statement=True)
        present = validate_response_requirements(
            requirements,
            answer="The value cannot be verified from the available evidence.",
            evidence=[],
            citations=[],
        )
        missing = validate_response_requirements(
            requirements, answer="The value is probably ten.", evidence=[], citations=[]
        )
        self.assertTrue(present.valid)
        self.assertFalse(missing.valid)


def _plan(tool_name: str | None) -> TaskPlan:
    return TaskPlan(goal="goal", steps=[TaskStep(description="step", tool_name=tool_name)])


def _evidence() -> EvidenceReference:
    return EvidenceReference(
        source_id="documents:test",
        source_path="test.md",
        locator="line 1",
        excerpt="Fact",
        metadata={"observed_at": datetime.now(UTC).isoformat()},
    )


if __name__ == "__main__":
    unittest.main()
