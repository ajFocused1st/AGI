from __future__ import annotations

import unittest

from agi_agent_runtime.prompt_payloads import (
    compact_evidence,
    compact_requirements,
    compact_tool_definitions,
    compact_tool_outcomes,
)
from agi_contracts import EvidenceReference, TaskRequirements, ToolResult
from agi_tools import Calculator, ToolRegistry


def _evidence(query: str, index: int) -> EvidenceReference:
    return EvidenceReference(
        source_id="documents1:a.md",
        source_path="a.md",
        locator=f"lines {index}-{index}",
        excerpt=f"{query} fact {index}",
        metadata={"retrieval_tool": "read_only_file_search", "retrieval_query": query},
    )


class PromptPayloadTests(unittest.TestCase):
    def test_later_search_is_not_crowded_out_by_the_cap(self) -> None:
        evidence = [_evidence("Atlas", i) for i in range(1, 21)] + [
            _evidence("Beacon", i) for i in range(1, 4)
        ]
        compact = compact_evidence(evidence, limit=8)
        self.assertEqual(len(compact), 8)
        self.assertEqual(sum(item["excerpt"].startswith("Beacon") for item in compact), 3)
        self.assertEqual(set(compact[0]), {"evidence_id", "source", "excerpt"})

    def test_requirements_keep_only_non_default_fields(self) -> None:
        self.assertEqual(compact_requirements(TaskRequirements()), {})
        compact = compact_requirements(
            TaskRequirements(requires_citations=True, minimum_citation_count=2)
        )
        self.assertEqual(compact, {"requires_citations": True, "minimum_citation_count": 2})

    def test_tool_definitions_omit_host_only_metadata(self) -> None:
        registry = ToolRegistry()
        registry.register(Calculator())
        (definition,) = compact_tool_definitions(registry.tools.values())
        self.assertEqual(set(definition), {"name", "description", "arguments", "required"})
        self.assertIn("expression", definition["arguments"])

    def test_tool_outcomes_report_failures_and_empty_searches(self) -> None:
        from datetime import UTC, datetime
        from uuid import uuid4

        now = datetime.now(UTC)
        outcomes = compact_tool_outcomes(
            [
                ToolResult(request_id=uuid4(), tool_name="read_only_file_search", success=True,
                           output={"matches": []}, started_at=now, completed_at=now, duration_ms=0),
                ToolResult(request_id=uuid4(), tool_name="delete_file", success=False, error="tool request denied",
                           started_at=now, completed_at=now, duration_ms=0),
            ]
        )
        self.assertEqual(outcomes[0], {"tool": "read_only_file_search", "ok": True, "match_count": 0})
        self.assertEqual(outcomes[1]["error"], "tool request denied")


if __name__ == "__main__":
    unittest.main()
