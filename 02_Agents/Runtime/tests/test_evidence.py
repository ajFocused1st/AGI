from __future__ import annotations

import unittest
from datetime import UTC, datetime

from agi_agent_runtime.evidence import build_evidence_bundle
from agi_contracts import ToolRequest, ToolResult


class EvidenceBundleTests(unittest.TestCase):
    def test_host_ids_hashes_and_change_detection(self) -> None:
        request = ToolRequest(
            tool_name="read_only_file_search", arguments={"query": "Orion"}
        )
        first = build_evidence_bundle([request], [_result(request, "Fact one")])
        repeated = build_evidence_bundle([request], [_result(request, "Fact one")])
        changed = build_evidence_bundle([request], [_result(request, "Fact two")])
        self.assertEqual(first.items[0].source_hash, repeated.items[0].source_hash)
        self.assertNotEqual(first.items[0].evidence_id, repeated.items[0].evidence_id)
        self.assertEqual(first.bundle_hash, repeated.bundle_hash)
        self.assertNotEqual(first.bundle_hash, changed.bundle_hash)

    def test_calculator_result_becomes_tool_evidence(self) -> None:
        request = ToolRequest(
            tool_name="calculator", arguments={"expression": "37 * 14"}
        )
        now = datetime.now(UTC)
        bundle = build_evidence_bundle(
            [request],
            [
                ToolResult(
                    request_id=request.request_id,
                    tool_name=request.tool_name,
                    success=True,
                    output={"expression": "37 * 14", "value": 518},
                    started_at=now,
                    completed_at=now,
                    duration_ms=0,
                )
            ],
        )

        self.assertEqual(bundle.evidence_count, 1)
        self.assertEqual(bundle.items[0].source_id, "tool:calculator")
        self.assertEqual(bundle.items[0].source_type, "tool_result")
        self.assertEqual(bundle.items[0].excerpt, "37 * 14 = 518")


def _result(request: ToolRequest, snippet: str) -> ToolResult:
    now = datetime.now(UTC)
    return ToolResult(
        request_id=request.request_id,
        tool_name=request.tool_name,
        success=True,
        output={
            "matches": [
                {
                    "source_id": "documents:orion.md",
                    "relative_path": "orion.md",
                    "content_sha256": "a" * 64,
                    "line_start": 1,
                    "line_end": 1,
                    "snippet": snippet,
                }
            ]
        },
        started_at=now,
        completed_at=now,
        duration_ms=0,
    )


if __name__ == "__main__":
    unittest.main()
