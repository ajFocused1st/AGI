from __future__ import annotations

import unittest

from agi_agent_runtime.tool_arguments import validate_tool_arguments
from agi_contracts import RiskLevel, TaskRequirements, ToolDefinition


def _search_definition() -> ToolDefinition:
    return ToolDefinition(
        name="read_only_file_search",
        description="Search approved documents",
        input_schema={
            "type": "object",
            "properties": {
                "query": {"type": "string", "minLength": 1, "maxLength": 200},
                "max_matches": {"type": "integer", "minimum": 1, "maximum": 100},
            },
            "required": ["query"],
            "additionalProperties": False,
        },
        output_schema={"type": "object"},
        risk_level=RiskLevel.LOW,
    )


class ToolArgumentTests(unittest.TestCase):
    def test_valid_search(self) -> None:
        result = validate_tool_arguments(
            _search_definition(),
            {"query": "Orion project facts", "max_matches": 5},
            TaskRequirements(task_subjects=["Orion"]),
        )
        self.assertTrue(result.dispatch_allowed)

    def test_missing_empty_and_whitespace_queries(self) -> None:
        for arguments in ({}, {"query": ""}, {"query": "   "}):
            with self.subTest(arguments=arguments):
                result = validate_tool_arguments(
                    _search_definition(), arguments, TaskRequirements()
                )
                self.assertFalse(result.valid)

    def test_unknown_argument_and_invalid_limit(self) -> None:
        unknown = validate_tool_arguments(
            _search_definition(), {"query": "Orion", "root": "C:\\"}, TaskRequirements()
        )
        limit = validate_tool_arguments(
            _search_definition(), {"query": "Orion", "max_matches": 101}, TaskRequirements()
        )
        self.assertEqual(unknown.unknown_arguments, ["root"])
        self.assertFalse(limit.valid)

    def test_sensitive_traversal_and_vague_queries(self) -> None:
        for query in ("../secrets", "C:\\credentials.json", "summarize the documents"):
            with self.subTest(query=query):
                result = validate_tool_arguments(
                    _search_definition(), {"query": query}, TaskRequirements()
                )
                self.assertFalse(result.valid)

    def test_required_subject_is_generalized(self) -> None:
        result = validate_tool_arguments(
            _search_definition(),
            {"query": "project facts"},
            TaskRequirements(task_subjects=["Orion"]),
        )
        self.assertIn("required task subject", result.semantic_errors[-1])


if __name__ == "__main__":
    unittest.main()
