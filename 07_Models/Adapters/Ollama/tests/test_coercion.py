"""Schema-driven coercion replaying real qwen3.5:9b outputs from live checkpoint A."""

from __future__ import annotations

import json
import unittest
from typing import Annotated, Any

from agi_ollama.adapter import parse_structured_output
from agi_ollama.errors import OllamaResponseValidationError
from pydantic import BaseModel, ConfigDict, Field

Short = Annotated[str, Field(min_length=1, max_length=200)]


class Step(BaseModel):
    model_config = ConfigDict(extra="forbid")
    step_id: str | int | None = None
    description: Short
    success_criteria: Short | None = None


class Call(BaseModel):
    model_config = ConfigDict(extra="forbid")
    tool_name: Short | None = None
    arguments: dict[str, Any] = Field(default_factory=dict)
    purpose: Short | None = None
    related_step_id: str | int | None = None


class Plan(BaseModel):
    model_config = ConfigDict(extra="forbid")
    plan_id: Any | None = None
    goal: Short | None = None
    steps: list[Step] = Field(min_length=1)
    tool_calls: list[Call]


SCHEMA = Plan.model_json_schema()


class CoercionTests(unittest.TestCase):
    def test_empty_optional_purpose_becomes_null(self) -> None:
        raw = (
            '{"plan_id":null,"goal":"Use the calculator to compute 48 multiplied by 23.","steps":'
            '[{"step_id":1,"description":"Compute the product."}],"tool_calls":[{"related_step_id":1,'
            '"purpose":"","tool_name":"calculator","arguments":{"expression":"48 * 23"}}]}'
        )
        parsed, changes = parse_structured_output(raw, SCHEMA)
        self.assertIsNone(parsed["tool_calls"][0]["purpose"])
        self.assertEqual(parsed["tool_calls"][0]["arguments"], {"expression": "48 * 23"})
        self.assertEqual(changes, ["empty_to_null:tool_calls[0].purpose"])

    def test_echoed_prompt_key_is_dropped(self) -> None:
        raw = (
            '{"plan_id":null,"goal":"Delete cedar.md.","steps":[{"step_id":1,"description":'
            '"Attempt to delete the file."}],"tool_calls":[],"correction":{"missing_capabilities":'
            '["delete_file"]}}'
        )
        parsed, changes = parse_structured_output(raw, SCHEMA)
        self.assertNotIn("correction", parsed)
        self.assertEqual(changes, ["dropped_key:correction"])

    def test_garbage_key_is_dropped(self) -> None:
        raw = (
            '{"plan_id":"x","goal":"Find facts.","steps":[{"step_id":1,"description":"Search."}],'
            '"tool_calls":[],"}]` is invalid JSON. Corrected: `{":"plan_id"}'
        )
        parsed, changes = parse_structured_output(raw, SCHEMA)
        self.assertEqual(set(parsed), {"plan_id", "goal", "steps", "tool_calls"})
        self.assertEqual(len(changes), 1)

    def test_valid_output_is_untouched(self) -> None:
        raw = json.dumps({"steps": [{"description": "Plan"}], "tool_calls": []})
        parsed, changes = parse_structured_output(raw, SCHEMA)
        self.assertEqual(changes, [])
        self.assertEqual(parsed, json.loads(raw))

    def test_tool_arguments_keep_unknown_keys_for_host_validation(self) -> None:
        raw = json.dumps(
            {"steps": [{"description": "S"}], "tool_calls": [{"tool_name": "t", "arguments": {"q": "x", "path": "../.env"}}]}
        )
        parsed, changes = parse_structured_output(raw, SCHEMA)
        self.assertEqual(parsed["tool_calls"][0]["arguments"], {"q": "x", "path": "../.env"})
        self.assertEqual(changes, [])

    def test_empty_required_field_still_fails(self) -> None:
        raw = json.dumps({"steps": [{"description": ""}], "tool_calls": []})
        with self.assertRaises(OllamaResponseValidationError):
            parse_structured_output(raw, SCHEMA)

    def test_missing_required_field_still_fails(self) -> None:
        raw = json.dumps({"steps": [{"description": "S"}], "extra": 1})
        with self.assertRaises(OllamaResponseValidationError):
            parse_structured_output(raw, SCHEMA)

    def test_non_json_still_fails(self) -> None:
        with self.assertRaises(OllamaResponseValidationError):
            parse_structured_output("Here is the plan: {", SCHEMA)


if __name__ == "__main__":
    unittest.main()
