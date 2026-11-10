"""Frozen, output-specific graders. Never execute text supplied by the model."""

from __future__ import annotations

import json
import re
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Any

CATEGORIES = {
    "arithmetic", "logical_reasoning", "ambiguity_detection", "instruction_following",
    "coding", "planning", "contradiction_detection", "business_reasoning",
    "hallucination_resistance", "tool_selection",
}


def strict_json(text: str) -> Any:
    def reject_constant(value: str) -> None:
        raise ValueError(f"Non-finite JSON value: {value}")

    def unique_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError("Duplicate JSON key")
            result[key] = value
        return result

    return json.loads(text, parse_constant=reject_constant, object_pairs_hook=unique_keys)


def grade(answer: str, rule: dict[str, Any]) -> tuple[bool, str]:
    """Whitespace at the edges is ignored; explanatory additions are not accepted."""
    value = answer.strip()
    kind = rule["kind"]
    if not value:
        return False, "No final answer"
    if kind in {"exact", "choice"}:
        expected = rule["expected"]
        passed = value == expected if kind == "exact" else value.casefold() == expected.casefold()
        return passed, "Exact requested output" if passed else "Requested output did not match"
    if kind == "number":
        if not re.fullmatch(r"[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?", value):
            return False, "Expected one finite number without prose, currency symbols, or separators"
        try:
            difference = abs(Decimal(value) - Decimal(rule["expected"]))
            passed = difference <= Decimal(rule["tolerance"])
        except (InvalidOperation, OverflowError):
            return False, "Invalid numeric answer"
        return passed, "Within tolerance" if passed else "Numeric result outside tolerance"
    if kind in {"json", "precedence"}:
        try:
            parsed = strict_json(value)
        except (ValueError, RecursionError):
            return False, "Expected strict JSON without Markdown or prose"
        if kind == "json":
            # Canonical JSON distinguishes true from 1, which Python equality does not.
            passed = json.dumps(parsed, sort_keys=True) == json.dumps(rule["expected"], sort_keys=True)
            return passed, "Expected JSON value" if passed else "JSON value or type differs"
        if not isinstance(parsed, list) or any(not isinstance(item, str) for item in parsed):
            return False, "Expected an array of task names"
        if len(parsed) != len(rule["items"]) or set(parsed) != set(rule["items"]):
            return False, "Tasks must occur exactly once"
        passed = all(parsed.index(a) < parsed.index(b) for a, b in rule["before"])
        return passed, "All dependency constraints satisfied" if passed else "Dependency violated"
    raise ValueError(f"Unknown grader: {kind}")


def reference_answer(rule: dict[str, Any]) -> str:
    if rule["kind"] == "precedence":
        return json.dumps(rule["reference"])
    if rule["kind"] == "json":
        return json.dumps(rule["expected"])
    return str(rule["expected"])


def load_suite(path: Path) -> dict[str, Any]:
    suite = strict_json(path.read_text(encoding="utf-8"))
    if not isinstance(suite, dict) or suite.get("benchmark_id") != "qwen-baseline-v1":
        raise ValueError("Unexpected baseline definition")
    cases = suite.get("cases", [])
    if len(cases) != 40 or len({case["id"] for case in cases}) != 40:
        raise ValueError("Baseline requires 40 uniquely identified cases")
    if {case["category"] for case in cases} != CATEGORIES:
        raise ValueError("Baseline category coverage differs")
    for category in CATEGORIES:
        if sum(case["category"] == category for case in cases) != 4:
            raise ValueError("Each category must contain four cases")
    for case in cases:
        if not isinstance(case["prompt"], str) or not 1 <= len(case["prompt"]) <= 4000:
            raise ValueError("Each prompt must contain 1-4000 characters")
        rule = case["grading"]
        if rule["kind"] == "number":
            expected, tolerance = Decimal(rule["expected"]), Decimal(rule["tolerance"])
            if not expected.is_finite() or not tolerance.is_finite() or tolerance < 0:
                raise ValueError("Invalid numeric grading rule")
        if not grade(reference_answer(rule), rule)[0]:
            raise ValueError(f"Invalid grading reference: {case['id']}")
    return suite
