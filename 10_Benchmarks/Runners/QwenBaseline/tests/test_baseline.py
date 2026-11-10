from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from agi_ollama import HttpResponse, OllamaTimeoutError  # noqa: E402
from agi_qwen_baseline.grading import grade, load_suite, reference_answer  # noqa: E402
from agi_qwen_baseline.runner import (  # noqa: E402
    DEFAULT_SUITE, MeasuredTransport, Options, main, run_cases, summarize, write_report,
)


class GradingTests(unittest.TestCase):
    def test_complete_suite_accepts_references_and_rejects_unrelated_answers(self) -> None:
        suite = load_suite(DEFAULT_SUITE)
        self.assertEqual(len(suite["cases"]), 40)
        for case in suite["cases"]:
            with self.subTest(case=case["id"]):
                self.assertTrue(grade(reference_answer(case["grading"]), case["grading"])[0])
                self.assertFalse(grade("not an answer", case["grading"])[0])
                self.assertFalse(grade("", case["grading"])[0])

    def test_number_grader_rejects_prose_and_uses_explicit_tolerance(self) -> None:
        rule = {"kind": "number", "expected": "219.30", "tolerance": "0.001"}
        self.assertTrue(grade("219.3001", rule)[0])
        for value in ["219.31", "NaN", "Infinity", "The result is 219.30", "$219.30", "219.30 or 200"]:
            self.assertFalse(grade(value, rule)[0])

    def test_json_grader_rejects_extra_keys_duplicate_keys_and_boolean_as_number(self) -> None:
        rule = {"kind": "json", "expected": {"count": 1}}
        self.assertTrue(grade('{"count":1}', rule)[0])
        for value in ['{"count":true}', '{"count":1,"extra":0}', '{"count":0,"count":1}',
                      '```json\n{"count":1}\n```', '{"count":NaN}']:
            self.assertFalse(grade(value, rule)[0])

    def test_planning_grader_accepts_multiple_valid_orders_and_rejects_missing_tasks(self) -> None:
        rule = next(case["grading"] for case in load_suite(DEFAULT_SUITE)["cases"]
                    if case["id"] == "planning-01")
        for value in ['["A","B","C","D"]', '["B","A","C","D"]']:
            self.assertTrue(grade(value, rule)[0])
        for value in ['["A","C","B","D"]', '["A","B","C"]', '["A","A","C","D"]']:
            self.assertFalse(grade(value, rule)[0])

    def test_options_reserve_input_context_and_enforce_budgets(self) -> None:
        Options().validate()
        for options in [Options(max_tokens=4096), Options(timeout_seconds=121),
                        Options(repetitions=0), Options(temperature=float("nan"))]:
            with self.assertRaises(ValueError):
                options.validate()

    def test_fixture_answers_have_independent_arithmetic_and_logic_checks(self) -> None:
        self.assertEqual(347 * 26, 9022)
        self.assertAlmostEqual(240 * .85 * 1.075, 219.3)
        from fractions import Fraction
        self.assertEqual(25 / (Fraction(7, 12) - Fraction(3, 8)), 120)
        self.assertEqual((12 * 2 + 18 * 3 + 30 * 5) / 10, 22.8)
        # Truth-table check of logic-03, independent of the model and its answer text.
        from itertools import product
        satisfying = [(a, b, c) for a, b, c in product([False, True], repeat=3)
                      if sum((a, b, c)) == 1 and a == (not b) and b == (not c)
                      and c == (not a and not b)]
        self.assertEqual(satisfying, [(False, True, False)])

    def test_interrupted_run_can_be_reported_without_replaying_requests(self) -> None:
        from dataclasses import asdict
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp)
            (directory / "definition.json").write_bytes(DEFAULT_SUITE.read_bytes())
            (directory / "manifest.json").write_text(json.dumps({
                "status": "running", "options": asdict(Options()),
            }), encoding="utf-8")
            self.assertEqual(main(["--recover-directory", str(directory)]), 0)
            summary = json.loads((directory / "summary.json").read_text())
            self.assertEqual(summary["attempted"], 0)
            self.assertEqual(summary["unattempted"], 40)
            manifest = json.loads((directory / "manifest.json").read_text())
            self.assertEqual(manifest["status"], "interrupted")


class FakeAdapter:
    def __init__(self, transport: SimpleNamespace, outputs: list[object]) -> None:
        self.transport = transport
        self.outputs = iter(outputs)
        self.requests = []

    async def generate(self, request: object) -> object:
        self.requests.append(request)
        self.transport.calls += 1
        self.transport.last_metrics = {"eval_count": 10, "prompt_eval_count": 20, "done_reason": "stop"}
        output = next(self.outputs)
        if isinstance(output, Exception):
            raise output
        if output == "TRUNCATED":
            self.transport.last_metrics["done_reason"] = "length"
            return SimpleNamespace(content="YES")
        return SimpleNamespace(content=output)


class RunnerTests(unittest.IsolatedAsyncioTestCase):
    async def test_transport_counts_attempts_but_retains_no_thinking_or_response_text(self) -> None:
        from unittest.mock import AsyncMock, patch
        from agi_ollama.transport import UrllibHttpTransport
        envelope = {"done_reason": "stop", "eval_count": 12, "prompt_eval_count": 20,
                    "message": {"content": "YES", "thinking": "private trace"}}
        transport = MeasuredTransport()
        with patch.object(UrllibHttpTransport, "post_json", new=AsyncMock(
                return_value=HttpResponse(200, json.dumps(envelope).encode(), {}))):
            await transport.post_json()
        self.assertEqual(transport.calls, 1)
        self.assertEqual(transport.last_metrics["eval_count"], 12)
        self.assertNotIn("message", transport.last_metrics)
        self.assertNotIn("private trace", json.dumps(transport.last_metrics))

    def small_suite(self) -> dict:
        return {"cases": [
            {"id": "one", "category": "logic", "prompt": "A question",
             "grading": {"kind": "choice", "expected": "YES"}},
            {"id": "two", "category": "logic", "prompt": "Another question",
             "grading": {"kind": "choice", "expected": "NO"}},
        ]}

    async def test_direct_requests_no_grading_leakage_and_incremental_artifacts(self) -> None:
        transport = SimpleNamespace(calls=0, last_metrics={})
        adapter = FakeAdapter(transport, ["YES", "NO"])
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)
            rows = await run_cases(suite=self.small_suite(), options=Options(), adapter=adapter,
                                   transport=transport, directory=path, progress=lambda _: None)
            self.assertEqual(len((path / "attempts.jsonl").read_text().splitlines()), 2)
            summary = summarize(rows, 2)
            self.assertEqual(summary["passed"], 2)
            self.assertEqual(summary["model_calls"], 2)
            write_report(path, {"options": {"model": "fake", "think": True,
                         "context_tokens": 4096, "max_tokens": 2048}, "status": "completed"}, rows, summary)
            self.assertEqual(json.loads((path / "summary.json").read_text())["attempted"], 2)
            self.assertTrue((path / "report.md").is_file())
        for request in adapter.requests:
            self.assertEqual(len(request.messages), 1)
            self.assertIsNone(request.response_schema)
            self.assertNotIn("grading", request.messages[0].content)
        self.assertNotIn("YES", adapter.requests[0].messages[0].content)

    async def test_truncated_correct_text_is_not_a_pass(self) -> None:
        transport = SimpleNamespace(calls=0, last_metrics={})
        adapter = FakeAdapter(transport, ["TRUNCATED", "NO"])
        with tempfile.TemporaryDirectory() as tmp:
            rows = await run_cases(suite=self.small_suite(), options=Options(), adapter=adapter,
                                   transport=transport, directory=Path(tmp), progress=lambda _: None)
        self.assertFalse(rows[0]["passed"])
        self.assertTrue(rows[0]["truncated"])
        self.assertEqual(summarize(rows, 2)["truncated"], 1)

    async def test_timeout_stops_collection_without_retry_or_inflated_coverage(self) -> None:
        transport = SimpleNamespace(calls=0, last_metrics={})
        adapter = FakeAdapter(transport, [OllamaTimeoutError("private provider output"), "NO"])
        with tempfile.TemporaryDirectory() as tmp:
            rows = await run_cases(suite=self.small_suite(), options=Options(), adapter=adapter,
                                   transport=transport, directory=Path(tmp), progress=lambda _: None)
        self.assertEqual(len(rows), 1)
        self.assertEqual(transport.calls, 1)
        self.assertEqual(summarize(rows, 2)["unattempted"], 1)
        self.assertNotIn("private provider output", json.dumps(rows))

    async def test_insufficient_time_starts_no_request(self) -> None:
        transport = SimpleNamespace(calls=0, last_metrics={})
        adapter = FakeAdapter(transport, [])
        with tempfile.TemporaryDirectory() as tmp:
            rows = await run_cases(suite=self.small_suite(), options=Options(total_seconds=1),
                                   adapter=adapter, transport=transport, directory=Path(tmp),
                                   progress=lambda _: None)
        self.assertEqual(rows, [])
        self.assertEqual(transport.calls, 0)

    async def test_inline_thinking_is_never_persisted(self) -> None:
        transport = SimpleNamespace(calls=0, last_metrics={})
        adapter = FakeAdapter(transport, ["<think>private trace</think>YES", "NO"])
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)
            rows = await run_cases(suite=self.small_suite(), options=Options(), adapter=adapter,
                                   transport=transport, directory=path, progress=lambda _: None)
            self.assertNotIn("private trace", (path / "attempts.jsonl").read_text())
        self.assertFalse(rows[0]["passed"])


if __name__ == "__main__":
    unittest.main()
