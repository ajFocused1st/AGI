from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from agi_agent_runtime import AgentRuntime, AtomicJsonRunStore
from agi_contracts import (
    AgentProfile,
    AgentRequest,
    ModelRequest,
    ModelResponse,
    ModelUsage,
    RunRecord,
)
from agi_evaluation import EvaluationExpectations, RuleBasedEvaluator
from agi_tools import Calculator, ToolRegistry


class PhaseAdapter:
    def __init__(self, replies: dict[str, dict[str, object]]) -> None:
        self.replies = replies
        self.requests: list[ModelRequest] = []

    async def generate(self, request: ModelRequest) -> ModelResponse:
        self.requests.append(request)
        phase = str(request.metadata["phase"])
        structured = self.replies[phase]
        return ModelResponse(
            request_id=request.request_id,
            model=request.model,
            content=json.dumps(structured),
            structured_output=structured,
            usage=ModelUsage(input_tokens=10, output_tokens=5, total_tokens=15),
        )


class AgentRuntimeTests(unittest.IsolatedAsyncioTestCase):
    async def test_calculator_loop_saves_round_trippable_record(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            registry = ToolRegistry()
            registry.register(Calculator())
            adapter = PhaseAdapter(
                {
                    "plan": {
                        "goal": "Calculate 37 * 14",
                        "steps": [
                            {
                                "description": "Use the approved calculator",
                                "tool_name": "calculator",
                                "tool_arguments": {"expression": "37 * 14"},
                            }
                        ],
                    },
                    "answer": {
                        "answer": "37 multiplied by 14 is 518.",
                        "cited_evidence_ids": [],
                        "insufficient_evidence": False,
                    },
                    "critique": {
                        "critique": "The answer follows the calculator result.",
                        "suggestions": [],
                    },
                }
            )
            runtime = AgentRuntime(
                model_adapter=adapter,
                tool_executor=registry,
                evaluator=RuleBasedEvaluator(),
                run_store=AtomicJsonRunStore(Path(temporary) / "logs"),
            )
            profile = AgentProfile(
                id="researcher",
                description="Test profile",
                model="scripted",
                allowed_tools=["calculator"],
                permissions=["calculator.execute"],
            )
            response = await runtime.run(
                AgentRequest(goal="Calculate 37 * 14"),
                profile,
                expectations=EvaluationExpectations(
                    required_answer_substrings=("518",),
                    expected_tools=("calculator",),
                    expected_tool_call_count=1,
                ),
                known_unsupported_claims=[],
            )

            run_path = Path(temporary) / "logs" / str(response.run_id) / "run.json"
            record = RunRecord.model_validate_json(run_path.read_text(encoding="utf-8"))
            self.assertEqual(record.response.answer, response.answer)
            self.assertEqual(record.tool_results[0].output["value"], 518)
            self.assertEqual(record.evaluation.verdict.value, "passed")
            self.assertEqual(record.usage.total_tokens, 45)

    async def test_unknown_destructive_tool_is_denied_and_logged(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            registry = ToolRegistry()
            registry.register(Calculator())
            adapter = PhaseAdapter(
                {
                    "plan": {
                        "goal": "Delete atlas.md",
                        "steps": [
                            {
                                "description": "Attempt deletion",
                                "tool_name": "delete_file",
                                "tool_arguments": {"path": "atlas.md"},
                            }
                        ],
                    },
                    "answer": {
                        "answer": "I cannot perform the unapproved destructive action.",
                        "cited_evidence_ids": [],
                        "insufficient_evidence": False,
                    },
                    "critique": {
                        "critique": "The refusal follows the permission decision.",
                        "suggestions": [],
                    },
                }
            )
            runtime = AgentRuntime(
                model_adapter=adapter,
                tool_executor=registry,
                evaluator=RuleBasedEvaluator(),
                run_store=AtomicJsonRunStore(Path(temporary) / "logs"),
            )
            profile = AgentProfile(
                id="researcher",
                description="Test profile",
                model="scripted",
                allowed_tools=["calculator"],
                permissions=["calculator.execute"],
            )
            response = await runtime.run(
                AgentRequest(goal="Delete atlas.md"),
                profile,
                expectations=EvaluationExpectations(expect_denial=True),
                known_unsupported_claims=[],
            )
            record_path = Path(temporary) / "logs" / str(response.run_id) / "run.json"
            record = RunRecord.model_validate_json(record_path.read_text(encoding="utf-8"))
            self.assertFalse(record.permission_decisions[0].allowed)
            self.assertFalse(record.tool_results[0].success)
            self.assertIn("denied", record.tool_results[0].error)

    def test_store_rejects_source_workspace_location(self) -> None:
        with self.assertRaises(ValueError):
            AtomicJsonRunStore("C:\\AGI\\logs", forbidden_root="C:\\AGI")


if __name__ == "__main__":
    unittest.main()
