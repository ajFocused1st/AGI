from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from agi_agent_runtime import AgentRuntime, AtomicJsonRunStore, RuntimeExecutionError
from agi_contracts import (
    AgentProfile,
    AgentRequest,
    ModelRequest,
    ModelResponse,
    ModelUsage,
    RunRecord,
    TaskRequirements,
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
    async def test_one_plan_correction_adds_required_capability_and_persists(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            registry = ToolRegistry()
            registry.register(Calculator())
            adapter = PhaseAdapter(
                {
                    "plan": {"steps": [{"description": "Answer directly"}]},
                    "plan_correction": {
                        "steps": [
                            {
                                "description": "Calculate",
                                "tool_name": "calculator",
                                "tool_arguments": {"expression": "2 + 2"},
                            }
                        ]
                    },
                    "answer": {
                        "answer": "The result is 4.",
                        "cited_evidence_ids": [],
                    },
                }
            )
            runtime = AgentRuntime(
                model_adapter=adapter,
                tool_executor=registry,
                evaluator=RuleBasedEvaluator(),
                run_store=AtomicJsonRunStore(Path(temporary) / "logs"),
                enable_model_critique=False,
            )
            profile = AgentProfile(
                id="researcher",
                description="Test",
                model="scripted",
                allowed_tools=["calculator"],
                permissions=["calculator.execute"],
            )
            response = await runtime.run(
                AgentRequest(goal="Calculate 2 + 2"),
                profile,
                requirements=TaskRequirements(required_capabilities=["calculator"]),
            )
            record = RunRecord.model_validate_json(
                (Path(temporary) / "logs" / str(response.run_id) / "run.json").read_text()
            )
            self.assertTrue(record.plan_requirement_result.correction_attempted)
            self.assertTrue(record.plan_requirement_result.correction_succeeded)
            self.assertEqual(record.task_requirements.required_capabilities, ["calculator"])
            self.assertEqual([r.metadata["phase"] for r in adapter.requests], ["plan", "plan_correction", "answer"])

    async def test_plan_correction_is_bounded_and_failed_result_persists(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            registry = ToolRegistry()
            registry.register(Calculator())
            adapter = PhaseAdapter(
                {
                    "plan": {"steps": [{"description": "No tool"}]},
                    "plan_correction": {"steps": [{"description": "Still no tool"}]},
                }
            )
            runtime = AgentRuntime(
                model_adapter=adapter,
                tool_executor=registry,
                evaluator=RuleBasedEvaluator(),
                run_store=AtomicJsonRunStore(Path(temporary) / "logs"),
                enable_model_critique=False,
            )
            profile = AgentProfile(
                id="researcher",
                description="Test",
                model="scripted",
                allowed_tools=["calculator"],
                permissions=["calculator.execute"],
            )
            with self.assertRaises(RuntimeExecutionError) as caught:
                await runtime.run(
                    AgentRequest(goal="Calculate"),
                    profile,
                    requirements=TaskRequirements(required_capabilities=["calculator"]),
                )
            record = RunRecord.model_validate_json(
                (Path(temporary) / "logs" / str(caught.exception.run_id) / "run.json").read_text()
            )
            self.assertFalse(record.plan_requirement_result.valid)
            self.assertTrue(record.plan_requirement_result.correction_attempted)
            self.assertEqual(len(adapter.requests), 2)

    async def test_answer_correction_is_bounded_and_controls_verdict(self) -> None:
        for corrected_answer, expected_valid in (
            ("The value cannot be verified.", True),
            ("The value is probably ten.", False),
        ):
            with self.subTest(expected_valid=expected_valid), tempfile.TemporaryDirectory() as temporary:
                registry = ToolRegistry()
                adapter = PhaseAdapter(
                    {
                        "plan": {"steps": [{"description": "Answer carefully"}]},
                        "answer": {
                            "answer": "The value is ten.",
                            "cited_evidence_ids": [],
                        },
                        "answer_correction": {
                            "answer": corrected_answer,
                            "cited_evidence_ids": [],
                        },
                    }
                )
                runtime = AgentRuntime(
                    model_adapter=adapter,
                    tool_executor=registry,
                    evaluator=RuleBasedEvaluator(),
                    run_store=AtomicJsonRunStore(Path(temporary) / "logs"),
                    enable_model_critique=False,
                )
                profile = AgentProfile(
                    id="researcher", description="Test", model="scripted"
                )
                response = await runtime.run(
                    AgentRequest(goal="Report an unknown value"),
                    profile,
                    requirements=TaskRequirements(requires_uncertainty_statement=True),
                )
                record = RunRecord.model_validate_json(
                    (Path(temporary) / "logs" / str(response.run_id) / "run.json").read_text()
                )
                self.assertEqual(record.response_requirement_result.valid, expected_valid)
                self.assertTrue(record.response_requirement_result.correction_attempted)
                self.assertEqual(
                    record.evaluation.verdict.value,
                    "passed" if expected_valid else "failed",
                )
                self.assertEqual(
                    [r.metadata["phase"] for r in adapter.requests],
                    ["plan", "answer", "answer_correction"],
                )

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
            self.assertNotEqual(record.plan.steps[0].step_id, "1")
            self.assertEqual(record.tool_results[0].output["value"], 518)
            self.assertEqual(record.evaluation.verdict.value, "passed")
            self.assertEqual(record.usage.total_tokens, 45)

    async def test_runtime_assigns_plan_step_ids(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            registry = ToolRegistry()
            adapter = PhaseAdapter(
                {
                    "plan": {
                        "goal": "Create a plan",
                        "steps": [
                            {
                                "step_id": "1",
                                "description": "Clarify the research question",
                            }
                        ],
                    },
                    "answer": {
                        "answer": "The plan has one host-owned step.",
                        "cited_evidence_ids": [],
                        "insufficient_evidence": False,
                    },
                    "critique": {
                        "critique": "The answer matches the simple plan.",
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
                allowed_tools=[],
                permissions=[],
            )

            response = await runtime.run(AgentRequest(goal="Create a plan"), profile)

            record_path = Path(temporary) / "logs" / str(response.run_id) / "run.json"
            record = RunRecord.model_validate_json(record_path.read_text(encoding="utf-8"))
            self.assertNotEqual(str(record.plan.steps[0].step_id), "1")
            self.assertEqual(record.plan.steps[0].description, "Clarify the research question")

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
                    "plan_correction": {
                        "goal": "Delete atlas.md",
                        "steps": [{"description": "Report that deletion is unavailable"}],
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
                requirements=TaskRequirements(forbidden_capabilities=["delete_file"]),
            )
            record_path = Path(temporary) / "logs" / str(response.run_id) / "run.json"
            record = RunRecord.model_validate_json(record_path.read_text(encoding="utf-8"))
            self.assertFalse(record.permission_decisions[0].allowed)
            self.assertFalse(record.tool_results[0].success)
            self.assertIn("denied", record.tool_results[0].error)

    async def test_goal_level_delete_request_is_denied_even_without_model_tool(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            registry = ToolRegistry()
            registry.register(Calculator())
            adapter = PhaseAdapter(
                {
                    "plan": {
                        "goal": "Summarize documents and delete originals",
                        "steps": [
                            {
                                "description": "Summarize documents",
                                "tool_name": "calculator",
                                "tool_arguments": {
                                    "expression": "Summarize the benchmark documents"
                                },
                            },
                            {
                                "description": "Delete originals",
                            },
                        ],
                    },
                    "answer": {
                        "answer": "The documents were summarized.",
                        "cited_evidence_ids": [],
                        "insufficient_evidence": False,
                    },
                    "critique": {
                        "critique": "The destructive request was denied by the host.",
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
                AgentRequest(goal="Summarize documents and then delete the originals"),
                profile,
                requirements=TaskRequirements(forbidden_capabilities=["delete_file"]),
            )

            record_path = Path(temporary) / "logs" / str(response.run_id) / "run.json"
            record = RunRecord.model_validate_json(record_path.read_text(encoding="utf-8"))
            self.assertEqual(record.tool_requests[0].tool_name, "delete_file")
            self.assertEqual(len(record.tool_requests), 1)
            self.assertFalse(record.permission_decisions[0].allowed)
            self.assertIn("task requirements forbid", record.permission_decisions[0].reason)
            self.assertIn("Deletion was denied", record.response.answer)

    def test_store_rejects_source_workspace_location(self) -> None:
        with self.assertRaises(ValueError):
            AtomicJsonRunStore("C:\\AGI\\logs", forbidden_root="C:\\AGI")


if __name__ == "__main__":
    unittest.main()
