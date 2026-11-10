from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from agi_agent_runtime import (
    AgentRuntime,
    AtomicJsonRunStore,
    BudgetPolicy,
    ReasoningTier,
    RoutingPolicy,
)
from agi_agent_runtime.runtime import _model_telemetry
from agi_contracts import (
    AgentProfile,
    AgentRequest,
    ModelRequest,
    ModelResponse,
    ModelUsage,
    RiskLevel,
    RunRecord,
    TaskRequirements,
)
from agi_evaluation import EvaluationExpectations, RuleBasedEvaluator
from agi_tools import BaseTool, Calculator, ToolRegistry
from pydantic import BaseModel, ConfigDict, Field


class PhaseAdapter:
    def __init__(self, replies: dict[str, dict[str, object]]) -> None:
        self.replies = replies
        self.requests: list[ModelRequest] = []

    async def generate(self, request: ModelRequest) -> ModelResponse:
        self.requests.append(request)
        phase = str(request.metadata["phase"])
        structured = self.replies[phase]
        if phase in {"plan", "plan_correction"}:
            structured = _separate_test_tool_calls(structured)
        return ModelResponse(
            request_id=request.request_id,
            model=request.model,
            content=json.dumps(structured),
            structured_output=structured,
            usage=ModelUsage(input_tokens=10, output_tokens=5, total_tokens=15),
        )


class CalculatorCitationAdapter:
    def __init__(
        self, *, answer_value: int = 518, correction_value: int | None = None
    ) -> None:
        self.requests: list[ModelRequest] = []
        self.answer_value = answer_value
        self.correction_value = correction_value

    async def generate(self, request: ModelRequest) -> ModelResponse:
        self.requests.append(request)
        phase = str(request.metadata["phase"])
        if phase == "plan":
            structured = _separate_test_tool_calls(
                {
                    "steps": [
                        {
                            "description": "Calculate with the approved calculator",
                            "tool_name": "calculator",
                            "tool_arguments": {"expression": "37 * 14"},
                        }
                    ]
                }
            )
        elif phase == "answer":
            payload = json.loads(request.messages[-1].content)
            evidence_id = payload["evidence"][0]["evidence_id"]
            structured = {
                "answer": f"37 multiplied by 14 is {self.answer_value}.",
                "claims": [
                    {
                        "text": f"37 multiplied by 14 is {self.answer_value}.",
                        "evidence_ids": [evidence_id],
                    }
                ],
                "cited_evidence_ids": [evidence_id],
            }
        elif phase == "answer_correction":
            payload = json.loads(request.messages[-1].content)
            evidence_id = payload["evidence"][0]["evidence_id"]
            value = self.correction_value if self.correction_value is not None else self.answer_value
            structured = {
                "answer": f"37 multiplied by 14 is {value}.",
                "claims": [
                    {
                        "text": f"37 multiplied by 14 is {value}.",
                        "evidence_ids": [evidence_id],
                    }
                ],
                "cited_evidence_ids": [evidence_id],
            }
        elif phase == "critique":
            structured = {
                "critique": "The calculator evidence supports the answer.",
                "suggestions": [],
            }
        else:
            raise AssertionError(f"unexpected phase: {phase}")
        return ModelResponse(
            request_id=request.request_id,
            model=request.model,
            content=json.dumps(structured),
            structured_output=structured,
            usage=ModelUsage(input_tokens=10, output_tokens=5, total_tokens=15),
        )


class SearchInput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    query: str = Field(min_length=1)
    limit: int = Field(default=5, ge=1, le=10)


class SearchOutput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    matches: list[dict[str, object]]


class InMemorySearch(BaseTool[SearchInput, SearchOutput]):
    name = "read_only_file_search"
    description = "Search test documents."
    risk_level = RiskLevel.LOW
    permissions_required = ("filesystem.search.readonly",)
    input_model = SearchInput
    output_model = SearchOutput

    async def run(self, tool_input: SearchInput) -> SearchOutput:
        return SearchOutput(
            matches=[
                {
                    "source_id": "documents:atlas.md",
                    "relative_path": "atlas.md",
                    "content_sha256": "b" * 64,
                    "line_start": 6,
                    "line_end": 6,
                    "snippet": "[A4] Mina Chen owns the Atlas review.",
                }
            ]
        )


class CompareSearch(BaseTool[SearchInput, SearchOutput]):
    name = "read_only_file_search"
    description = "Search comparison fixture documents."
    risk_level = RiskLevel.LOW
    permissions_required = ("filesystem.search.readonly",)
    input_model = SearchInput
    output_model = SearchOutput

    async def run(self, tool_input: SearchInput) -> SearchOutput:
        return SearchOutput(
            matches=[
                {
                    "source_id": "documents:atlas.md",
                    "relative_path": "atlas.md",
                    "content_sha256": "a" * 64,
                    "line_start": 4,
                    "line_end": 4,
                    "snippet": "[A2] The Atlas trial processed 120 samples.",
                },
                {
                    "source_id": "documents:atlas.md",
                    "relative_path": "atlas.md",
                    "content_sha256": "a" * 64,
                    "line_start": 5,
                    "line_end": 5,
                    "snippet": "[A3] The trial completed in 4 hours.",
                },
                {
                    "source_id": "documents:beacon.md",
                    "relative_path": "beacon.md",
                    "content_sha256": "c" * 64,
                    "line_start": 3,
                    "line_end": 3,
                    "snippet": "[B1] The Beacon trial processed 90 samples.",
                },
                {
                    "source_id": "documents:beacon.md",
                    "relative_path": "beacon.md",
                    "content_sha256": "c" * 64,
                    "line_start": 4,
                    "line_end": 4,
                    "snippet": "[B2] The trial completed in 3 hours.",
                },
            ]
        )


class SearchCitationAdapter:
    def __init__(
        self, *, owner: str = "Mina Chen", correction_owner: str | None = None
    ) -> None:
        self.requests: list[ModelRequest] = []
        self.owner = owner
        self.correction_owner = correction_owner

    async def generate(self, request: ModelRequest) -> ModelResponse:
        self.requests.append(request)
        phase = str(request.metadata["phase"])
        if phase == "plan":
            structured = _separate_test_tool_calls(
                {
                    "steps": [
                        {
                            "description": "Search approved documents",
                            "tool_name": "read_only_file_search",
                            "tool_arguments": {"query": "Atlas owner", "limit": 1},
                        }
                    ]
                }
            )
        elif phase == "answer":
            payload = json.loads(request.messages[-1].content)
            evidence_id = payload["evidence"][0]["evidence_id"]
            text = f"{self.owner} owns the Atlas review."
            structured = {
                "answer": text,
                "claims": [{"text": text, "evidence_ids": [evidence_id]}],
                "cited_evidence_ids": [evidence_id],
            }
        elif phase == "answer_correction":
            payload = json.loads(request.messages[-1].content)
            evidence_id = payload["evidence"][0]["evidence_id"]
            owner = self.correction_owner if self.correction_owner is not None else self.owner
            text = f"{owner} owns the Atlas review."
            structured = {
                "answer": text,
                "claims": [{"text": text, "evidence_ids": [evidence_id]}],
                "cited_evidence_ids": [evidence_id],
            }
        elif phase == "critique":
            structured = {
                "critique": "The document evidence was cited.",
                "suggestions": [],
            }
        else:
            raise AssertionError(f"unexpected phase: {phase}")
        return ModelResponse(
            request_id=request.request_id,
            model=request.model,
            content=json.dumps(structured),
            structured_output=structured,
            usage=ModelUsage(input_tokens=10, output_tokens=5, total_tokens=15),
        )


class AgentRuntimeTests(unittest.IsolatedAsyncioTestCase):
    def test_model_telemetry_preserves_redacted_first_error(self) -> None:
        telemetry = _model_telemetry(
            "answer",
            {
                "structured_output": {
                    "first_attempt_valid": False,
                    "repair_attempted": True,
                    "repair_succeeded": True,
                    "final_valid": True,
                    "attempt_count": 2,
                    "first_error": {
                        "category": "schema_validation",
                        "path": "<root>",
                    },
                }
            },
        )

        self.assertEqual(
            telemetry,
            {
                "phase": "answer",
                "first_attempt_valid": False,
                "repair_attempted": True,
                "repair_succeeded": True,
                "final_valid": True,
                "attempt_count": 2,
                "first_error": {
                    "category": "schema_validation",
                    "path": "<root>",
                },
            },
        )

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
                    "answer": {
                        "answer": "The required tool call was blocked.",
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
                AgentRequest(goal="Calculate"),
                profile,
                requirements=TaskRequirements(required_capabilities=["calculator"]),
            )
            record = RunRecord.model_validate_json(
                (Path(temporary) / "logs" / str(response.run_id) / "run.json").read_text()
            )
            self.assertFalse(record.plan_requirement_result.valid)
            self.assertTrue(record.plan_requirement_result.correction_attempted)
            self.assertIn("tool_dispatch_blocked", [error.code for error in record.errors])
            self.assertEqual(len(adapter.requests), 3)

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

    async def test_uncertainty_requirement_is_explicit_in_answer_prompt(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            registry = ToolRegistry()
            adapter = PhaseAdapter(
                {
                    "plan": {"steps": [{"description": "Answer carefully"}]},
                    "answer": {"answer": "The available evidence does not verify this."},
                }
            )
            runtime = AgentRuntime(
                model_adapter=adapter,
                tool_executor=registry,
                evaluator=RuleBasedEvaluator(),
                run_store=AtomicJsonRunStore(Path(temporary) / "logs"),
                enable_model_critique=False,
            )
            profile = AgentProfile(id="researcher", description="Test", model="scripted")

            response = await runtime.run(
                AgentRequest(goal="Report an unknown value"),
                profile,
                requirements=TaskRequirements(requires_uncertainty_statement=True),
            )

            record = RunRecord.model_validate_json(
                (Path(temporary) / "logs" / str(response.run_id) / "run.json").read_text()
            )
            self.assertTrue(record.response_requirement_result.valid)
            answer_request = next(
                request for request in adapter.requests if request.metadata["phase"] == "answer"
            )
            self.assertIn(
                "requires_uncertainty_statement=true",
                answer_request.messages[0].content,
            )

    async def test_failed_answer_correction_uses_evidence_fallback(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            registry = ToolRegistry()
            registry.register(InMemorySearch())
            adapter = PhaseAdapter(
                {
                    "plan": {
                        "steps": [
                            {
                                "description": "Search documents",
                                "tool_name": "read_only_file_search",
                                "tool_arguments": {"query": "Atlas owner"},
                            }
                        ]
                    },
                    "answer": {
                        "answer": "I found evidence but will not cite it.",
                        "cited_evidence_ids": [],
                    },
                    "answer_correction": {
                        "answer": None,
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
                allowed_tools=["read_only_file_search"],
                permissions=["filesystem.search.readonly"],
            )

            response = await runtime.run(
                AgentRequest(goal="Find facts about Atlas."),
                profile,
                requirements=TaskRequirements(
                    required_capabilities=["read_only_file_search"],
                    requires_grounding=True,
                    requires_citations=True,
                    minimum_citation_count=1,
                ),
            )

            record = RunRecord.model_validate_json(
                (Path(temporary) / "logs" / str(response.run_id) / "run.json").read_text()
            )
            self.assertTrue(record.response_requirement_result.valid)
            self.assertGreaterEqual(len(record.response.citations), 1)
            self.assertEqual(record.evaluation.verdict.value, "passed")

    async def test_fallback_prefers_non_heading_evidence(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            registry = ToolRegistry()
            registry.register(InMemorySearch())
            adapter = PhaseAdapter(
                {
                    "plan": {
                        "steps": [
                            {
                                "description": "Search documents",
                                "tool_name": "read_only_file_search",
                                "tool_arguments": {"query": "Atlas owner"},
                            }
                        ]
                    },
                    "answer": {"answer": "No citations.", "cited_evidence_ids": []},
                    "answer_correction": {"answer": None, "cited_evidence_ids": []},
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
                allowed_tools=["read_only_file_search"],
                permissions=["filesystem.search.readonly"],
            )

            response = await runtime.run(
                AgentRequest(goal="Find Atlas facts."),
                profile,
                requirements=TaskRequirements(
                    required_capabilities=["read_only_file_search"],
                    requires_grounding=True,
                    requires_citations=True,
                    minimum_citation_count=1,
                ),
            )

            record = RunRecord.model_validate_json(
                (Path(temporary) / "logs" / str(response.run_id) / "run.json").read_text()
            )
            self.assertNotIn("#", record.response.answer)

    async def test_compare_fallback_includes_multiple_evidence_facts(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            registry = ToolRegistry()
            registry.register(CompareSearch())
            adapter = PhaseAdapter(
                {
                    "plan": {
                        "steps": [
                            {
                                "description": "Search comparison facts",
                                "tool_name": "read_only_file_search",
                                "tool_arguments": {"query": "Atlas Beacon comparison"},
                            }
                        ]
                    },
                    "answer": {"answer": "No citations.", "cited_evidence_ids": []},
                    "answer_correction": {"answer": None, "cited_evidence_ids": []},
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
                allowed_tools=["read_only_file_search"],
                permissions=["filesystem.search.readonly"],
            )

            response = await runtime.run(
                AgentRequest(goal="Compare Atlas and Beacon sample counts and completion times."),
                profile,
                requirements=TaskRequirements(
                    required_capabilities=["read_only_file_search"],
                    requires_grounding=True,
                    requires_citations=True,
                    minimum_citation_count=2,
                ),
            )

            record = RunRecord.model_validate_json(
                (Path(temporary) / "logs" / str(response.run_id) / "run.json").read_text()
            )
            # The fallback is a labelled safety net ranked by generic goal overlap,
            # not tuned to reproduce every benchmark fact.
            self.assertEqual(record.answer_source.value, "host_fallback")
            self.assertGreaterEqual(len(record.response.citations), 2)
            self.assertIn("120", record.response.answer)
            self.assertIn("90", record.response.answer)

    async def test_invalid_optional_tool_call_is_dropped_without_repair(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            registry = ToolRegistry()
            registry.register(Calculator())
            adapter = PhaseAdapter(
                {
                    "plan": {
                        "steps": [
                            {
                                "description": "Do not calculate",
                                "tool_name": "calculator",
                                "tool_arguments": {"expression": "estimated revenue"},
                            }
                        ]
                    },
                    "answer": {
                        "answer": "No calculation was needed.",
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
                AgentRequest(goal="Create a plan without calculations."),
                profile,
                known_unsupported_claims=[],
            )

            record = RunRecord.model_validate_json(
                (Path(temporary) / "logs" / str(response.run_id) / "run.json").read_text()
            )
            self.assertEqual(record.tool_requests, [])
            self.assertEqual(
                [request.metadata["phase"] for request in adapter.requests],
                ["plan", "answer"],
            )

    async def test_ungrounded_task_claims_do_not_force_unsupported_failure(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            registry = ToolRegistry()
            adapter = PhaseAdapter(
                {
                    "plan": {"steps": [{"description": "Create a plan"}]},
                    "answer": {
                        "claims": [
                            {
                                "text": "The plan has three steps.",
                                "evidence_ids": [],
                            }
                        ],
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
            profile = AgentProfile(id="researcher", description="Test", model="scripted")

            response = await runtime.run(
                AgentRequest(goal="Create a plan."),
                profile,
                requirements=TaskRequirements(requires_grounding=False),
            )

            record = RunRecord.model_validate_json(
                (Path(temporary) / "logs" / str(response.run_id) / "run.json").read_text()
            )
            self.assertTrue(record.response_requirement_result.valid)
            self.assertEqual(record.evaluation.verdict.value, "passed")

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
            self.assertEqual(record.metadata["budget"]["tier"], "standard")
            self.assertEqual(record.metadata["budget"]["model_attempts"], 3)
            self.assertEqual(record.metadata["budget"]["tool_attempts"], 1)

    async def test_null_tool_call_placeholders_are_ignored(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            registry = ToolRegistry()
            registry.register(Calculator())
            adapter = PhaseAdapter(
                {
                    "plan": {
                        "goal": "Calculate and cite evidence",
                        "steps": [
                            {"description": "Calculate 37 * 14"},
                            {"description": "Cite the calculator evidence"},
                        ],
                        "tool_calls": [
                            {
                                "tool_name": "calculator",
                                "arguments": {"expression": "37 * 14"},
                            },
                            {
                                "tool_name": None,
                                "arguments": {},
                                "purpose": "Cite the calculator output as evidence.",
                            },
                        ],
                    },
                    "answer": {
                        "answer": "37 multiplied by 14 is 518.",
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
                description="Test profile",
                model="scripted",
                allowed_tools=["calculator"],
                permissions=["calculator.execute"],
            )

            response = await runtime.run(
                AgentRequest(goal="Use the calculator to compute 37 * 14."),
                profile,
                requirements=TaskRequirements(required_capabilities=["calculator"]),
                known_unsupported_claims=[],
            )

            record = RunRecord.model_validate_json(
                (Path(temporary) / "logs" / str(response.run_id) / "run.json").read_text()
            )
            self.assertEqual([request.tool_name for request in record.tool_requests], ["calculator"])
            self.assertEqual(record.evaluation.verdict.value, "passed")

    async def test_calculator_result_is_citable_answer_evidence(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            registry = ToolRegistry()
            registry.register(Calculator())
            adapter = CalculatorCitationAdapter()
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
                requirements=TaskRequirements(
                    required_capabilities=["calculator"],
                    requires_grounding=True,
                    requires_citations=True,
                    minimum_citation_count=1,
                ),
            )

            record = RunRecord.model_validate_json(
                (Path(temporary) / "logs" / str(response.run_id) / "run.json").read_text()
            )
            self.assertEqual(record.evidence_bundle.evidence_count, 1)
            self.assertEqual(record.evidence_bundle.items[0].source_id, "tool:calculator")
            self.assertEqual(record.evidence_bundle.items[0].excerpt, "37 * 14 = 518")
            self.assertEqual(len(record.response.citations), 1)
            self.assertTrue(record.response_requirement_result.valid)
            self.assertEqual(record.evaluation.verdict.value, "passed")

    async def test_calculator_citation_fails_when_answer_conflicts_with_result(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            registry = ToolRegistry()
            registry.register(Calculator())
            adapter = CalculatorCitationAdapter(answer_value=519)
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
                requirements=TaskRequirements(
                    required_capabilities=["calculator"],
                    requires_grounding=True,
                    requires_citations=True,
                    minimum_citation_count=1,
                ),
            )

            record = RunRecord.model_validate_json(
                (Path(temporary) / "logs" / str(response.run_id) / "run.json").read_text()
            )
            self.assertEqual(record.evidence_bundle.items[0].excerpt, "37 * 14 = 518")
            self.assertFalse(record.response_requirement_result.valid)
            self.assertTrue(record.response_requirement_result.correction_attempted)
            self.assertEqual(record.evaluation.verdict.value, "failed")
            self.assertFalse(
                next(
                    check
                    for check in record.evaluation.rule_based_checks
                    if check.name == "calculator_evidence_consistency"
                ).passed
            )

    async def test_calculator_support_failure_triggers_one_answer_repair(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            registry = ToolRegistry()
            registry.register(Calculator())
            adapter = CalculatorCitationAdapter(answer_value=519, correction_value=518)
            runtime = AgentRuntime(
                model_adapter=adapter,
                tool_executor=registry,
                evaluator=RuleBasedEvaluator(),
                run_store=AtomicJsonRunStore(Path(temporary) / "logs"),
                enable_model_critique=False,
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
                requirements=TaskRequirements(
                    required_capabilities=["calculator"],
                    requires_grounding=True,
                    requires_citations=True,
                    minimum_citation_count=1,
                ),
            )

            record = RunRecord.model_validate_json(
                (Path(temporary) / "logs" / str(response.run_id) / "run.json").read_text()
            )
            self.assertEqual(
                [request.metadata["phase"] for request in adapter.requests],
                ["plan", "answer", "answer_correction"],
            )
            self.assertTrue(record.response_requirement_result.valid)
            self.assertTrue(record.response_requirement_result.correction_attempted)
            self.assertEqual(record.evaluation.verdict.value, "passed")
            self.assertIn("518", record.response.answer)

    async def test_document_citation_fails_when_answer_invents_entity(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            registry = ToolRegistry()
            registry.register(InMemorySearch())
            adapter = SearchCitationAdapter(owner="Nora Patel")
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
                allowed_tools=["read_only_file_search"],
                permissions=["filesystem.search.readonly"],
            )

            response = await runtime.run(
                AgentRequest(goal="Who owns the Atlas review?"),
                profile,
                requirements=TaskRequirements(
                    required_capabilities=["read_only_file_search"],
                    requires_grounding=True,
                    requires_citations=True,
                    minimum_citation_count=1,
                ),
            )

            record = RunRecord.model_validate_json(
                (Path(temporary) / "logs" / str(response.run_id) / "run.json").read_text()
            )
            self.assertEqual(record.evidence_bundle.items[0].excerpt, "[A4] Mina Chen owns the Atlas review.")
            self.assertEqual(record.evaluation.verdict.value, "failed")
            self.assertFalse(
                next(
                    check
                    for check in record.evaluation.rule_based_checks
                    if check.name == "document_evidence_support"
                ).passed
            )

    async def test_document_support_failure_triggers_one_answer_repair(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            registry = ToolRegistry()
            registry.register(InMemorySearch())
            adapter = SearchCitationAdapter(owner="Nora Patel", correction_owner="Mina Chen")
            runtime = AgentRuntime(
                model_adapter=adapter,
                tool_executor=registry,
                evaluator=RuleBasedEvaluator(),
                run_store=AtomicJsonRunStore(Path(temporary) / "logs"),
                enable_model_critique=False,
            )
            profile = AgentProfile(
                id="researcher",
                description="Test profile",
                model="scripted",
                allowed_tools=["read_only_file_search"],
                permissions=["filesystem.search.readonly"],
            )

            response = await runtime.run(
                AgentRequest(goal="Who owns the Atlas review?"),
                profile,
                requirements=TaskRequirements(
                    required_capabilities=["read_only_file_search"],
                    requires_grounding=True,
                    requires_citations=True,
                    minimum_citation_count=1,
                ),
            )

            record = RunRecord.model_validate_json(
                (Path(temporary) / "logs" / str(response.run_id) / "run.json").read_text()
            )
            self.assertEqual(
                [request.metadata["phase"] for request in adapter.requests],
                ["plan", "answer", "answer_correction"],
            )
            self.assertTrue(record.response_requirement_result.valid)
            self.assertTrue(record.response_requirement_result.correction_attempted)
            self.assertEqual(record.evaluation.verdict.value, "passed")
            self.assertIn("Mina Chen", record.response.answer)

    async def test_model_budget_failure_is_persisted(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            registry = ToolRegistry()
            adapter = PhaseAdapter(
                {
                    "plan": {
                        "goal": "Create a plan",
                        "steps": [{"description": "Answer directly"}],
                    },
                    "answer": {
                        "answer": "This should not be reached.",
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
                routing_policy=RoutingPolicy(
                    tier=ReasoningTier.STANDARD,
                    budget=BudgetPolicy(
                        max_model_calls=1,
                        max_tool_calls=0,
                        max_input_tokens=4096,
                        max_output_tokens=512,
                        max_total_tokens=4608,
                        max_wallclock_seconds=20,
                    ),
                    allow_tools=False,
                    allow_retrieval=False,
                    allow_critique=False,
                    allow_revision=False,
                    native_thinking=False,
                    description="Test budget with one model call.",
                ),
            )
            profile = AgentProfile(id="researcher", description="Test", model="scripted")

            with self.assertRaises(Exception) as raised:
                await runtime.run(AgentRequest(goal="Create a plan"), profile)

            self.assertIn("run budget exhausted", str(raised.exception))
            self.assertEqual([r.metadata["phase"] for r in adapter.requests], ["plan"])
            run_id = raised.exception.run_id
            record = RunRecord.model_validate_json(
                (Path(temporary) / "logs" / str(run_id) / "run.json").read_text()
            )
            self.assertEqual(record.errors[-1].code, "budget_exceeded")
            self.assertEqual(record.metadata["budget"]["model_attempts"], 1)
            self.assertFalse(record.metadata["budget"]["events"][-1]["accepted"])

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
            self.assertIn("denied by host policy", record.response.answer)
            self.assertIn("delete_file", record.response.answer)

    def test_store_rejects_source_workspace_location(self) -> None:
        with self.assertRaises(ValueError):
            AtomicJsonRunStore("C:\\AGI\\logs", forbidden_root="C:\\AGI")


if __name__ == "__main__":
    unittest.main()


def _separate_test_tool_calls(value: dict[str, object]) -> dict[str, object]:
    normalized = dict(value)
    calls: list[dict[str, object]] = [
        dict(call)
        for call in value.get("tool_calls", [])
        if isinstance(call, dict)
    ]
    steps: list[dict[str, object]] = []
    for raw in value.get("steps", []):
        step = dict(raw)
        tool_name = step.pop("tool_name", None)
        arguments = step.pop("tool_arguments", {})
        if tool_name is not None:
            calls.append(
                {
                    "tool_name": tool_name,
                    "arguments": arguments,
                    "purpose": step.get("description"),
                    "related_step_id": step.get("step_id"),
                }
            )
        steps.append(step)
    normalized["steps"] = steps
    normalized["tool_calls"] = calls
    return normalized

