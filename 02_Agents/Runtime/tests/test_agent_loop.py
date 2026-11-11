"""Bounded observe/follow-up loop (correction plan, step 10)."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from agi_agent_runtime import AgentRuntime, AtomicJsonRunStore
from agi_contracts import AgentProfile, AgentRequest, RunRecord, TaskRequirements
from agi_evaluation import RuleBasedEvaluator
from agi_ollama import ScriptedModelAdapter, ScriptedReply
from agi_tools import (
    ReadOnlyFileReader,
    ReadOnlyFileReaderConfig,
    ReadOnlyFileSearch,
    ReadOnlyFileSearchConfig,
    SearchRoot,
    ToolRegistry,
)

SEARCH = "read_only_file_search"
READER = "read_only_file_reader"


def reply(payload: dict) -> ScriptedReply:
    return ScriptedReply(content="", structured_output=payload)


def plan(*calls: dict) -> ScriptedReply:
    return reply({"steps": [{"description": "Research"}], "tool_calls": list(calls)})


class AgentLoopTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self) -> None:
        self._documents = tempfile.TemporaryDirectory()
        self._logs = tempfile.TemporaryDirectory()
        root = Path(self._documents.name)
        (root / "kestrel.md").write_text("# Kestrel Pump\n\n[K1] Kestrel moves 40 litres per minute.\n", encoding="utf-8")
        (root / "osprey.md").write_text("# Osprey Pump\n\n[O1] Osprey moves 55 litres per minute.\n", encoding="utf-8")
        roots = (SearchRoot(label="documents1", path=root),)
        self.registry = ToolRegistry()
        self.registry.register(ReadOnlyFileSearch(ReadOnlyFileSearchConfig(roots=roots)))
        self.registry.register(ReadOnlyFileReader(ReadOnlyFileReaderConfig(roots=roots)))
        self.profile = AgentProfile(
            id="researcher",
            description="loop probe",
            model="scripted",
            allowed_tools=[SEARCH, READER],
            permissions=["filesystem.search.readonly", "filesystem.read.readonly"],
            allowed_document_roots=[str(root)],
            max_steps=4,
            max_tool_calls=6,
        )

    def tearDown(self) -> None:
        self._documents.cleanup()
        self._logs.cleanup()

    async def run_case(self, replies: list[ScriptedReply], requirements: TaskRequirements, goal: str):
        adapter = ScriptedModelAdapter(replies)
        runtime = AgentRuntime(
            model_adapter=adapter,
            tool_executor=self.registry,
            evaluator=RuleBasedEvaluator(),
            run_store=AtomicJsonRunStore(Path(self._logs.name) / "logs"),
            enable_model_critique=False,
        )
        response = await runtime.run(
            AgentRequest(goal=goal, agent_profile_id="researcher"), self.profile, requirements=requirements
        )
        record = RunRecord.model_validate_json(
            (Path(self._logs.name) / "logs" / str(response.run_id) / "run.json").read_text(encoding="utf-8")
        )
        return record, [request.metadata["phase"] for request in adapter.requests]

    async def test_missing_subject_triggers_one_follow_up_search(self) -> None:
        requirements = TaskRequirements(
            required_capabilities=[SEARCH], requires_grounding=True, task_subjects=["Kestrel", "Osprey"]
        )
        record, phases = await self.run_case(
            [
                plan({"tool_name": SEARCH, "arguments": {"query": "Kestrel flow rate"}}),
                reply({"tool_calls": [{"tool_name": SEARCH, "arguments": {"query": "Osprey flow rate"}}]}),
                reply({"answer": "Kestrel moves 40 and Osprey 55 litres per minute."}),
            ],
            requirements,
            "Compare the Kestrel and Osprey pump flow rates.",
        )
        self.assertEqual(phases, ["plan", "observe", "answer"])
        self.assertEqual(
            [request.arguments["query"] for request in record.tool_requests],
            ["Kestrel flow rate", "Osprey flow rate"],
        )
        excerpts = " ".join(item.excerpt for item in record.evidence_bundle.items)
        self.assertIn("[K1]", excerpts)
        self.assertIn("[O1]", excerpts)
        self.assertEqual(record.metadata["follow_up_rounds"], 1)

    async def test_no_gaps_means_no_observe_call(self) -> None:
        requirements = TaskRequirements(required_capabilities=[SEARCH], requires_grounding=True, task_subjects=["Kestrel"])
        record, phases = await self.run_case(
            [
                plan({"tool_name": SEARCH, "arguments": {"query": "Kestrel flow rate"}}),
                reply({"answer": "Kestrel moves 40 litres per minute."}),
            ],
            requirements,
            "What is the Kestrel pump flow rate?",
        )
        self.assertEqual(phases, ["plan", "answer"])
        self.assertNotIn("follow_up_rounds", record.metadata)

    async def test_malformed_reader_guess_is_never_dispatched(self) -> None:
        requirements = TaskRequirements(required_capabilities=[SEARCH], requires_grounding=True)
        record, phases = await self.run_case(
            [
                plan(
                    {"tool_name": SEARCH, "arguments": {"query": "Kestrel flow rate"}},
                    {"tool_name": READER, "arguments": {"root_label": "documents1", "relative_path": "/guess.md"}},
                ),
                reply({"answer": "Kestrel moves 40 litres per minute."}),
            ],
            requirements,
            "What is the Kestrel pump flow rate?",
        )
        self.assertEqual([request.tool_name for request in record.tool_requests], [SEARCH])
        self.assertTrue(all(result.success for result in record.tool_results))
        reader_history = [item for item in record.tool_argument_histories if item.tool_name == READER]
        self.assertFalse(reader_history[0].validation_result.valid)
        self.assertEqual(phases, ["plan", "answer"])

    async def test_follow_up_reader_uses_real_paths_and_forbidden_proposals_are_denied(self) -> None:
        requirements = TaskRequirements(
            required_capabilities=[SEARCH],
            requires_grounding=True,
            task_subjects=["Kestrel", "Osprey"],
            forbidden_capabilities=["delete_file"],
        )
        record, phases = await self.run_case(
            [
                plan({"tool_name": SEARCH, "arguments": {"query": "Kestrel flow rate"}}),
                reply(
                    {
                        "tool_calls": [
                            {"tool_name": READER, "arguments": {"root_label": "documents1", "relative_path": "osprey.md"}},
                            {"tool_name": "delete_file", "arguments": {"path": "kestrel.md"}},
                        ]
                    }
                ),
                reply({"answer": "Kestrel moves 40 and Osprey 55 litres per minute."}),
            ],
            requirements,
            "Compare the Kestrel and Osprey pump flow rates.",
        )
        self.assertEqual(phases, ["plan", "observe", "answer"])
        dispatched = [result.tool_name for result in record.tool_results if result.success]
        self.assertEqual(dispatched, [SEARCH, READER])
        denied = [decision.tool_name for decision in record.permission_decisions if not decision.allowed]
        self.assertIn("delete_file", denied)
        self.assertIn("[O1]", " ".join(item.excerpt for item in record.evidence_bundle.items))


class CalculatorAfterEvidenceTests(AgentLoopTests):
    async def test_calculation_operands_come_from_evidence_in_follow_up(self) -> None:
        from agi_tools import Calculator

        self.registry.register(Calculator())
        self.profile = self.profile.model_copy(
            update={
                "allowed_tools": [SEARCH, READER, "calculator"],
                "permissions": [*self.profile.permissions, "calculator.execute"],
            }
        )
        requirements = TaskRequirements(
            required_capabilities=[SEARCH, "calculator"],
            requires_grounding=True,
            task_subjects=["Kestrel", "Osprey"],
        )
        record, phases = await self.run_case(
            [
                # The planner cannot write the expression before seeing the numbers.
                plan({"tool_name": SEARCH, "arguments": {"query": "Kestrel Osprey litres per minute"}}),
                plan({"tool_name": SEARCH, "arguments": {"query": "Kestrel Osprey litres per minute"}}),
                reply({"tool_calls": [{"tool_name": "calculator", "arguments": {"expression": "55 - 40"}}]}),
                reply({"answer": "Osprey moves 15 more litres per minute than Kestrel."}),
            ],
            requirements,
            "By how many litres per minute does Osprey outpace Kestrel?",
        )
        self.assertEqual(phases, ["plan", "plan_correction", "observe", "answer"])
        self.assertEqual(
            record.metadata["host_actions"],
            [{"action": "deferred_required_capability", "capabilities": ["calculator"]}],
        )
        calculator = [result for result in record.tool_results if result.tool_name == "calculator"]
        self.assertTrue(calculator and calculator[0].success)
        self.assertEqual(calculator[0].output["value"], 15)


if __name__ == "__main__":
    unittest.main()
