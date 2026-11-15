"""Tool calls a small model misspells are still performed, and lost ones are recorded.

Also covers the host's search on the user's own question, which runs beside the plan.
"""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from agi_agent_runtime import AgentRuntime, AtomicJsonRunStore
from agi_agent_runtime.drafts import FollowUpDraft, PlanDraft, ToolCallDraft
from agi_contracts import AgentProfile, AgentRequest, RunRecord, TaskRequirements
from agi_evaluation import RuleBasedEvaluator
from agi_ollama import ScriptedModelAdapter, ScriptedReply
from agi_tools import (
    Calculator,
    ReadOnlyFileSearch,
    ReadOnlyFileSearchConfig,
    SearchRoot,
    ToolRegistry,
)

SEARCH = "read_only_file_search"
CALCULATOR = "calculator"
# Reaches only the Kestrel document, so the Osprey fact needs a follow-up search.
RIVAL_GOAL = "How does Kestrel measure up against its rival?"
KESTREL_GOAL = "What is the Kestrel pump flow rate?"


def reply(payload: dict) -> ScriptedReply:
    return ScriptedReply(content="", structured_output=payload)


def plan(*calls: dict) -> ScriptedReply:
    return reply({"steps": [{"description": "Research"}], "tool_calls": list(calls)})


class ToolCallDraftTests(unittest.TestCase):
    def test_name_and_tool_are_read_as_the_tool_name(self) -> None:
        for key in ("name", "tool"):
            with self.subTest(key=key):
                draft = ToolCallDraft.model_validate({key: SEARCH, "arguments": {"query": "x"}})
                self.assertEqual(draft.tool_name, SEARCH)
                self.assertEqual(draft.arguments, {"query": "x"})
                self.assertFalse(draft.model_extra)

    def test_an_explicit_tool_name_wins_over_an_alias(self) -> None:
        draft = ToolCallDraft.model_validate({"tool_name": SEARCH, "name": "something_else"})
        self.assertEqual(draft.tool_name, SEARCH)

    def test_the_text_null_means_no_tool(self) -> None:
        for spelled in ("null", "NULL", " none ", ""):
            with self.subTest(spelled=spelled):
                draft = ToolCallDraft.model_validate({"tool_name": spelled})
                self.assertIsNone(draft.tool_name)
                self.assertTrue(draft.is_empty())

    def test_a_null_alias_does_not_hide_arguments(self) -> None:
        draft = ToolCallDraft.model_validate({"name": "null", "arguments": {"query": "x"}})
        self.assertIsNone(draft.tool_name)
        self.assertFalse(draft.is_empty())

    def test_missing_arguments_become_an_empty_object(self) -> None:
        self.assertEqual(ToolCallDraft.model_validate({"tool_name": SEARCH, "arguments": None}).arguments, {})

    def test_arguments_written_beside_the_envelope_are_kept(self) -> None:
        draft = ToolCallDraft.model_validate({"tool_name": SEARCH, "query": "Kestrel flow rate"})
        self.assertEqual(draft.model_extra, {"query": "Kestrel flow rate"})

    def test_schemas_let_the_adapter_keep_misspelled_keys(self) -> None:
        # The Ollama adapter deletes unknown keys wherever a schema forbids them.
        for model in (PlanDraft, FollowUpDraft):
            with self.subTest(model=model.__name__):
                schema = model.model_json_schema()
                self.assertIsNot(schema["$defs"]["ToolCallDraft"].get("additionalProperties"), False)


class TolerantRuntimeTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self) -> None:
        self._documents = tempfile.TemporaryDirectory()
        self._logs = tempfile.TemporaryDirectory()
        root = Path(self._documents.name)
        (root / "kestrel.md").write_text(
            "# Kestrel Pump\n\n[K1] Kestrel moves 40 litres per minute.\n", encoding="utf-8"
        )
        (root / "osprey.md").write_text(
            "# Osprey Pump\n\n[O1] Osprey moves 55 litres per minute.\n", encoding="utf-8"
        )
        self.registry = ToolRegistry()
        self.registry.register(
            ReadOnlyFileSearch(ReadOnlyFileSearchConfig(roots=(SearchRoot(label="documents1", path=root),)))
        )
        self.registry.register(Calculator())
        self.profile = AgentProfile(
            id="researcher",
            description="tolerant tool call probe",
            model="scripted",
            allowed_tools=[SEARCH, CALCULATOR],
            permissions=["filesystem.search.readonly", "calculator.execute"],
            allowed_document_roots=[str(root)],
            max_steps=4,
            max_tool_calls=6,
        )
        self.search_required = TaskRequirements(required_capabilities=[SEARCH], requires_grounding=True)

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
        return record, adapter

    @staticmethod
    def queries(record: RunRecord) -> list[str]:
        return [request.arguments["query"] for request in record.tool_requests if request.tool_name == SEARCH]

    # --- misspelled tool calls ---------------------------------------------------

    async def test_plan_call_spelled_with_name_and_loose_arguments_is_performed(self) -> None:
        record, _ = await self.run_case(
            [
                plan({"name": SEARCH, "query": "Kestrel litres per minute", "note": "not an argument"}),
                reply({"answer": "Kestrel moves 40 litres per minute."}),
            ],
            self.search_required,
            KESTREL_GOAL,
        )
        self.assertIn("Kestrel litres per minute", self.queries(record))
        searched = [r for r in record.tool_requests if r.arguments.get("query") == "Kestrel litres per minute"]
        # Only keys the tool declares are moved into its arguments.
        self.assertEqual(searched[0].arguments, {"query": "Kestrel litres per minute"})
        self.assertTrue(all(result.success for result in record.tool_results))

    async def test_follow_up_call_spelled_with_tool_is_performed(self) -> None:
        requirements = self.search_required.model_copy(update={"task_subjects": ["Kestrel", "Osprey"]})
        record, _ = await self.run_case(
            [
                plan({"tool_name": SEARCH, "arguments": {"query": "Kestrel flow rate"}}),
                reply({"tool_calls": [{"tool": SEARCH, "query": "Osprey litres per minute"}]}),
                reply({"answer": "Kestrel moves 40 and Osprey 55 litres per minute."}),
            ],
            requirements,
            RIVAL_GOAL,
        )
        self.assertEqual(self.queries(record)[-1], "Osprey litres per minute")
        self.assertIn("[O1]", " ".join(item.excerpt for item in record.evidence_bundle.items))
        self.assertNotIn("follow_up_call_unusable", [error.code for error in record.errors])

    async def test_follow_up_call_naming_no_tool_is_recorded_as_an_error(self) -> None:
        requirements = self.search_required.model_copy(update={"task_subjects": ["Kestrel", "Osprey"]})
        record, _ = await self.run_case(
            [
                plan({"tool_name": SEARCH, "arguments": {"query": "Kestrel flow rate"}}),
                reply({"tool_calls": [{"tool_name": "null", "arguments": {"query": "Osprey litres per minute"}}]}),
                reply({"answer": "Kestrel moves 40 litres per minute."}),
            ],
            requirements,
            RIVAL_GOAL,
        )
        self.assertNotIn("Osprey litres per minute", self.queries(record))
        lost = [error for error in record.errors if error.code == "follow_up_call_unusable"]
        self.assertEqual(len(lost), 1)
        self.assertEqual(lost[0].details, {"count": 1})

    async def test_an_empty_follow_up_entry_is_not_an_error(self) -> None:
        requirements = self.search_required.model_copy(update={"task_subjects": ["Kestrel", "Osprey"]})
        record, _ = await self.run_case(
            [
                plan({"tool_name": SEARCH, "arguments": {"query": "Kestrel flow rate"}}),
                reply({"tool_calls": [{"tool_name": "null", "arguments": {}}]}),
                reply({"answer": "Kestrel moves 40 litres per minute."}),
            ],
            requirements,
            RIVAL_GOAL,
        )
        self.assertNotIn("follow_up_call_unusable", [error.code for error in record.errors])

    async def test_prompts_show_the_key_the_model_must_write_back(self) -> None:
        requirements = self.search_required.model_copy(update={"task_subjects": ["Kestrel", "Osprey"]})
        _, adapter = await self.run_case(
            [
                plan({"tool_name": SEARCH, "arguments": {"query": "Kestrel flow rate"}}),
                reply({"tool_calls": []}),
                reply({"answer": "Kestrel moves 40 litres per minute."}),
            ],
            requirements,
            RIVAL_GOAL,
        )
        observe = next(item for item in adapter.requests if item.metadata["phase"] == "observe")
        self.assertIn('"tool_name"', observe.messages[0].content)
        user_payload = observe.messages[1].content
        self.assertIn(f'"tool_name": "{SEARCH}"', user_payload)
        self.assertNotIn('"tool":', user_payload)
        self.assertNotIn('"name":', user_payload)

    # --- the host's search on the user's own question -----------------------------

    async def test_question_is_searched_first_and_labelled_as_a_host_action(self) -> None:
        record, _ = await self.run_case(
            [
                plan({"tool_name": SEARCH, "arguments": {"query": "Kestrel litres"}}),
                reply({"answer": "Kestrel moves 40 litres per minute."}),
            ],
            self.search_required,
            KESTREL_GOAL,
        )
        self.assertEqual(self.queries(record), [KESTREL_GOAL, "Kestrel litres"])
        self.assertIn(
            {"action": "host_question_search", "tool_name": SEARCH}, record.metadata["host_actions"]
        )

    async def test_a_passage_found_by_both_searches_is_one_evidence_item(self) -> None:
        record, _ = await self.run_case(
            [
                plan({"tool_name": SEARCH, "arguments": {"query": "Kestrel litres"}}),
                reply({"answer": "Kestrel moves 40 litres per minute."}),
            ],
            self.search_required,
            KESTREL_GOAL,
        )
        excerpts = [item.excerpt for item in record.evidence_bundle.items]
        self.assertEqual(len(excerpts), len(set(excerpts)))
        self.assertEqual(sum("[K1]" in excerpt for excerpt in excerpts), 1)

    async def test_question_already_planned_is_not_searched_twice(self) -> None:
        record, _ = await self.run_case(
            [
                plan({"tool_name": SEARCH, "arguments": {"query": "what is the  kestrel pump flow rate?"}}),
                reply({"answer": "Kestrel moves 40 litres per minute."}),
            ],
            self.search_required,
            KESTREL_GOAL,
        )
        self.assertEqual(len(self.queries(record)), 1)
        self.assertNotIn("host_actions", record.metadata)

    async def test_calculation_only_run_gets_no_search(self) -> None:
        record, _ = await self.run_case(
            [
                plan({"tool_name": CALCULATOR, "arguments": {"expression": "37 * 14"}}),
                reply({"answer": "37 multiplied by 14 is 518."}),
            ],
            TaskRequirements(required_capabilities=[CALCULATOR], maximum_tool_calls=1),
            "Use the calculator to compute 37 multiplied by 14.",
        )
        self.assertEqual([request.tool_name for request in record.tool_requests], [CALCULATOR])

    async def test_question_search_never_exceeds_the_tool_call_limit(self) -> None:
        record, _ = await self.run_case(
            [
                plan({"tool_name": SEARCH, "arguments": {"query": "Kestrel litres"}}),
                reply({"answer": "Kestrel moves 40 litres per minute."}),
            ],
            self.search_required.model_copy(update={"maximum_tool_calls": 1}),
            KESTREL_GOAL,
        )
        self.assertEqual(self.queries(record), ["Kestrel litres"])

    async def test_question_search_respects_a_forbidden_search(self) -> None:
        record, _ = await self.run_case(
            [
                plan({"tool_name": CALCULATOR, "arguments": {"expression": "37 * 14"}}),
                reply({"answer": "37 multiplied by 14 is 518."}),
            ],
            TaskRequirements(required_capabilities=[CALCULATOR], forbidden_capabilities=[SEARCH]),
            "Use the calculator to compute 37 multiplied by 14.",
        )
        self.assertEqual(
            [result.tool_name for result in record.tool_results if result.success], [CALCULATOR]
        )


if __name__ == "__main__":
    unittest.main()
