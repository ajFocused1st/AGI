"""Tool calls a small model misspells are still performed, and lost ones are recorded.

Also covers the host's search on the user's own question, which runs beside the plan.
"""

from __future__ import annotations

import re
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


class CitingAdapter(ScriptedModelAdapter):
    """Answers by citing the first evidence item it is shown, as the live model did."""

    def __init__(self, replies: list[ScriptedReply], claims: list[str]) -> None:
        super().__init__(replies)
        self._claims = claims

    async def generate(self, request):  # type: ignore[override]
        if "answer" in str(request.metadata.get("phase")):
            shown = re.search(r'"evidence_id": "([0-9a-f-]{36})"', request.messages[-1].content)
            ids = [shown.group(1)] if shown else []
            self._replies.appendleft(
                reply({"claims": [{"text": text, "evidence_ids": ids} for text in self._claims]})
            )
        return await super().generate(request)


class PhaseAdapter(ScriptedModelAdapter):
    """Replies by request phase, so a test does not depend on how many calls a path makes."""

    BY_PHASE = {
        "plan": {"steps": [{"description": "Research"}], "tool_calls": [{"tool_name": SEARCH, "arguments": {"query": "Kestrel litres"}}]},
        "observe": {"tool_calls": []},
    }

    def __init__(self, answer: str) -> None:
        super().__init__([])
        self._answer = answer

    async def generate(self, request):  # type: ignore[override]
        phase = str(request.metadata.get("phase"))
        payload = self.BY_PHASE.get(phase.replace("_correction", "")) or {"answer": self._answer}
        self._replies.append(reply(payload))
        return await super().generate(request)


class FakeResponseValidationError(Exception):
    """Named like the adapters' output errors (OllamaResponseValidationError)."""


class UnreadableFirstAnswerAdapter(ScriptedModelAdapter):
    """The first answer is cut off mid-JSON; later answers are fine."""

    def __init__(self, replies: list[ScriptedReply]) -> None:
        super().__init__(replies)
        self.failed = False

    async def generate(self, request):  # type: ignore[override]
        if request.metadata.get("phase") == "answer" and not self.failed:
            self.failed = True
            self.requests.append(request.model_copy(deep=True))
            raise FakeResponseValidationError("Model output was not strict JSON (Unterminated string)")
        return await super().generate(request)


class UnreadablePhaseAdapter(ScriptedModelAdapter):
    """Output for one phase is cut off the first time; everything else is scripted."""

    def __init__(self, replies: list[ScriptedReply], phase: str) -> None:
        super().__init__(replies)
        self.phase = phase
        self.failed = False

    async def generate(self, request):  # type: ignore[override]
        if request.metadata.get("phase") == self.phase and not self.failed:
            self.failed = True
            self.requests.append(request.model_copy(deep=True))
            raise FakeResponseValidationError("Model output was not strict JSON (Unterminated string)")
        return await super().generate(request)


class TruncatedIdAdapter(CitingAdapter):
    """Cites the evidence id with its last character missing, as the live model did."""

    async def generate(self, request):  # type: ignore[override]
        if "answer" in str(request.metadata.get("phase")):
            shown = re.search(r'"evidence_id": "([0-9a-f-]{36})"', request.messages[-1].content)
            self._replies.appendleft(
                reply({"claims": [{"text": text, "evidence_ids": [shown.group(1)[:-1]]} for text in self._claims]})
            )
        return await ScriptedModelAdapter.generate(self, request)


class MistypedIdAdapter(CitingAdapter):
    """Cites the evidence id with one character changed in the middle (live Northwind Q8)."""

    async def generate(self, request):  # type: ignore[override]
        if "answer" in str(request.metadata.get("phase")):
            shown = re.search(r'"evidence_id": "([0-9a-f-]{36})"', request.messages[-1].content).group(1)
            typo = shown[:10] + ("0" if shown[10] != "0" else "1") + shown[11:]
            self._replies.appendleft(
                reply({"claims": [{"text": text, "evidence_ids": [typo]} for text in self._claims]})
            )
        return await ScriptedModelAdapter.generate(self, request)


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

    async def run_case(
        self,
        replies: list[ScriptedReply],
        requirements: TaskRequirements,
        goal: str,
        adapter: ScriptedModelAdapter | None = None,
        host_plan: bool = False,
    ):
        adapter = adapter or ScriptedModelAdapter(replies)
        runtime = AgentRuntime(
            model_adapter=adapter,
            tool_executor=self.registry,
            evaluator=RuleBasedEvaluator(),
            run_store=AtomicJsonRunStore(Path(self._logs.name) / "logs"),
            enable_model_critique=False,
            host_plan_search_questions=host_plan,
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

    # --- a calculation over document numbers -----------------------------------------

    async def test_empty_plan_for_a_document_calculation_searches_then_calculates(self) -> None:
        # Live, real-documents run: the model planned no tools for a question that needed a
        # search and the calculator, so dispatch was blocked and nothing was searched.
        goal = "How many litres per minute do Kestrel and Osprey move combined?"
        record, _ = await self.run_case(
            [
                plan(),
                plan(),
                reply({"tool_calls": [{"tool_name": CALCULATOR, "arguments": {"expression": "40 + 55"}}]}),
                reply({"answer": "Together they move 95 litres per minute."}),
            ],
            TaskRequirements(required_capabilities=[SEARCH, CALCULATOR]),
            goal,
        )
        self.assertEqual([request.tool_name for request in record.tool_requests], [SEARCH, CALCULATOR])
        self.assertEqual(self.queries(record), [goal])
        self.assertTrue(all(result.success for result in record.tool_results))
        self.assertNotIn("tool_dispatch_blocked", [error.code for error in record.errors])
        self.assertEqual(
            record.metadata["host_actions"],
            [
                {"action": "host_required_retrieval", "tool_name": SEARCH},
                {"action": "deferred_required_capability", "capabilities": [CALCULATOR]},
            ],
        )

    async def test_calculation_planned_before_its_numbers_waits_instead_of_failing(self) -> None:
        # Live runs: the plan's calculator call had no expression yet; the repair invented
        # numbers and logged tool_argument_repair_failed, though the follow-up later succeeded.
        goal = "How many litres per minute do Kestrel and Osprey move combined?"
        record, adapter = await self.run_case(
            [
                plan(
                    {"tool_name": SEARCH, "arguments": {"query": "Kestrel Osprey litres per minute"}},
                    {"tool_name": CALCULATOR, "arguments": {"expression": None}},
                ),
                reply({"tool_calls": [{"tool_name": CALCULATOR, "arguments": {"expression": "40 + 55"}}]}),
                reply({"answer": "Together they move 95 litres per minute."}),
            ],
            TaskRequirements(required_capabilities=[SEARCH, CALCULATOR]),
            goal,
        )
        self.assertNotIn("tool_argument_repair_failed", [error.code for error in record.errors])
        self.assertNotIn("tool_argument_repair", [item.metadata.get("phase") for item in adapter.requests])
        calculations = [r for r in record.tool_results if r.tool_name == CALCULATOR]
        self.assertEqual([r.success for r in calculations], [True])

    # --- citations a task did not ask for ----------------------------------------------

    PLAN_STEPS = [
        "Step 1: Gather every pump specification sheet.",
        "Step 2: Tabulate maintenance intervals for each pump.",
        "Step 3: Recommend a replacement schedule.",
    ]

    async def test_unsupported_citations_on_a_plan_are_removed_not_failed(self) -> None:
        # Live benchmark v1 case 02: a three-step plan cited a document on every step, so
        # the steps were checked as facts and the run failed.
        adapter = CitingAdapter(
            [plan({"tool_name": SEARCH, "arguments": {"query": "Kestrel pump"}})], self.PLAN_STEPS
        )
        record, _ = await self.run_case(
            [], TaskRequirements(), "Write three steps for reviewing the Kestrel pump.", adapter
        )
        self.assertTrue(record.evidence_bundle.items)  # there was something to (mis)cite
        self.assertEqual(record.response.citations, [])
        self.assertNotIn("kestrel.md", record.response.answer)
        self.assertIn({"action": "removed_unsupported_citations"}, record.metadata["host_actions"])
        self.assertNotEqual(record.evaluation.verdict.value, "failed")

    async def test_unsupported_citations_still_fail_when_sources_are_required(self) -> None:
        adapter = CitingAdapter(
            [plan({"tool_name": SEARCH, "arguments": {"query": "Kestrel pump"}})], self.PLAN_STEPS
        )
        record, _ = await self.run_case(
            [],
            TaskRequirements(
                required_capabilities=[SEARCH],
                requires_grounding=True,
                requires_citations=True,
                minimum_citation_count=1,
            ),
            "Write three steps for reviewing the Kestrel pump.",
            adapter,
        )
        self.assertNotIn("removed_unsupported_citations", str(record.metadata.get("host_actions")))
        self.assertEqual(record.evaluation.verdict.value, "failed")
        # The correction is told exactly what was unsupported, not just "unsupported_claims".
        correction = next(r for r in adapter.requests if r.metadata.get("phase") == "answer_correction")
        payload = correction.messages[-1].content
        self.assertIn('"problems_to_fix"', payload)
        self.assertIn("document_evidence_support:", payload)
        self.assertIn('"how_to_fix"', payload)

    # --- "not found" answers -------------------------------------------------------------

    WARRANTY_GOAL = "What is the Kestrel pump warranty period?"
    NOT_FOUND = "The available evidence does not contain the Kestrel pump warranty period."

    def not_found_script(self) -> list[ScriptedReply]:
        return [
            plan({"tool_name": SEARCH, "arguments": {"query": "Kestrel pump warranty"}}),
            reply({"answer": self.NOT_FOUND}),
            reply({"tool_calls": []}),  # the follow-up after a "not found" draft finds nothing new
            reply({"answer": self.NOT_FOUND}),
        ]

    async def test_an_unconfirmed_not_found_answer_is_partial(self) -> None:
        record, _ = await self.run_case(
            self.not_found_script(), TaskRequirements(required_capabilities=[SEARCH]), self.WARRANTY_GOAL
        )
        self.assertEqual(record.evaluation.verdict.value, "partial")
        check = next(c for c in record.evaluation.rule_based_checks if c.name == "absence_verified")
        self.assertFalse(check.passed)

    async def test_an_expected_not_found_answer_still_passes(self) -> None:
        record, _ = await self.run_case(
            self.not_found_script(),
            TaskRequirements(required_capabilities=[SEARCH], requires_uncertainty_statement=True),
            self.WARRANTY_GOAL,
        )
        self.assertEqual(record.evaluation.verdict.value, "passed")

    # --- the host's own plan for a search-only question -------------------------------

    async def test_search_only_question_skips_the_planning_call(self) -> None:
        record, adapter = await self.run_case(
            [reply({"answer": "Kestrel moves 40 litres per minute."})],
            self.search_required,
            KESTREL_GOAL,
            host_plan=True,
        )
        phases = [request.metadata.get("phase") for request in adapter.requests]
        self.assertNotIn("plan", phases)
        self.assertEqual(phases, ["answer"])
        self.assertEqual(self.queries(record), [KESTREL_GOAL])
        self.assertIn({"action": "host_plan", "tool_name": SEARCH}, record.metadata["host_actions"])
        self.assertTrue(record.plan_requirement_result.valid)
        self.assertEqual(record.evaluation.verdict.value, "passed")

    async def test_tasks_with_real_planning_decisions_still_get_a_model_plan(self) -> None:
        cases = {
            "calculation": TaskRequirements(required_capabilities=[SEARCH, CALCULATOR]),
            "refusal": TaskRequirements(required_capabilities=[SEARCH], forbidden_capabilities=["delete_file"]),
            "comparison": TaskRequirements(required_capabilities=[SEARCH], task_subjects=["Kestrel", "Osprey"]),
        }
        for name, requirements in cases.items():
            with self.subTest(name=name):
                record, adapter = await self.run_case(
                    [],
                    requirements,
                    RIVAL_GOAL,
                    PhaseAdapter("Kestrel moves 40 litres per minute."),
                    host_plan=True,
                )
                self.assertEqual(adapter.requests[0].metadata.get("phase"), "plan")
                self.assertNotIn("host_plan", str(record.metadata.get("host_actions")))

    async def test_the_model_plans_when_the_setting_is_off(self) -> None:
        _, adapter = await self.run_case(
            [plan({"tool_name": SEARCH, "arguments": {"query": "Kestrel litres"}}), reply({"answer": "x"})],
            self.search_required,
            KESTREL_GOAL,
        )
        self.assertEqual(adapter.requests[0].metadata.get("phase"), "plan")

    # --- model output that cannot be read ------------------------------------------------

    async def test_an_unreadable_answer_is_a_failed_attempt_not_a_crash(self) -> None:
        adapter = UnreadableFirstAnswerAdapter(
            [
                plan({"tool_name": SEARCH, "arguments": {"query": "Kestrel litres"}}),
                reply({"answer": "Kestrel moves 40 litres per minute."}),
            ]
        )
        record, _ = await self.run_case([], TaskRequirements(required_capabilities=[SEARCH]), KESTREL_GOAL, adapter)
        unreadable = [e for e in record.errors if e.code == "answer_output_unreadable"]
        self.assertEqual(len(unreadable), 1)
        self.assertEqual(unreadable[0].details["phase"], "answer")
        self.assertEqual(record.response.answer, "Kestrel moves 40 litres per minute.")
        correction = next(r for r in adapter.requests if r.metadata.get("phase") == "answer_correction")
        self.assertIn("malformed or cut off", correction.messages[-1].content)

    async def test_an_evidence_id_missing_a_character_is_matched(self) -> None:
        adapter = TruncatedIdAdapter(
            [plan({"tool_name": SEARCH, "arguments": {"query": "Kestrel litres"}})],
            ["Kestrel moves 40 litres per minute."],
        )
        record, _ = await self.run_case(
            [],
            TaskRequirements(
                required_capabilities=[SEARCH], requires_grounding=True, requires_citations=True, minimum_citation_count=1
            ),
            KESTREL_GOAL,
            adapter,
        )
        self.assertEqual(record.status.value, "completed")
        self.assertEqual(len(record.response.citations), 1)
        self.assertIn("evidence_id_repaired", [e.code for e in record.errors])

    async def test_an_evidence_id_with_a_mistyped_character_is_matched(self) -> None:
        adapter = MistypedIdAdapter(
            [plan({"tool_name": SEARCH, "arguments": {"query": "Kestrel litres"}})],
            ["Kestrel moves 40 litres per minute."],
        )
        record, _ = await self.run_case(
            [],
            TaskRequirements(
                required_capabilities=[SEARCH], requires_grounding=True, requires_citations=True, minimum_citation_count=1
            ),
            KESTREL_GOAL,
            adapter,
        )
        self.assertEqual(len(record.response.citations), 1)
        self.assertEqual(record.evaluation.verdict.value, "passed")

    def test_an_invented_evidence_id_is_dropped_not_guessed(self) -> None:
        from uuid import uuid4

        from agi_agent_runtime.runtime import _reconcile_evidence_ids
        from agi_contracts import EvidenceReference

        evidence = [
            EvidenceReference(evidence_id=uuid4(), source_id="d", source_path="d.md", excerpt="x")
            for _ in range(3)
        ]
        payload, changed = _reconcile_evidence_ids({"cited_evidence_ids": [str(uuid4())]}, evidence)
        self.assertEqual(payload["cited_evidence_ids"], [])
        self.assertEqual(changed, 1)

    # --- two-step questions: search for the person the answer named ----------------------

    AUDIT_GOAL = "What is the job title of the person in charge of the Lantern Audit?"

    def write_audit_documents(self) -> None:
        root = Path(self._documents.name)
        (root / "reviews.md").write_text("# Reviews\n\nThe Lantern Audit is led by Miran Seln.\n", encoding="utf-8")
        # Shares no word with the question, so only a search for the name finds it.
        (root / "directory.md").write_text(
            "# Staff directory\n\nMiran Seln works as Inventory Coordinator, from Dale.\n", encoding="utf-8"
        )

    async def test_an_answer_naming_the_described_person_triggers_a_search_for_them(self) -> None:
        self.write_audit_documents()
        record, adapter = await self.run_case(
            [
                plan({"tool_name": SEARCH, "arguments": {"query": "Lantern Audit leader"}}),
                reply({"answer": "Miran Seln leads the Lantern Audit."}),
                reply({"answer": "Miran Seln, who leads the Lantern Audit, is the Inventory Coordinator."}),
            ],
            TaskRequirements(required_capabilities=[SEARCH]),
            self.AUDIT_GOAL,
        )
        bridge = next(a for a in record.metadata["host_actions"] if a["action"] == "host_bridge_search")
        self.assertTrue(bridge["queries"][0].startswith("Miran Seln"))
        self.assertIn("Inventory Coordinator", " ".join(i.excerpt for i in record.evidence_bundle.items))
        bridge_request = next(r for r in adapter.requests if r.metadata.get("phase") == "answer_bridge")
        self.assertIn("Your previous answer named Miran Seln", bridge_request.messages[-1].content)
        self.assertIn("Inventory Coordinator", record.response.answer)

    async def test_a_question_asking_who_mid_sentence_is_not_searched_again(self) -> None:
        # Live multihop case mh-03: "Compare who is responsible ..." asks for the names.
        self.write_audit_documents()
        record, _ = await self.run_case(
            [
                plan({"tool_name": SEARCH, "arguments": {"query": "Lantern Audit leader"}}),
                reply({"answer": "Miran Seln leads the Lantern Audit."}),
            ],
            TaskRequirements(required_capabilities=[SEARCH]),
            "Compare who leads the Lantern Audit and who leads the Thistle Cycle.",
        )
        self.assertNotIn("host_bridge_search", str(record.metadata.get("host_actions")))

    async def test_a_who_question_is_not_searched_again(self) -> None:
        self.write_audit_documents()
        record, adapter = await self.run_case(
            [
                plan({"tool_name": SEARCH, "arguments": {"query": "Lantern Audit leader"}}),
                reply({"answer": "Miran Seln leads the Lantern Audit."}),
            ],
            TaskRequirements(required_capabilities=[SEARCH]),
            "Who is in charge of the Lantern Audit?",
        )
        self.assertNotIn("host_bridge_search", str(record.metadata.get("host_actions")))
        self.assertNotIn("answer_bridge", [r.metadata.get("phase") for r in adapter.requests])

    async def test_citations_still_resolve_after_a_later_search(self) -> None:
        # Live benchmarks: a search made after the answer rebuilt the evidence with new ids,
        # and the answer's citations pointed at nothing.
        self.write_audit_documents()
        adapter = CitingAdapter(
            [plan({"tool_name": SEARCH, "arguments": {"query": "Lantern Audit leader"}})],
            ["Miran Seln leads the Lantern Audit."],
        )
        record, _ = await self.run_case(
            [],
            TaskRequirements(
                required_capabilities=[SEARCH], requires_grounding=True, requires_citations=True, minimum_citation_count=1
            ),
            self.AUDIT_GOAL,
            adapter,
        )
        self.assertIn("host_bridge_search", str(record.metadata.get("host_actions")))
        evidence_ids = {item.evidence_id for item in record.evidence_bundle.items}
        self.assertTrue(record.response.citations)
        self.assertTrue(all(c.evidence_id in evidence_ids for c in record.response.citations))

    async def test_a_cut_off_follow_up_or_plan_is_not_a_crash(self) -> None:
        requirements = self.search_required.model_copy(update={"task_subjects": ["Kestrel", "Osprey"]})
        adapter = UnreadablePhaseAdapter(
            [
                plan({"tool_name": SEARCH, "arguments": {"query": "Kestrel flow rate"}}),
                reply({"answer": "Kestrel moves 40 litres per minute."}),
            ],
            phase="observe",
        )
        record, _ = await self.run_case([], requirements, RIVAL_GOAL, adapter)
        self.assertIn("follow_up_output_unreadable", [e.code for e in record.errors])
        self.assertEqual(record.status.value, "completed")

        adapter = UnreadablePhaseAdapter(
            [
                plan({"tool_name": SEARCH, "arguments": {"query": "Kestrel litres"}}),
                reply({"answer": "Kestrel moves 40 litres per minute."}),
            ],
            phase="plan",
        )
        record, _ = await self.run_case([], self.search_required, KESTREL_GOAL, adapter)
        self.assertIn("plan_output_unreadable", [e.code for e in record.errors])
        self.assertEqual(record.status.value, "completed")
        self.assertEqual(record.response.answer, "Kestrel moves 40 litres per minute.")

    async def test_an_empty_repair_does_not_replace_the_answer(self) -> None:
        # Live runs: a repair returned nothing and the user saw "The response requirements
        # could not be satisfied." in place of a correct but imperfectly cited answer.
        record, _ = await self.run_case(
            [
                plan({"tool_name": SEARCH, "arguments": {"query": "Kestrel litres"}}),
                # An uncited claim in a grounded task, so a repair is requested.
                reply({"claims": [{"text": "Kestrel moves 40 litres per minute, says engineer Dana Ruiz.", "evidence_ids": []}]}),
                reply({"claims": []}),
            ],
            TaskRequirements(required_capabilities=[SEARCH], requires_grounding=True),
            KESTREL_GOAL,
        )
        self.assertIn("answer_correction_empty", [e.code for e in record.errors])
        self.assertTrue(record.response.answer.startswith("Kestrel moves 40 litres per minute"))

    async def test_an_empty_first_answer_is_never_graded_as_an_answer(self) -> None:
        # Live Veyra Q35: an empty first answer was shown as "The response requirements
        # could not be satisfied." and graded passed.
        record, _ = await self.run_case(
            [
                plan({"tool_name": SEARCH, "arguments": {"query": "Kestrel litres"}}),
                reply({"claims": []}),
                reply({"answer": "Kestrel moves 40 litres per minute."}),
            ],
            self.search_required,
            KESTREL_GOAL,
        )
        self.assertIn("answer_output_empty", [e.code for e in record.errors])
        self.assertEqual(record.response.answer, "Kestrel moves 40 litres per minute.")

        record, _ = await self.run_case(
            [plan({"tool_name": SEARCH, "arguments": {"query": "Kestrel litres"}}), reply({"claims": []}), reply({"claims": []})],
            self.search_required,
            KESTREL_GOAL,
        )
        self.assertNotEqual(record.evaluation.verdict.value, "passed")
        self.assertNotIn("could not be satisfied", record.response.answer)

    def test_repair_advice_names_terms_only_when_the_checks_did(self) -> None:
        from agi_agent_runtime.runtime import _correction_advice

        named = _correction_advice(["document_evidence_support: supported concrete terms: 3/5; missing cited entities: dana, ruiz"])
        self.assertIn("dana, ruiz", named)
        vague = _correction_advice(["document_evidence_support: supported concrete terms: 1/5"])
        self.assertNotIn("Remove", vague)
        self.assertIn("Keep answering the question", vague)

    async def test_a_blocked_run_without_its_required_tool_cannot_pass(self) -> None:
        # Live run: no tool ran, the answer said "the evidence does not contain" it, and the
        # verdict was "passed".
        record, _ = await self.run_case(
            [plan(), plan(), reply({"answer": "The available evidence does not contain the charter date."})],
            TaskRequirements(required_capabilities=[CALCULATOR]),
            "What is the charter date?",
        )
        self.assertIn("tool_dispatch_blocked", [error.code for error in record.errors])
        self.assertEqual(record.tool_results, [])
        self.assertEqual(record.evaluation.verdict.value, "failed")


if __name__ == "__main__":
    unittest.main()
