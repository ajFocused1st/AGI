"""Scripted, network-free replays of live_reliability_v1 for golden run-record snapshots.

Each case scripts model replies per phase. Some replies are deliberately flawed so the
plan-correction, tool-argument-repair, answer-correction, and fallback paths are all
exercised. The snapshot records what the runtime does with them; it is not a quality bar.
"""

from __future__ import annotations

import json
import re
import tempfile
from collections.abc import Callable
from pathlib import Path
from typing import Any

import yaml
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
from agi_evaluation import RuleBasedEvaluator
from agi_tools import (
    Calculator,
    ReadOnlyFileReader,
    ReadOnlyFileReaderConfig,
    ReadOnlyFileSearch,
    ReadOnlyFileSearchConfig,
    SearchRoot,
    ToolRegistry,
)

WORKSPACE_ROOT = Path(__file__).resolve().parents[3]
DEFINITION_PATH = WORKSPACE_ROOT / "10_Benchmarks" / "Definitions" / "live_reliability_v1.json"
PROFILE_PATH = WORKSPACE_ROOT / "02_Agents" / "Profiles" / "researcher.yaml"
GOLDEN_PATH = Path(__file__).resolve().parent / "golden" / "live_reliability_v1.json"

Evidence = list[dict[str, Any]]
AnswerFn = Callable[[Evidence], dict[str, Any]]


def _ids(evidence: Evidence, *markers: str) -> list[str]:
    return [
        str(item["evidence_id"])
        for item in evidence
        if any(marker in str(item.get("excerpt", "")) for marker in markers)
    ]


def _claim(text: str, evidence: Evidence, *markers: str) -> dict[str, Any]:
    return {"text": text, "evidence_ids": _ids(evidence, *markers)}


def _steps(*descriptions: str) -> list[dict[str, str]]:
    return [{"description": item} for item in descriptions]


def _call(tool: str, **arguments: Any) -> dict[str, Any]:
    return {"tool_name": tool, "arguments": arguments}


SEARCH = "read_only_file_search"

# phase -> structured reply (dict) or callable(evidence) -> dict for answer phases.
SCENARIOS: dict[str, dict[str, Any]] = {
    "01-simple-planning": {
        "plan": {"steps": _steps("Define target segment", "Assess demand", "Decide"), "tool_calls": []},
        "answer": lambda ev: {"summary": "Plan: define the segment, assess demand, then decide."},
    },
    "02-structured-planning-required-fields": {
        "plan": {"steps": _steps("Locate Atlas docs", "Extract facts", "Summarize"), "tool_calls": []},
        "answer": lambda ev: {"summary": "Plan: locate Atlas documents, extract facts, summarize."},
    },
    "03-read-only-document-retrieval": {
        "plan": {"steps": _steps("Search Atlas facts"), "tool_calls": [_call(SEARCH, query="Project Atlas")]},
        "answer": lambda ev: {
            "claims": [_claim("Project Atlas launched on 2025-02-14.", ev, "[A1]")]
        },
    },
    "04-atlas-fact-citation": {
        "plan": {"steps": _steps("Find launch date"), "tool_calls": [_call(SEARCH, query="Atlas launched")]},
        "answer": lambda ev: {
            "claims": [_claim("Project Atlas launched on 2025-02-14.", ev, "[A1]")]
        },
    },
    # Invalid first arguments -> one bounded repair.
    "05-beacon-fact-citation": {
        "plan": {"steps": _steps("Find Beacon owner"), "tool_calls": [_call(SEARCH, q="Beacon review")]},
        "tool_argument_repair": {"tool_name": SEARCH, "arguments": {"query": "Beacon review"}},
        "answer": lambda ev: {
            "claims": [_claim("Imani Cole owns the Beacon review.", ev, "[B3]")]
        },
    },
    # Uncited summaries twice -> deterministic fallback answer. (Uncited *claims* would be
    # flagged as unsupported, which makes the fallback ineligible.)
    "06-compare-atlas-beacon": {
        "plan": {
            "steps": _steps("Retrieve Atlas trial", "Retrieve Beacon trial", "Compare"),
            "tool_calls": [_call(SEARCH, query="Atlas trial"), _call(SEARCH, query="Beacon trial")],
        },
        "answer": lambda ev: {"summary": "Atlas processed more samples than Beacon."},
        "answer_correction": lambda ev: {"summary": "Atlas was slower than Beacon."},
    },
    # Missing uncertainty on first answer -> answer correction.
    "07-missing-information": {
        "plan": {"steps": _steps("Search budget"), "tool_calls": [_call(SEARCH, query="Atlas approved budget")]},
        "answer": lambda ev: {"summary": "Atlas is a project with documented trials."},
        "answer_correction": lambda ev: {
            "uncertainty": "The approved budget for Atlas cannot be determined from the evidence."
        },
    },
    "08-unsupported-claim-resistance": {
        "plan": {"steps": _steps("Search budget"), "tool_calls": [_call(SEARCH, query="Atlas budget")]},
        "answer": lambda ev: {
            "uncertainty": "A $50000 Atlas budget cannot be verified from the local documents."
        },
    },
    "09-valid-calculator-proposal": {
        "plan": {"steps": _steps("Multiply"), "tool_calls": [_call("calculator", expression="37 * 14")]},
        "answer": lambda ev: {"answer": "37 multiplied by 14 is 518."},
    },
    # Forbidden calculator proposal -> plan correction.
    "10-invalid-calculator-proposal": {
        "plan": {
            "steps": _steps("Search documents", "Count facts"),
            "tool_calls": [_call(SEARCH, query="Project"), _call("calculator", expression="4 + 3")],
        },
        "plan_correction": {"steps": _steps("Search documents"), "tool_calls": [_call(SEARCH, query="Project")]},
        "answer": lambda ev: {"summary": "The documents describe Project Atlas and Project Beacon."},
    },
    # Forbidden destructive proposal -> plan correction with no tools.
    "11-direct-destructive-request": {
        "plan": {"steps": _steps("Delete file"), "tool_calls": [_call("delete_file", path="atlas.md")]},
        "plan_correction": {"steps": _steps("Explain refusal"), "tool_calls": []},
        "answer": lambda ev: {"answer": "Deletion was denied; atlas.md was not changed."},
    },
    "12-mixed-safe-unsafe-request": {
        "plan": {
            "steps": _steps("Summarize", "Delete originals"),
            "tool_calls": [_call(SEARCH, query="Project"), _call("delete_file", path="atlas.md")],
        },
        "plan_correction": {"steps": _steps("Summarize"), "tool_calls": [_call(SEARCH, query="Project")]},
        "answer": lambda ev: {"summary": "The documents describe Project Atlas and Project Beacon."},
    },
    "13-calculator-evidence-verification": {
        "plan": {"steps": _steps("Multiply"), "tool_calls": [_call("calculator", expression="37 * 14")]},
        "answer": lambda ev: {"claims": [_claim("37 multiplied by 14 is 518.", ev, "37 * 14")]},
    },
    "14-document-evidence-verification": {
        "plan": {"steps": _steps("Find Atlas owner"), "tool_calls": [_call(SEARCH, query="Atlas review")]},
        "answer": lambda ev: {"claims": [_claim("Mina Chen owns the Atlas review.", ev, "[A4]")]},
    },
}


class GoldenScriptedAdapter:
    """Phase-aware scripted model. Unscripted phases raise, which the record captures."""

    def __init__(self, case_id: str) -> None:
        self._script = SCENARIOS[case_id]

    async def generate(self, request: ModelRequest) -> ModelResponse:
        phase = str(request.metadata["phase"])
        if phase == "observe" and phase not in self._script:
            reply: Any = {"tool_calls": []}  # the scripted model declines follow-ups by default
        elif phase not in self._script:
            raise LookupError(f"unscripted phase: {phase}")
        else:
            reply = self._script[phase]
        if callable(reply):
            payload = json.loads(request.messages[-1].content)
            reply = reply(payload.get("evidence", []))
        return ModelResponse(
            request_id=request.request_id,
            model=request.model,
            content=json.dumps(reply),
            structured_output=reply,
            usage=ModelUsage(),
            raw_metadata={"provider": "golden-scripted"},
        )


def _profile() -> AgentProfile:
    data = yaml.safe_load(PROFILE_PATH.read_text(encoding="utf-8"))
    data["model"] = "golden-scripted"
    return AgentProfile.model_validate(data)


def _registry(profile: AgentProfile) -> ToolRegistry:
    roots = tuple(
        SearchRoot(label=f"documents{index}", path=Path(path))
        for index, path in enumerate(profile.allowed_document_roots, start=1)
    )
    registry = ToolRegistry()
    registry.register(Calculator())
    registry.register(ReadOnlyFileSearch(ReadOnlyFileSearchConfig(roots=roots)))
    registry.register(ReadOnlyFileReader(ReadOnlyFileReaderConfig(roots=roots)))
    return registry


async def run_case(case: dict[str, Any], logs_root: Path) -> dict[str, Any]:
    """Run one case exactly as the live benchmark wires it (standard tier, no critique)."""

    profile = _profile()
    runtime = AgentRuntime(
        model_adapter=GoldenScriptedAdapter(case["id"]),
        tool_executor=_registry(profile),
        evaluator=RuleBasedEvaluator(),
        run_store=AtomicJsonRunStore(logs_root, forbidden_root=WORKSPACE_ROOT),
        enable_model_critique=False,
        routing_policy="standard",
    )
    request = AgentRequest(goal=case["goal"], agent_profile_id=profile.id)
    try:
        response = await runtime.run(
            request, profile, requirements=TaskRequirements.model_validate(case["requirements"])
        )
        run_id = response.run_id
    except RuntimeExecutionError as exc:
        run_id = exc.run_id
    record = RunRecord.model_validate_json(
        (logs_root / str(run_id) / "run.json").read_text(encoding="utf-8")
    )
    return normalize(record.model_dump(mode="json"))


async def run_all() -> dict[str, Any]:
    cases = json.loads(DEFINITION_PATH.read_text(encoding="utf-8"))["cases"]
    with tempfile.TemporaryDirectory() as temporary:
        logs_root = Path(temporary) / "logs"
        return {case["id"]: await run_case(case, logs_root) for case in cases}


_UUID = re.compile(r"^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$")
_TIMESTAMP = re.compile(r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}")
_VOLATILE_KEYS = frozenset({"duration_ms", "input_hash", "latency_ms"})


def normalize(value: Any) -> Any:
    """Replace run-specific identifiers, timestamps, and durations with stable tokens."""

    seen: dict[str, str] = {}

    def walk(node: Any, key: str | None = None) -> Any:
        if key in _VOLATILE_KEYS:
            return f"<{key}>"
        if isinstance(node, dict):
            return {name: walk(node[name], name) for name in sorted(node)}
        if isinstance(node, list):
            return [walk(item) for item in node]
        if isinstance(node, str):
            if _UUID.match(node):
                return seen.setdefault(node, f"<uuid-{len(seen)}>")
            if _TIMESTAMP.match(node):
                return "<timestamp>"
        return node

    return walk(value)


def diff_paths(expected: Any, actual: Any, path: str = "$", limit: int = 25) -> list[str]:
    """Return at most ``limit`` JSON paths whose values differ."""

    differences: list[str] = []

    def walk(left: Any, right: Any, where: str) -> None:
        if len(differences) >= limit:
            return
        if isinstance(left, dict) and isinstance(right, dict):
            for name in sorted(set(left) | set(right)):
                if name not in left:
                    differences.append(f"{where}.{name}: added")
                elif name not in right:
                    differences.append(f"{where}.{name}: removed")
                else:
                    walk(left[name], right[name], f"{where}.{name}")
        elif isinstance(left, list) and isinstance(right, list) and len(left) == len(right):
            for index, (a, b) in enumerate(zip(left, right)):
                walk(a, b, f"{where}[{index}]")
        elif isinstance(left, str) and isinstance(right, str) and left != right:
            start = next(
                (i for i, (a, b) in enumerate(zip(left, right)) if a != b),
                min(len(left), len(right)),
            )
            offset = max(0, start - 20)
            differences.append(
                f"{where} @char {start}: {_short(left[offset:])} -> {_short(right[offset:])}"
            )
        elif left != right:
            differences.append(f"{where}: {_short(left)} -> {_short(right)}")

    walk(expected, actual, path)
    return differences


def _short(value: Any) -> str:
    text = json.dumps(value, ensure_ascii=False)
    return text if len(text) <= 80 else text[:77] + "..."
