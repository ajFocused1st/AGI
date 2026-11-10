"""Compact, model-facing prompt payloads.

The model only needs what it can act on: tool names, descriptions, and argument fields;
non-default requirements; step descriptions; tool outcomes not already represented as
evidence; and evidence ids with excerpts. Hashes, timestamps, permissions, output schemas,
and duplicated tool output stay host-side in the run record.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from typing import Any

from agi_contracts import EvidenceReference, TaskPlan, TaskRequirements, ToolDefinition, ToolResult

MAX_PROMPT_EVIDENCE = 24
_ARGUMENT_KEYS = ("type", "description", "enum", "minimum", "maximum", "maxLength", "default")


def compact_tool_definitions(definitions: Iterable[ToolDefinition]) -> list[dict[str, Any]]:
    compact: list[dict[str, Any]] = []
    for definition in definitions:
        schema = definition.input_schema
        arguments = {
            name: {key: spec[key] for key in _ARGUMENT_KEYS if key in spec}
            for name, spec in schema.get("properties", {}).items()
            if isinstance(spec, Mapping)
        }
        compact.append(
            {
                "name": definition.name,
                "description": definition.description,
                "arguments": arguments,
                "required": list(schema.get("required", [])),
            }
        )
    return compact


def compact_requirements(requirements: TaskRequirements | None) -> dict[str, Any] | None:
    if requirements is None:
        return None
    defaults = TaskRequirements().model_dump(mode="json")
    current = requirements.model_dump(mode="json")
    return {
        key: value
        for key, value in current.items()
        if key != "requirement_policy_version" and value != defaults.get(key)
    }


def compact_plan(plan: TaskPlan) -> list[str]:
    return [step.description for step in plan.steps]


def compact_tool_outcomes(results: Iterable[ToolResult]) -> list[dict[str, Any]]:
    """Outcomes only; successful outputs reach the model as evidence instead."""

    outcomes: list[dict[str, Any]] = []
    for result in results:
        outcome: dict[str, Any] = {"tool": result.tool_name, "ok": result.success}
        if result.error:
            outcome["error"] = result.error[:200]
        matches = result.output.get("matches") if result.success else None
        if isinstance(matches, list):
            outcome["match_count"] = len(matches)
        outcomes.append(outcome)
    return outcomes


def compact_evidence(
    evidence: Iterable[EvidenceReference], limit: int = MAX_PROMPT_EVIDENCE
) -> list[dict[str, Any]]:
    """Round-robin across retrieval queries so a later search is never crowded out."""

    groups: dict[str, list[EvidenceReference]] = {}
    for item in evidence:
        key = f"{item.metadata.get('retrieval_tool')}|{item.metadata.get('retrieval_query')}"
        groups.setdefault(key, []).append(item)
    selected: list[EvidenceReference] = []
    queues = [list(group) for group in groups.values()]
    while queues and len(selected) < limit:
        for queue in queues:
            if queue and len(selected) < limit:
                selected.append(queue.pop(0))
        queues = [queue for queue in queues if queue]
    order = {id(item): index for index, item in enumerate(evidence)} if isinstance(evidence, list) else {}
    selected.sort(key=lambda item: order.get(id(item), 0))
    return [
        {
            "evidence_id": str(item.evidence_id),
            "source": f"{item.source_path} ({item.locator or 'source'})",
            "excerpt": item.excerpt,
        }
        for item in selected
    ]
