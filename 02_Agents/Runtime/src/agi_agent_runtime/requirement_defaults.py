"""Deterministic, model-free default task requirements for callers that supply none.

Without requirements a run cannot be held to grounding: a model may skip retrieval,
claim "no evidence", and still pass. These rules use only the goal text and the
profile's allowed tools. They never grant capabilities; they only require approved
ones or forbid mutation. Callers that pass explicit requirements are unaffected.
"""

from __future__ import annotations

import re
from collections.abc import Collection

from agi_contracts import TaskRequirements

INFERRED_POLICY_VERSION = "task-requirements.inferred-v1"
SEARCH_TOOL = "read_only_file_search"
CALCULATOR_TOOL = "calculator"

_PLAN_ONLY = re.compile(
    r"\b(create|make|draft|write|give|outline|propose)\b[^.?!]*\b(plan|steps|roadmap|outline)\b",
    re.IGNORECASE,
)
_ARITHMETIC = re.compile(
    r"\b(calculate|compute|multiply|multiplied|divide|divided|subtract|product of|sum of|"
    r"squared|cubed|square root)\b"
    r"|\d\s*[-+*/x×÷]\s*\d",
    re.IGNORECASE,
)
_RESEARCH = re.compile(
    r"\?|\b(what|when|who|whom|which|where|why|how|find|search|look up|summari[sz]e|"
    r"compare|explain|describe|list|identify|report|state|verify|tell)\b",
    re.IGNORECASE,
)
_CITATION_REQUEST = re.compile(r"\b(cite|cited|citing|citations?|sources?)\b", re.IGNORECASE)
_MUTATION_VERBS = {
    "delete_file": re.compile(r"\b(delete|remove|erase|wipe|destroy|shred|purge)\b", re.IGNORECASE),
    "write_file": re.compile(r"\b(overwrite|modify|edit|truncate|append to|write to)\b", re.IGNORECASE),
    "move_file": re.compile(r"\b(rename|move)\b", re.IGNORECASE),
}


def infer_task_requirements(goal: str, allowed_tools: Collection[str]) -> TaskRequirements:
    allowed = set(allowed_tools)
    forbidden = sorted(
        capability
        for capability, pattern in _MUTATION_VERBS.items()
        if pattern.search(goal) and capability not in allowed
    )
    required: list[str] = []
    grounded = False
    if _PLAN_ONLY.search(goal):
        pass  # A plan request needs no tools or evidence.
    elif _ARITHMETIC.search(goal) and CALCULATOR_TOOL in allowed:
        required.append(CALCULATOR_TOOL)
        # An explicit citation request makes the computed result citable evidence.
        grounded = bool(_CITATION_REQUEST.search(goal))
    elif _RESEARCH.search(goal) and SEARCH_TOOL in allowed:
        required.append(SEARCH_TOOL)
        grounded = True
    return TaskRequirements(
        required_capabilities=required,
        requires_grounding=grounded,
        requires_citations=grounded,
        minimum_citation_count=1 if grounded else 0,
        forbidden_capabilities=forbidden,
        requirement_policy_version=INFERRED_POLICY_VERSION,
    )
