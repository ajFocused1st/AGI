"""Provider-neutral host validation for plans and final responses."""

from __future__ import annotations

from collections.abc import Collection, Mapping

from agi_evaluation import contains_uncertainty_statement

from agi_contracts import (
    Citation,
    EvidenceReference,
    PlanRequirementResult,
    ResponseRequirementResult,
    TaskPlan,
    TaskRequirements,
    ToolDefinition,
)


def validate_plan_requirements(
    requirements: TaskRequirements,
    plan: TaskPlan,
    available_tools: Mapping[str, ToolDefinition] | Collection[str],
    *,
    correction_attempted: bool = False,
    proposed_capabilities: list[str] | None = None,
) -> PlanRequirementResult:
    """Validate proposals only; this function never authorizes or executes tools."""

    names = available_tools.keys() if isinstance(available_tools, Mapping) else available_tools
    canonical = {name.casefold(): name for name in names}
    proposed_raw = proposed_capabilities or [
        step.tool_name for step in plan.steps if step.tool_name is not None
    ]
    proposed = [canonical.get(name.casefold(), name) for name in proposed_raw]
    proposed_unique = list(dict.fromkeys(proposed))
    known = set(canonical.values())
    unknown = sorted({name for name in proposed_unique if name not in known})
    required = list(requirements.required_capabilities)
    unavailable_required = sorted(set(required) - known)
    missing = sorted(set(required) - set(proposed_unique))
    forbidden = set(requirements.forbidden_capabilities) & set(proposed_unique)
    if requirements.allowed_capabilities is not None:
        forbidden |= set(proposed_unique) - set(requirements.allowed_capabilities)
    tool_call_count = len(proposed_raw)
    messages: list[str] = []
    if missing:
        messages.append("Missing required capabilities: " + ", ".join(missing))
    if unavailable_required:
        messages.append(
            "Required capabilities are unknown to the registry: "
            + ", ".join(unavailable_required)
        )
    if forbidden:
        messages.append("Forbidden or disallowed capabilities proposed: " + ", ".join(sorted(forbidden)))
    if unknown:
        messages.append("Unknown capabilities proposed: " + ", ".join(unknown))
    if (
        requirements.maximum_tool_calls is not None
        and tool_call_count > requirements.maximum_tool_calls
    ):
        messages.append(
            f"Planned tool calls ({tool_call_count}) exceed maximum "
            f"({requirements.maximum_tool_calls})"
        )
    valid = not messages
    return PlanRequirementResult(
        valid=valid,
        required_capabilities=required,
        proposed_capabilities=proposed_unique,
        missing_capabilities=missing,
        forbidden_capabilities_proposed=sorted(forbidden),
        unknown_capabilities_proposed=unknown,
        correction_attempted=correction_attempted,
        correction_succeeded=correction_attempted and valid,
        validation_messages=messages,
    )


def validate_response_requirements(
    requirements: TaskRequirements,
    *,
    answer: str,
    evidence: list[EvidenceReference],
    citations: list[Citation],
    unsupported_claims: list[str] | None = None,
    correction_attempted: bool = False,
) -> ResponseRequirementResult:
    """Validate response evidence, citations, uncertainty, and known unsupported claims."""

    evidence_ids = {item.evidence_id for item in evidence}
    citations_resolve = all(item.evidence_id in evidence_ids for item in citations)
    citation_satisfied = (
        not requirements.requires_citations
        or (
            len(citations) >= requirements.minimum_citation_count
            and citations_resolve
        )
    )
    grounding_present = bool(evidence) and citations_resolve
    grounding_satisfied = not requirements.requires_grounding or grounding_present
    uncertainty_present = _contains_uncertainty_statement(answer)
    uncertainty_satisfied = (
        not requirements.requires_uncertainty_statement or uncertainty_present
    )
    no_unsupported_claims = not unsupported_claims
    missing: list[str] = []
    messages: list[str] = []
    if not citation_satisfied:
        missing.append("citations")
        messages.append(
            f"At least {requirements.minimum_citation_count} resolved citation(s) required"
        )
    if not grounding_satisfied:
        missing.append("grounding_evidence")
        messages.append("Grounding requires evidence retrieved during this run")
    if not uncertainty_satisfied:
        missing.append("uncertainty_statement")
        messages.append("A clear uncertainty statement is required")
    if not no_unsupported_claims:
        missing.append("unsupported_claims")
        messages.append("The existing evaluator detected unsupported claims")
    valid = not missing
    return ResponseRequirementResult(
        valid=valid,
        citations_required=requirements.requires_citations,
        citation_count=len(citations),
        citation_requirement_satisfied=citation_satisfied,
        grounding_required=requirements.requires_grounding,
        grounding_evidence_present=grounding_present,
        uncertainty_required=requirements.requires_uncertainty_statement,
        uncertainty_statement_present=uncertainty_present,
        missing_requirements=missing,
        correction_attempted=correction_attempted,
        correction_succeeded=correction_attempted and valid,
        validation_messages=messages,
    )


def _contains_uncertainty_statement(answer: str) -> bool:
    return contains_uncertainty_statement(answer)
