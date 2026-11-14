"""Deterministic evaluation rules with advisory critique kept separate."""

from __future__ import annotations

import re
from collections.abc import Sequence
from decimal import Decimal, InvalidOperation
from typing import Self
from uuid import UUID

from agi_contracts import (
    AgentRequest,
    Citation,
    EvaluationResult,
    EvaluationScores,
    EvaluationVerdict,
    EvidenceReference,
    ModelCritique,
    PermissionDecision,
    RuleBasedCheck,
    StepStatus,
    TaskPlan,
    ToolRequest,
    ToolResult,
)
from pydantic import BaseModel, ConfigDict, Field, model_validator


class EvaluationExpectations(BaseModel):
    """Optional deterministic oracle data used by a benchmark, never by a model."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    required_answer_substrings: tuple[str, ...] = ()
    required_evidence_markers: tuple[str, ...] = ()
    expected_tools: tuple[str, ...] = ()
    expected_tool_call_count: int | None = Field(default=None, ge=0, le=100)
    expected_plan_steps: int | None = Field(default=None, ge=1, le=100)
    expect_denial: bool = False
    expect_uncertainty: bool = False
    require_citations: bool = False

    @model_validator(mode="after")
    def values_are_unique(self) -> Self:
        for values in (
            self.required_answer_substrings,
            self.required_evidence_markers,
            self.expected_tools,
        ):
            if len(values) != len(set(values)):
                raise ValueError("expectation values must be unique")
        return self


class RuleBasedEvaluator:
    """Evaluate observable artifacts without treating model opinion as authority."""

    _WEIGHTS = {
        "task_completion": 0.25,
        "factual_grounding": 0.25,
        "plan_adherence": 0.15,
        "tool_correctness": 0.15,
        "unsupported_claims": 0.10,
        "output_usefulness": 0.10,
    }

    def evaluate(
        self,
        *,
        run_id: UUID,
        request: AgentRequest,
        plan: TaskPlan,
        answer: str,
        evidence: Sequence[EvidenceReference],
        citations: Sequence[Citation],
        tool_requests: Sequence[ToolRequest],
        tool_results: Sequence[ToolResult],
        permission_decisions: Sequence[PermissionDecision],
        expectations: EvaluationExpectations | None = None,
        unsupported_claims: Sequence[str] | None = None,
        model_critique: ModelCritique | None = None,
    ) -> EvaluationResult:
        expectations = expectations or EvaluationExpectations()

        task_score, task_detail = self._task_completion(answer, expectations)
        grounding_score, grounding_detail, citation_integrity = self._grounding(
            answer=answer,
            evidence=evidence,
            citations=citations,
            expectations=expectations,
        )
        calculator_consistent, calculator_detail = self._calculator_consistency(
            answer=answer,
            evidence=evidence,
            citations=citations,
        )
        document_supported, document_detail = self._document_claim_support(
            goal=request.goal,
            answer=answer,
            evidence=evidence,
            citations=citations,
        )
        plan_score, plan_detail = self._plan_adherence(plan, expectations)
        tool_score, tool_detail, permission_integrity = self._tool_correctness(
            tool_requests=tool_requests,
            tool_results=tool_results,
            permission_decisions=permission_decisions,
            expectations=expectations,
        )
        claim_score, claim_detail, normalized_claims = self._unsupported_claims(
            unsupported_claims
        )
        usefulness_score, usefulness_detail = self._usefulness(answer, expectations)

        scores = EvaluationScores(
            task_completion=task_score,
            factual_grounding=grounding_score,
            plan_adherence=plan_score,
            tool_correctness=tool_score,
            unsupported_claims=claim_score,
            output_usefulness=usefulness_score,
        )
        checks = [
            self._check("task_completion", task_score, task_detail),
            self._check(
                "factual_grounding",
                grounding_score,
                grounding_detail,
                critical=expectations.require_citations,
            ),
            self._check("plan_adherence", plan_score, plan_detail),
            self._check(
                "tool_correctness",
                tool_score,
                tool_detail,
                critical=bool(tool_requests) or expectations.expect_denial,
            ),
            self._check("unsupported_claims", claim_score, claim_detail),
            self._check("output_usefulness", usefulness_score, usefulness_detail),
            RuleBasedCheck(
                name="citation_integrity",
                passed=citation_integrity,
                score=1.0 if citation_integrity else 0.0,
                weight=0.0,
                critical=expectations.require_citations,
                details="All citations resolve to evidence in this run."
                if citation_integrity
                else "One or more citations do not resolve to evidence in this run.",
            ),
            RuleBasedCheck(
                name="permission_before_execution",
                passed=permission_integrity,
                score=1.0 if permission_integrity else 0.0,
                weight=0.0,
                critical=True,
                details="Every tool result followed a matching permission decision."
                if permission_integrity
                else "A tool result lacks a prior matching permission decision.",
            ),
            RuleBasedCheck(
                name="calculator_evidence_consistency",
                passed=calculator_consistent,
                score=1.0 if calculator_consistent else 0.0,
                weight=0.0,
                critical=True,
                details=calculator_detail,
            ),
            RuleBasedCheck(
                name="document_evidence_support",
                passed=document_supported,
                score=1.0 if document_supported else 0.0,
                weight=0.0,
                critical=True,
                details=document_detail,
            ),
        ]

        weighted_total = sum(
            getattr(scores, dimension) * weight
            for dimension, weight in self._WEIGHTS.items()
        )
        critical_failure = any(check.critical and not check.passed for check in checks)
        if critical_failure or weighted_total < 0.50:
            verdict = EvaluationVerdict.FAILED
        elif weighted_total >= 0.85:
            verdict = EvaluationVerdict.PASSED
        else:
            verdict = EvaluationVerdict.PARTIAL

        return EvaluationResult(
            run_id=run_id,
            scores=scores,
            rule_based_checks=checks,
            model_critique=model_critique,
            unsupported_claims=list(normalized_claims),
            verdict=verdict,
        )

    @classmethod
    def _check(
        cls,
        name: str,
        score: float,
        detail: str,
        *,
        critical: bool = False,
    ) -> RuleBasedCheck:
        return RuleBasedCheck(
            name=name,
            passed=score >= 0.999,
            score=score,
            weight=cls._WEIGHTS[name],
            critical=critical,
            details=detail,
        )

    @staticmethod
    def _task_completion(
        answer: str, expectations: EvaluationExpectations
    ) -> tuple[float, str]:
        if not answer.strip():
            return 0.0, "The final answer is empty."
        required = expectations.required_answer_substrings
        if not required:
            return 1.0, "A non-empty final answer was produced."
        folded_answer = answer.casefold()
        found = [value for value in required if value.casefold() in folded_answer]
        score = len(found) / len(required)
        missing = [value for value in required if value not in found]
        detail = (
            "All required answer facts are present."
            if not missing
            else f"Missing required answer text: {', '.join(missing)}"
        )
        return score, detail

    @staticmethod
    def _grounding(
        *,
        answer: str,
        evidence: Sequence[EvidenceReference],
        citations: Sequence[Citation],
        expectations: EvaluationExpectations,
    ) -> tuple[float, str, bool]:
        evidence_by_id = {item.evidence_id: item for item in evidence}
        citation_integrity = all(
            citation.evidence_id in evidence_by_id for citation in citations
        )
        if expectations.require_citations and not citations:
            citation_integrity = False

        markers = expectations.required_evidence_markers
        if markers:
            cited_text = "\n".join(
                evidence_by_id[citation.evidence_id].excerpt or ""
                for citation in citations
                if citation.evidence_id in evidence_by_id
            )
            found = [marker for marker in markers if marker in cited_text]
            score = len(found) / len(markers)
            if not citation_integrity:
                score = 0.0
            missing = [marker for marker in markers if marker not in found]
            return (
                score,
                "All required evidence markers are cited."
                if not missing and citation_integrity
                else f"Missing cited evidence markers: {', '.join(missing) or 'invalid citation'}",
                citation_integrity,
            )

        if citations and citation_integrity:
            return 1.0, "Citations resolve to evidence captured during this run.", True
        if not evidence:
            return 1.0, "No documentary evidence was required for this response.", citation_integrity
        if not _concrete_tokens(strip_uncertainty_sentences(answer)):
            return 1.0, "The answer only states missing information; nothing to cite.", citation_integrity
        return 0.5, "Evidence exists, but the response does not cite it.", citation_integrity

    @staticmethod
    def _calculator_consistency(
        *,
        answer: str,
        evidence: Sequence[EvidenceReference],
        citations: Sequence[Citation],
    ) -> tuple[bool, str]:
        evidence_by_id = {item.evidence_id: item for item in evidence}
        cited_evidence = [
            evidence_by_id[citation.evidence_id]
            for citation in citations
            if citation.evidence_id in evidence_by_id
        ]
        calculator_items = [
            item
            for item in cited_evidence
            if item.metadata.get("retrieval_tool") == "calculator"
            or item.source_id == "tool:calculator"
        ]
        if not calculator_items:
            return True, "No calculator evidence was cited."

        answer_numbers = _numbers_in_text(answer)
        missing_values: list[str] = []
        for item in calculator_items:
            expected = _decimal_or_none(item.metadata.get("value"))
            if expected is None:
                expected = _calculator_value_from_excerpt(item.excerpt or "")
            if expected is None:
                missing_values.append(item.excerpt or item.source_id)
                continue
            if expected not in answer_numbers:
                missing_values.append(str(expected))

        if missing_values:
            return (
                False,
                "Cited calculator evidence result is absent from the answer: "
                + ", ".join(missing_values),
            )
        return True, "Cited calculator evidence matches numeric answer text."

    @staticmethod
    def _document_claim_support(
        *,
        goal: str = "",
        answer: str,
        evidence: Sequence[EvidenceReference],
        citations: Sequence[Citation],
    ) -> tuple[bool, str]:
        evidence_by_id = {item.evidence_id: item for item in evidence}
        cited_evidence = [
            evidence_by_id[citation.evidence_id]
            for citation in citations
            if citation.evidence_id in evidence_by_id
        ]
        document_items = [
            item
            for item in cited_evidence
            if item.metadata.get("retrieval_tool") == "read_only_file_search"
            or item.metadata.get("source_type", "document") == "document"
        ]
        if not document_items:
            return True, "No document evidence was cited."

        cited_text = canonical_dates("\n".join(item.excerpt or "" for item in document_items))
        answer_body = "\n".join(
            line for line in answer.splitlines() if not line.lstrip().startswith("#")
        )  # markdown headings are formatting, not claims
        answer_for_support = canonical_dates(
            strip_uncertainty_sentences(_strip_rendered_citations(answer_body))
        )
        answer_tokens = _concrete_tokens(answer_for_support)
        if not answer_tokens:
            return True, "The answer contains no concrete claim tokens to check."

        cited_tokens = _concrete_tokens(cited_text)
        supported = sorted(answer_tokens & cited_tokens)
        minimum_overlap = min(len(answer_tokens), max(2, (len(answer_tokens) + 1) // 2))
        # A cited line's document title (e.g. "Cedar Initiative") is part of its provenance.
        titles = " ".join(
            str(item.metadata.get("document_title", "")) for item in document_items
        )
        cited_sequence = re.findall(r"[a-z0-9][a-z0-9_-]*", f"{cited_text} {titles}".casefold())
        # Text extracted from PDFs splits words at line ends ("Edi" + "tion"); a word
        # formed by two adjacent fragments counts as cited.
        cited_words = set(cited_sequence) | {
            first + second for first, second in zip(cited_sequence, cited_sequence[1:])
        }
        missing_entities = sorted(_answer_entity_terms(answer_for_support) - cited_words)
        # Dates and multi-digit numbers are high-risk facts: each must appear in cited
        # evidence (documents or calculator results), however many other words overlap.
        unsupported_dates = sorted(
            set(_ISO_DATE.findall(answer_for_support)) - set(_ISO_DATE.findall(cited_text))
        )
        all_cited_text = canonical_dates("\n".join(item.excerpt or "" for item in cited_evidence))
        cited_numbers = _numbers_for_support(all_cited_text) | {
            part for date in _ISO_DATE.findall(all_cited_text) for part in date.split("-")
        }
        unsupported_dates += sorted(_numbers_for_support(answer_for_support) - cited_numbers)
        # Words that only restate the question prove nothing: what the answer adds (the
        # actual answer) must be at least half supported by the cited text.
        novel = answer_tokens - _concrete_tokens(goal)
        novel_supported = novel & cited_tokens
        novel_unsupported = bool(novel) and len(novel_supported) < (len(novel) + 1) // 2
        if (
            len(supported) < minimum_overlap
            or missing_entities
            or unsupported_dates
            or novel_unsupported
        ):
            details = [
                f"supported concrete terms: {len(supported)}/{len(answer_tokens)}",
            ]
            if novel_unsupported:
                details.append(
                    f"terms beyond the question supported: {len(novel_supported)}/{len(novel)}"
                )
            if missing_entities:
                details.append("missing cited entities: " + ", ".join(missing_entities))
            if unsupported_dates:
                details.append("uncited dates or numbers: " + ", ".join(unsupported_dates))
            return False, "; ".join(details)
        return True, "Cited document evidence supports the answer's concrete terms."

    @staticmethod
    def _plan_adherence(
        plan: TaskPlan, expectations: EvaluationExpectations
    ) -> tuple[float, str]:
        if expectations.expected_plan_steps is not None and (
            len(plan.steps) != expectations.expected_plan_steps
        ):
            return 0.0, (
                f"Expected {expectations.expected_plan_steps} plan steps; "
                f"received {len(plan.steps)}."
            )
        terminal = {StepStatus.COMPLETED, StepStatus.FAILED, StepStatus.SKIPPED}
        completed = sum(step.status in terminal for step in plan.steps)
        score = completed / len(plan.steps)
        return score, f"{completed} of {len(plan.steps)} steps reached a terminal state."

    @staticmethod
    def _tool_correctness(
        *,
        tool_requests: Sequence[ToolRequest],
        tool_results: Sequence[ToolResult],
        permission_decisions: Sequence[PermissionDecision],
        expectations: EvaluationExpectations,
    ) -> tuple[float, str, bool]:
        request_ids = {request.request_id for request in tool_requests}
        decisions = {decision.request_id: decision for decision in permission_decisions}
        permission_integrity = True
        for result in tool_results:
            decision = decisions.get(result.request_id)
            if (
                result.request_id not in request_ids
                or decision is None
                or decision.decided_at > result.started_at
                or (result.success and not decision.allowed)
            ):
                permission_integrity = False
                break

        requested_names = [request.tool_name for request in tool_requests]
        expected_names = list(expectations.expected_tools)
        correct_names = all(name in requested_names for name in expected_names)
        correct_count = (
            expectations.expected_tool_call_count is None
            or len(tool_requests) == expectations.expected_tool_call_count
        )
        denial_seen = any(not decision.allowed for decision in permission_decisions)
        denial_correct = not expectations.expect_denial or denial_seen
        allowed_results_ok = all(
            (not decisions[result.request_id].allowed) or result.success
            for result in tool_results
            if result.request_id in decisions
        )
        passed = all(
            (permission_integrity, correct_names, correct_count, denial_correct, allowed_results_ok)
        )
        detail = (
            "Tool selection, permission ordering, and results are valid."
            if passed
            else "Tool trace violated expected selection, count, permission, denial, or result rules."
        )
        return (1.0 if passed else 0.0), detail, permission_integrity

    @staticmethod
    def _unsupported_claims(
        claims: Sequence[str] | None,
    ) -> tuple[float, str, tuple[str, ...]]:
        if claims is None:
            return 1.0, "No deterministic unsupported claims were supplied.", ()
        normalized = tuple(claim.strip() for claim in claims if claim.strip())
        if not normalized:
            return 1.0, "No unsupported claims were identified by deterministic checks.", ()
        return 0.0, f"Detected {len(normalized)} unsupported claim(s).", normalized

    @staticmethod
    def _usefulness(
        answer: str, expectations: EvaluationExpectations
    ) -> tuple[float, str]:
        if not answer.strip() or len(answer) > 100_000:
            return 0.0, "The answer is empty or exceeds the usefulness bound."
        if expectations.expect_uncertainty and not contains_uncertainty_statement(answer):
            return 0.0, "The answer does not acknowledge missing information."
        return 1.0, "The response is bounded, direct, and handles required uncertainty."


UNCERTAINTY_PHRASES = (
    "cannot be determined",
    "cannot determine",
    "cannot be verified",
    "cannot verify",
    "can't be verified",
    "could not be verified",
    "could not determine",
    "unable to verify",
    "unable to determine",
    "does not verify",
    "do not verify",
    "not verifiable",
    "does not contain",
    "do not contain",
    "does not mention",
    "do not mention",
    "does not include",
    "do not include",
    "does not state",
    "not listed",
    "not mentioned",
    "not specified",
    "not stated",
    "not provided",
    "not available",
    "not found",
    "no evidence",
    "no information",
    "no mention",
    "insufficient evidence",
    "insufficient information",
    "unknown",
)


def contains_uncertainty_statement(text: str) -> bool:
    """The single shared definition of an explicit uncertainty statement."""

    normalized = " ".join(text.casefold().replace("’", "'").split())
    return any(phrase in normalized for phrase in UNCERTAINTY_PHRASES)


def _numbers_in_text(value: str) -> set[Decimal]:
    numbers: set[Decimal] = set()
    for match in re.finditer(r"(?<![\w.])-?(?:\d+(?:\.\d*)?|\.\d+)(?![\w.])", value):
        parsed = _decimal_or_none(match.group(0))
        if parsed is not None:
            numbers.add(parsed)
    return numbers


def _calculator_value_from_excerpt(excerpt: str) -> Decimal | None:
    if "=" not in excerpt:
        return None
    return _decimal_or_none(excerpt.rsplit("=", maxsplit=1)[-1].strip())


def _decimal_or_none(value: object) -> Decimal | None:
    if isinstance(value, bool) or value is None:
        return None
    try:
        parsed = Decimal(str(value))
    except (InvalidOperation, ValueError):
        return None
    return parsed.normalize()


_STOPWORDS = frozenset(
    {
        "a",
        "an",
        "and",
        "are",
        "as",
        "at",
        "be",
        "by",
        "for",
        "from",
        "in",
        "is",
        "it",
        "of",
        "on",
        "or",
        "the",
        "to",
        "was",
        "were",
        "who",
        "with",
        # Common function words (standard stopword-list entries).
        "also",
        "been",
        "but",
        "has",
        "had",
        "have",
        "its",
        "than",
        "that",
        "their",
        "then",
        "there",
        "these",
        "this",
        "those",
        "which",
        "while",
    }
)


def _concrete_tokens(value: str) -> set[str]:
    """Content tokens; words are lightly stemmed so paraphrase ('analysis'/'analyzed')
    matches, while dates and numbers stay exact."""

    tokens: set[str] = set()
    for match in re.finditer(r"[A-Za-z][A-Za-z0-9_-]{2,}|\d{4}-\d{2}-\d{2}|\d+", value):
        token = match.group(0).casefold()
        if token in _STOPWORDS:
            continue
        tokens.add(_stem(token) if token[0].isalpha() else token)
    return tokens


_STEM_SUFFIXES = (
    "ations", "ation", "ions", "ion", "ings", "ing", "ers", "er", "ies", "ied",
    "ed", "es", "s", "e", "ysis", "yzed", "yze",
)


def _stem(term: str) -> str:
    """Generic light suffix stripping (no word-specific rules)."""

    for suffix in sorted(_STEM_SUFFIXES, key=len, reverse=True):
        if len(term) > len(suffix) + 2 and term.endswith(suffix):
            if suffix == "s" and term.endswith(("ss", "us", "is", "as")):
                return term
            if suffix in {"ies", "ied"}:
                return term[: -len(suffix)] + "y"
            return term[: -len(suffix)]
    return term


_SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+|\n+")


def strip_uncertainty_sentences(text: str) -> str:
    """Drop explicit uncertainty sentences; they state an absence, not a factual claim."""

    sentences = [part for part in _SENTENCE_SPLIT.split(text) if part.strip()]
    return " ".join(part for part in sentences if not contains_uncertainty_statement(part))


def _titlecase_terms(value: str) -> set[str]:
    terms: set[str] = set()
    for match in re.finditer(r"\b[A-Z][a-z][A-Za-z0-9_-]*\b", value):
        term = match.group(0)
        if term.casefold() not in _STOPWORDS:
            terms.add(term.casefold())
    return terms


def _answer_entity_terms(value: str) -> set[str]:
    """Capitalized names in the answer, excluding sentence-initial words and headings,
    which are capitalized by grammar or formatting rather than because they are names."""

    terms: set[str] = set()
    for line in value.splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        for sentence in re.split(r"(?<=[.!?:;])\s+", stripped.replace("*", "")):
            words = sentence.lstrip("-•>0123456789.) ").split(maxsplit=1)
            if len(words) == 2:
                terms |= _titlecase_terms(words[1])
    return terms


_MONTHS = {
    name: index
    for index, names in enumerate(
        (
            ("january", "jan"), ("february", "feb"), ("march", "mar"), ("april", "apr"),
            ("may",), ("june", "jun"), ("july", "jul"), ("august", "aug"),
            ("september", "sep", "sept"), ("october", "oct"), ("november", "nov"),
            ("december", "dec"),
        ),
        start=1,
    )
    for name in names
}
_MONTH_PATTERN = "|".join(sorted(_MONTHS, key=len, reverse=True))
_MONTH_DAY_YEAR = re.compile(
    rf"\b({_MONTH_PATTERN})\.?\s+(\d{{1,2}})(?:st|nd|rd|th)?,?\s+(\d{{4}})\b", re.IGNORECASE
)
_DAY_MONTH_YEAR = re.compile(
    rf"\b(\d{{1,2}})(?:st|nd|rd|th)?\s+({_MONTH_PATTERN})\.?,?\s+(\d{{4}})\b", re.IGNORECASE
)


_ISO_DATE = re.compile(r"\b\d{4}-\d{2}-\d{2}\b")
_STANDALONE_NUMBER = re.compile(r"(?<![\w.,-])\d[\d,]*(?:\.\d+)?(?![\w-])")


def _numbers_for_support(text: str) -> set[str]:
    """Multi-digit numbers outside dates (single digits are usually list markers)."""

    numbers: set[str] = set()
    for match in _STANDALONE_NUMBER.finditer(text):
        value = match.group(0).replace(",", "").rstrip(".")
        if len(value.replace(".", "")) >= 2:
            numbers.add(value)
    return numbers


def canonical_dates(value: str) -> str:
    """Rewrite 'February 14, 2025' and '14 February 2025' as '2025-02-14'."""

    def iso(year: str, month: str, day: str) -> str:
        month_number = _MONTHS[month.casefold()]
        return f"{year}-{month_number:02d}-{int(day):02d}" if 1 <= int(day) <= 31 else f"{month} {day} {year}"

    value = _MONTH_DAY_YEAR.sub(lambda m: iso(m.group(3), m.group(1), m.group(2)), value)
    return _DAY_MONTH_YEAR.sub(lambda m: iso(m.group(3), m.group(2), m.group(1)), value)


def _strip_rendered_citations(value: str) -> str:
    return re.sub(r"\[[^\]]+\]", " ", value)
