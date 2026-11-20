"""Authoritative rule-based evaluation public API."""

from .evaluator import (
    UNCERTAINTY_PHRASES,
    EvaluationExpectations,
    RuleBasedEvaluator,
    canonical_dates,
    contains_uncertainty_statement,
    stated_absence_evidence,
    strip_uncertainty_sentences,
    unfinished_total,
    unverified_absence,
)

__version__ = "0.1.0"

__all__ = [
    "UNCERTAINTY_PHRASES",
    "EvaluationExpectations",
    "RuleBasedEvaluator",
    "canonical_dates",
    "contains_uncertainty_statement",
    "stated_absence_evidence",
    "strip_uncertainty_sentences",
    "unfinished_total",
    "unverified_absence",
]
