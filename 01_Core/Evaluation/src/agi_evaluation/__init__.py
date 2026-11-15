"""Authoritative rule-based evaluation public API."""

from .evaluator import (
    UNCERTAINTY_PHRASES,
    EvaluationExpectations,
    RuleBasedEvaluator,
    canonical_dates,
    contains_uncertainty_statement,
    strip_uncertainty_sentences,
    unverified_absence,
)

__version__ = "0.1.0"

__all__ = [
    "UNCERTAINTY_PHRASES",
    "EvaluationExpectations",
    "RuleBasedEvaluator",
    "canonical_dates",
    "contains_uncertainty_statement",
    "strip_uncertainty_sentences",
    "unverified_absence",
]
