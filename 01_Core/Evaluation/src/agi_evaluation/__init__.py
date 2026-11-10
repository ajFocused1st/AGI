"""Authoritative rule-based evaluation public API."""

from .evaluator import (
    UNCERTAINTY_PHRASES,
    EvaluationExpectations,
    RuleBasedEvaluator,
    contains_uncertainty_statement,
)

__version__ = "0.1.0"

__all__ = [
    "UNCERTAINTY_PHRASES",
    "EvaluationExpectations",
    "RuleBasedEvaluator",
    "contains_uncertainty_statement",
]
