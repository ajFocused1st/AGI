from __future__ import annotations

import unittest

from agi_evaluation import contains_uncertainty_statement


class SharedUncertaintyDetectorTests(unittest.TestCase):
    def test_real_checkpoint_answers_are_recognized(self) -> None:
        # Rejected by the old benchmark list but accepted by requirements.py (checkpoint A).
        self.assertTrue(
            contains_uncertainty_statement(
                "The available evidence does not verify whether Cedar had 12 staff members."
            )
        )
        self.assertTrue(
            contains_uncertainty_statement(
                "The available evidence does not contain a launch date for Project Atlas."
            )
        )
        self.assertTrue(contains_uncertainty_statement("The budget can’t be verified."))

    def test_plain_factual_answers_are_not_uncertain(self) -> None:
        for answer in (
            "Project Atlas launched on 2025-02-14.",
            "I cannot delete files.",
            "Mina Chen owns the Atlas review.",
        ):
            self.assertFalse(contains_uncertainty_statement(answer), answer)


class DateEquivalenceTests(unittest.TestCase):
    """Real qwen3.5:9b answer from step 8: same date restated in words was flagged unsupported."""

    def _supported(self, answer: str, excerpt: str) -> bool:
        from uuid import uuid4

        from agi_contracts import Citation, EvidenceReference
        from agi_evaluation.evaluator import RuleBasedEvaluator

        evidence = EvidenceReference(
            evidence_id=uuid4(), source_id="d:atlas.md", source_path="atlas.md", excerpt=excerpt
        )
        supported, _ = RuleBasedEvaluator._document_claim_support(
            answer=answer,
            evidence=[evidence],
            citations=[Citation(evidence_id=evidence.evidence_id, label="atlas.md")],
        )
        return supported

    def test_restated_dates_are_supported_in_both_directions(self) -> None:
        excerpt = "[A1] Project Atlas launched on 2025-02-14."
        self.assertTrue(self._supported("Project Atlas launched on February 14, 2025.", excerpt))
        self.assertTrue(self._supported("Project Atlas launched on 14 Feb 2025.", excerpt))
        self.assertTrue(
            self._supported(
                "Project Atlas launched on 2025-02-14.", "Project Atlas launched on February 14, 2025."
            )
        )

    def test_a_different_date_is_still_unsupported(self) -> None:
        excerpt = "[A1] Project Atlas launched on 2025-02-14."
        self.assertFalse(self._supported("Project Atlas launched on March 3, 2024.", excerpt))


if __name__ == "__main__":
    unittest.main()
