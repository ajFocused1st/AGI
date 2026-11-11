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


class GradingRegressionTests(DateEquivalenceTests):
    """Step 9b: correct checkpoint-B answers must pass; wrong answers must still fail."""

    CEDAR = [
        "[C1] The Cedar Initiative began operations on 2024-09-03.",
        "[C2] During its pilot, Cedar analyzed 240 specimens.",
        "[C3] That pilot took 6 hours to finish.",
        "[C4] Responsibility for the Cedar audit rests with Tomas Reyes.",
    ]

    def _supported_all(self, answer: str) -> bool:
        from uuid import uuid4

        from agi_contracts import Citation, EvidenceReference
        from agi_evaluation.evaluator import RuleBasedEvaluator

        evidence = [
            EvidenceReference(evidence_id=uuid4(), source_id="d", source_path="cedar.md", excerpt=x)
            for x in self.CEDAR
        ]
        supported, _ = RuleBasedEvaluator._document_claim_support(
            answer=answer,
            evidence=evidence,
            citations=[Citation(evidence_id=item.evidence_id, label="c") for item in evidence],
        )
        return supported

    def test_correct_paraphrased_answers_pass(self) -> None:
        for answer in (
            "The Cedar Initiative started operations on September 3rd, 2024. Its pilot phase "
            "lasted for 6 hours and involved the analysis of 240 specimens.",
            "# Summary of Local Documents\n\nThese documents describe the **Cedar Initiative**, "
            "which began operations on September 3, 2024.",
            "The available evidence does not contain the staffing level for Cedar.",
        ):
            self.assertTrue(self._supported_all(answer), answer)

    def test_wrong_facts_still_fail(self) -> None:
        for answer in (
            "Cedar analyzed 999 specimens during its pilot.",
            "The Cedar audit is led by Priya Natarajan.",
            "The Cedar Initiative began operations on March 3, 2023.",
            "The available evidence does not contain the budget. Cedar analyzed 999 specimens.",
        ):
            self.assertFalse(self._supported_all(answer), answer)


class DocumentTitleProvenanceTests(unittest.TestCase):
    """Checkpoint C held-out 06: 'Cedar Initiative' comes from the cited line's document title."""

    def _supported(self, answer: str, title: str | None) -> bool:
        from uuid import uuid4

        from agi_contracts import Citation, EvidenceReference
        from agi_evaluation.evaluator import RuleBasedEvaluator

        evidence = EvidenceReference(
            evidence_id=uuid4(),
            source_id="d:cedar.md",
            source_path="cedar.md",
            excerpt="[C2] During its pilot, Cedar analyzed 240 specimens.",
            metadata={"document_title": title} if title else {},
        )
        supported, _ = RuleBasedEvaluator._document_claim_support(
            answer=answer,
            evidence=[evidence],
            citations=[Citation(evidence_id=evidence.evidence_id, label="c")],
        )
        return supported

    def test_title_words_count_as_cited_provenance(self) -> None:
        answer = "During its pilot, the Cedar Initiative analyzed 240 specimens."
        self.assertTrue(self._supported(answer, "Cedar Initiative"))
        self.assertFalse(self._supported(answer, None))

    def test_wrong_name_still_fails_with_a_title(self) -> None:
        self.assertFalse(
            self._supported("During its pilot, the Cedar Program analyzed 240 specimens.", "Cedar Initiative")
        )


if __name__ == "__main__":
    unittest.main()
