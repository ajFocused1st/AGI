from __future__ import annotations

import unittest

from agi_agent_runtime.runtime import FALLBACK_NOTE

from operator_research_engine.answer_key import (
    across_runs,
    is_correct,
    load_dataset,
    score,
    summarize,
)

ITEM = {"id": 1, "category": "direct", "question": "q", "answer": "14,235", "accept": ["14235"]}
ABSENT = {"id": 2, "category": "not-in-documents", "question": "q", "answer": "Not stated",
          "not_in_documents": True}


class AnswerKeyScoringTests(unittest.TestCase):
    def test_number_with_or_without_separator_is_right(self) -> None:
        self.assertTrue(is_correct(ITEM, "There were 14235 packets."))
        self.assertTrue(is_correct(ITEM, "There were 14,235 packets."))
        self.assertFalse(is_correct(ITEM, "There were 8,735 packets."))

    def test_passages_shown_in_place_of_an_answer_never_count(self) -> None:
        self.assertFalse(is_correct(ITEM, FALLBACK_NOTE + "\n\n14,235 sealed packets"))

    def test_absent_fact_needs_a_statement_that_it_is_missing(self) -> None:
        self.assertTrue(is_correct(ABSENT, "None of the text mentions the supplier."))
        self.assertTrue(is_correct(ABSENT, "Pay grades are withheld from these documents."))
        self.assertFalse(is_correct(ABSENT, "The supplier was Harlow Ltd."))

    def test_false_pass_and_safe_miss(self) -> None:
        self.assertTrue(score(ITEM, "It was 8,735.", "passed", None)["false_pass"])
        self.assertTrue(score(ITEM, "It was 14,235.", "partial", None)["safe_miss"])
        row = score(ITEM, "It was 14,235.", "passed", None)
        self.assertFalse(row["false_pass"] or row["safe_miss"])

    def test_summary_and_runs(self) -> None:
        first = [score(ITEM, "14,235", "passed", None), score(ABSENT, "It was Harlow.", "passed", None)]
        second = [score(ITEM, "8,735", "failed", None), score(ABSENT, "Not stated.", "partial", None)]
        summary = summarize(first)
        self.assertEqual(summary["right"], 1)
        self.assertEqual(summary["false_passes"], [2])
        combined = across_runs([first, second])
        self.assertEqual(combined["right_per_run"], [1, 1])
        self.assertEqual(combined["sometimes_right"], [1, 2])
        self.assertEqual(combined["ever_false_pass"], [2])

    def test_veyra_dataset_has_its_documents_and_profile(self) -> None:
        dataset = load_dataset("veyra_v1")
        self.assertEqual(len(dataset.key), 40)
        self.assertTrue(dataset.profile.exists())
        self.assertEqual(len(list(dataset.documents.glob("*.txt"))), 3)


if __name__ == "__main__":
    unittest.main()
