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


class AnswerBeyondQuestionTests(unittest.TestCase):
    """Real textbook run: the answer restated the question and invented its ending."""

    GOAL = "What branch of chemistry studies substances derived from plants and animals?"
    EXCERPT = "The branch of chemistry that studies substances derived from plants and animals"

    def _supported(self, answer: str, excerpt: str) -> bool:
        from uuid import uuid4

        from agi_contracts import Citation, EvidenceReference
        from agi_evaluation.evaluator import RuleBasedEvaluator

        evidence = EvidenceReference(
            evidence_id=uuid4(), source_id="d:book.txt", source_path="book.txt", excerpt=excerpt
        )
        supported, _ = RuleBasedEvaluator._document_claim_support(
            goal=self.GOAL,
            answer=answer,
            evidence=[evidence],
            citations=[Citation(evidence_id=evidence.evidence_id, label="book.txt")],
        )
        return supported

    def test_invented_ending_is_unsupported(self) -> None:
        answer = "The branch of chemistry that studies substances derived from plants and animals is natural products."
        self.assertFalse(self._supported(answer, self.EXCERPT))

    def test_word_split_across_lines_counts_as_cited(self) -> None:
        excerpt = "To address this issue, the Eighth Edi tion of this textbook added Section 6.7."
        answer = "The Eighth Edition of this textbook added Section 6.7 to address this issue."
        self.assertTrue(self._supported(answer, excerpt))

    def test_answer_present_in_the_cited_text_is_supported(self) -> None:
        answer = "The branch of chemistry that studies substances derived from plants and animals is biochemistry."
        self.assertTrue(self._supported(answer, self.EXCERPT + " is biochemistry."))


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


class CorrectAnswerFalseAlarmTests(unittest.TestCase):
    """Real-documents runs (2026-11-15): correct answers were failed; wrong ones must still fail."""

    def _supported(
        self,
        answer: str,
        excerpt: str,
        *,
        goal: str = "",
        path: str = "atlas.md",
        computed: str | None = None,
    ) -> bool:
        from uuid import uuid4

        from agi_contracts import Citation, EvidenceReference
        from agi_evaluation.evaluator import RuleBasedEvaluator

        document = EvidenceReference(
            evidence_id=uuid4(), source_id=f"d:{path}", source_path=path, excerpt=excerpt
        )
        evidence = [document]
        if computed is not None:
            expression, value = computed.split(" = ")
            evidence.append(
                EvidenceReference(
                    evidence_id=uuid4(),
                    source_id="tool:calculator",
                    excerpt=computed,
                    metadata={"retrieval_tool": "calculator", "expression": expression, "value": int(value)},
                    source_path="tool:calculator",
                )
            )
        supported, _ = RuleBasedEvaluator._document_claim_support(
            goal=goal,
            answer=answer,
            evidence=evidence,
            citations=[Citation(evidence_id=document.evidence_id, label=path)],  # documents only
        )
        return supported

    def test_words_about_the_evidence_are_not_claims(self) -> None:
        goal = "Who owns the Atlas review? Use only approved local documents and cite the source."
        excerpt = "[A4] Mina Chen owns the Atlas review."
        lead_in = "The available evidence states that {} owns the Atlas review."
        self.assertTrue(self._supported(lead_in.format("Mina Chen"), excerpt, goal=goal))
        self.assertFalse(self._supported(lead_in.format("Nora Patel"), excerpt, goal=goal))

    def test_the_cited_file_name_is_provenance(self) -> None:
        excerpt = "Nila Orsen, Dispatch Manager, said a reliable ferry is not just a vessel that arrives."
        path = "Northwind_Ferry_Cooperative_Reference.txt"
        answer = "Nila Orsen is the Dispatch Manager of the {} Ferry Cooperative."
        self.assertTrue(self._supported(answer.format("Northwind"), excerpt, path=path))
        self.assertFalse(self._supported(answer.format("Southwind"), excerpt, path=path))

    def test_a_number_the_calculator_computed_is_supported(self) -> None:
        goal = "By how many tickets did Harbour Link's passenger sales increase from 2024 to 2025?"
        excerpt = "Harbour Link recorded 5,730 paid passenger tickets in 2024 and 6,415 in 2025."
        computed = "(6415 - 5730) = 685"
        answer = "Harbour Link's passenger sales increased by {} tickets from 2024 to 2025."
        self.assertTrue(self._supported(answer.format(685), excerpt, goal=goal, computed=computed))
        self.assertFalse(self._supported(answer.format(885), excerpt, goal=goal, computed=computed))
        # Without the calculator's result the number has no support.
        self.assertFalse(self._supported(answer.format(685), excerpt, goal=goal))

    def test_words_framing_a_computed_result_are_not_claims(self) -> None:
        # Live multihop case mh-04: a correct difference was failed on these words.
        goal = "Which pilot analyzed more specimens, Cedar or Delta, and by how many? Use the calculator."
        excerpt = "[C2] During its pilot, Cedar analyzed 240 specimens. [D1] The Delta pilot analyzed 150 specimens."
        answer = (
            "Based on the provided evidence, the Cedar pilot analyzed 240 specimens while the Delta "
            "pilot analyzed 150. The calculated difference between them is {} specimens; that "
            "number favours Cedar."
        )
        computed = "240 - 150 = 90"
        self.assertTrue(self._supported(answer.format(90), excerpt, goal=goal, computed=computed))
        self.assertFalse(self._supported(answer.format(80), excerpt, goal=goal, computed=computed))

    def test_naming_kinds_of_information_is_not_a_claim(self) -> None:
        # Live benchmark v1 case 08: a correct "no such budget" answer was failed.
        goal = "Using only local benchmark documents, state whether Atlas had a $50000 budget."
        excerpt = (
            "[A1] Project Atlas launched on 2025-02-14. [A2] The Atlas trial processed 120 samples. "
            "[A3] The trial completed in 4 hours. [A4] Mina Chen owns the Atlas review."
        )
        right = (
            "There is no mention of a $50,000 budget for Project Atlas. The documents only state "
            "launch date, sample count, completion time, and ownership."
        )
        wrong = "The documents state that Project Atlas had a $50,000 budget approved by Mina Chen."
        self.assertTrue(self._supported(right, excerpt, goal=goal))
        self.assertFalse(self._supported(wrong, excerpt, goal=goal))

    def test_a_year_from_the_question_is_given(self) -> None:
        # Live Veyra Q2: the answer repeated "2025" from the question and was failed.
        goal = "How many field workshops did the Archive conduct during 2025?"
        excerpt = "The Archive conducted thirty-eight field workshops during the reporting year."
        self.assertTrue(self._supported("The Archive conducted thirty-eight field workshops in 2025.", excerpt, goal=goal))
        self.assertFalse(self._supported("The Archive conducted thirty-eight field workshops in 2024.", excerpt, goal=goal))

    def test_a_date_built_from_the_given_year_and_a_retrieved_day(self) -> None:
        # Live Veyra Q1: "December 31, 2025" = the question's year + "on December 31" in evidence.
        from uuid import uuid4

        from agi_contracts import Citation, EvidenceReference
        from agi_evaluation.evaluator import RuleBasedEvaluator

        goal = "How many sealed accession packets did the Archive have at the end of 2025?"
        cited = EvidenceReference(
            evidence_id=uuid4(), source_id="d", source_path="ops.txt",
            excerpt="The permanent working stores contained 14,235 sealed accession packets at the year-end cutoff.",
        )
        cutoff = EvidenceReference(
            evidence_id=uuid4(), source_id="d", source_path="ops.txt",
            excerpt="The records team used a common reporting cutoff of 17:00 on December 31.",
        )

        def supported(answer: str) -> bool:
            ok, _ = RuleBasedEvaluator._document_claim_support(
                goal=goal, answer=answer, evidence=[cited, cutoff],
                citations=[Citation(evidence_id=cited.evidence_id, label="ops")],
            )
            return ok

        self.assertTrue(supported("The stores contained 14,235 sealed accession packets at the cutoff on December 31, 2025."))
        self.assertFalse(supported("The stores contained 14,235 sealed accession packets at the cutoff on December 30, 2025."))
        self.assertFalse(supported("The stores contained 14,235 sealed accession packets at the cutoff on December 31, 2024."))

    def test_month_names_are_not_treated_as_names(self) -> None:
        # Live Veyra Q31: "December" was flagged as a missing entity.
        excerpt = "The preliminary figure of 94 recorded in the December 16, 2025 minutes is replaced by 89."
        self.assertTrue(self._supported("The final figure is 89, replacing the 94 recorded in December 2025.", excerpt))
        self.assertFalse(self._supported("The final figure is 89, as approved by Priya Natarajan.", excerpt))

    def test_source_references_in_prose_are_not_claims(self) -> None:
        # Live Veyra Q39: "According to ... annual_operations_2025.txt (lines 158-174), ...".
        goal = "How many new sleeves were issued during the third quarter?"
        excerpt = "TABLE 2. Quarter Repair labels New sleeves Inspection ties Q3 74 111 39"
        answer = (
            "According to the evidence provided in Table 2 of annual_operations_2025.txt "
            "(lines 158-174), the number of new sleeves issued during the third quarter was {}."
        )
        self.assertTrue(self._supported(answer.format(111), excerpt, goal=goal))
        self.assertFalse(self._supported(answer.format(141), excerpt, goal=goal))

    def test_a_sentence_after_a_closing_quote_starts_with_a_capital(self) -> None:
        # Live Veyra Q5: "... amber tabs.' Additionally, ..." made "Additionally" a name.
        excerpt = "The official movement ledger uses amber tabs. The committee retained amber."
        answer = "The handbook states 'The official movement ledger uses amber tabs.' Additionally, the committee retained amber."
        self.assertTrue(self._supported(answer, excerpt))

    def test_arithmetic_shown_in_the_answer_is_recomputed(self) -> None:
        # Live Veyra Q16/Q17: the model added and multiplied cited numbers in the open.
        table = "Q1 52 96 31 Q2 68 104 35 Q3 74 111 39 Q4 63 88 33"
        sum_answer = "Repair labels per quarter were Q1=52, Q2=68, Q3=74 and Q4=63, a total of {}."
        self.assertTrue(self._supported(sum_answer.format(257), table))
        self.assertFalse(self._supported(sum_answer.format(267), table))
        crates = "The cache contains twelve training crates. Each crate holds eighteen demonstration trays."
        product = "The cache has 12 crates of 18 trays each, so it holds {} demonstration trays."
        self.assertTrue(self._supported(product.format(216), crates))
        self.assertFalse(self._supported(product.format(226), crates))
        # A total with no operands shown is still unverified.
        self.assertFalse(self._supported("The repair labels total 257.", table))

    def test_numbers_with_thousands_separators_are_one_number(self) -> None:
        excerpt = "Harbor Line recorded 12,480 tickets in 2024 and 14,235 in 2025."
        computed = "(12480 + 14235) = 26715"
        answer = "Harbor Line sold {} tickets across 2024 and 2025 combined."
        self.assertTrue(self._supported(answer.format("26,715"), excerpt, computed=computed))
        self.assertFalse(self._supported(answer.format("26,815"), excerpt, computed=computed))

    def test_cited_calculator_result_matches_a_separated_number(self) -> None:
        from uuid import uuid4

        from agi_contracts import Citation, EvidenceReference
        from agi_evaluation.evaluator import RuleBasedEvaluator

        result = EvidenceReference(
            evidence_id=uuid4(),
            source_id="tool:calculator",
            excerpt="8970 + 3455 + 6415 + 14235 = 33075",
            metadata={"retrieval_tool": "calculator", "value": 33075},
            source_path="tool:calculator",
        )
        citations = [Citation(evidence_id=result.evidence_id, label="calculator")]
        for answer, expected in (("The total is 33,075 tickets.", True), ("The total is 33,175 tickets.", False)):
            consistent, _ = RuleBasedEvaluator._calculator_consistency(
                answer=answer, evidence=[result], citations=citations
            )
            self.assertEqual(consistent, expected, answer)


class UnverifiedAbsenceTests(unittest.TestCase):
    """A "not found" answer cannot be confirmed unless the document itself says so."""

    def _absence(self, goal: str, answer: str, *excerpts: str) -> str | None:
        from uuid import uuid4

        from agi_contracts import EvidenceReference
        from agi_evaluation import unverified_absence

        evidence = [
            EvidenceReference(evidence_id=uuid4(), source_id="d", source_path="doc.txt", excerpt=text)
            for text in excerpts
        ]
        return unverified_absence(goal, answer, evidence)

    def test_not_found_after_a_missed_search_is_unverified(self) -> None:
        # Real first-run answer: the passage existed, the search missed it.
        goal = "Which service is the scheduled Bellwether Quay to Reedhaven crossing, and how long does it take?"
        answer = (
            "The available evidence does not contain the scheduled Bellwether Quay to Reedhaven "
            "crossing service name or duration."
        )
        self.assertIsNotNone(self._absence(goal, answer, "Reedhaven Loop is a circular route."))

    def test_absence_the_document_states_is_verified(self) -> None:
        goal = "What is the private home telephone number of Mara Ellery?"
        answer = "The available evidence does not contain the private home telephone number of Mara Ellery."
        stated = "Private employee home telephone numbers are outside the scope of this report."
        self.assertIsNone(self._absence(goal, answer, "Mara Ellery leads the review.", stated))
        self.assertIsNotNone(self._absence(goal, answer, "Mara Ellery leads the review."))

    def test_a_side_remark_about_something_not_asked_is_ignored(self) -> None:
        goal = "Who leads the Night Berth Renewal review?"
        answer = "Mara Ellery leads the Night Berth Renewal review. The evidence does not state her salary."
        self.assertIsNone(self._absence(goal, answer, "Mara Ellery leads the review."))

    def test_a_plain_answer_has_nothing_to_verify(self) -> None:
        self.assertIsNone(self._absence("Who owns Atlas?", "Mina Chen owns the Atlas review.", "x"))


if __name__ == "__main__":
    unittest.main()
