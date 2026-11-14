"""Off-topic search guard: a query must share a content word with the goal or evidence."""

from __future__ import annotations

import unittest

from agi_agent_runtime.runtime import _content_terms


def is_on_topic(query: str, goal: str, evidence: str = "") -> bool:
    known = _content_terms(goal) | _content_terms(evidence)
    terms = _content_terms(query)
    return not terms or bool(terms & known)


class SearchRelevanceTests(unittest.TestCase):
    GOAL = "Which measurements are consistent with the metric rulers shown in Figure PSS.2?"

    def test_invented_query_is_rejected(self) -> None:
        # Observed live: qwen searched for this while answering the ruler question.
        self.assertFalse(
            is_on_topic("search for files containing the word 'error' in their content", self.GOAL)
        )

    def test_goal_terms_pass_with_inflection_changes(self) -> None:
        self.assertTrue(is_on_topic("Figure PSS.2 metric ruler measurement", self.GOAL))

    def test_bridge_entity_from_evidence_passes(self) -> None:
        # Multi-hop: the second search names something only the evidence mentioned.
        self.assertTrue(
            is_on_topic("Cedar Labs founding year", "Who funds the vendor?", "The vendor is Cedar Labs.")
        )

    def test_function_words_alone_do_not_count_as_overlap(self) -> None:
        self.assertEqual(_content_terms("search the files for their content"), set())


if __name__ == "__main__":
    unittest.main()
