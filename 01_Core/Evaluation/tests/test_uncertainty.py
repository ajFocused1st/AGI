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


if __name__ == "__main__":
    unittest.main()
