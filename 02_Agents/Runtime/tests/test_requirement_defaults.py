from __future__ import annotations

import unittest

from agi_agent_runtime.requirement_defaults import INFERRED_POLICY_VERSION, infer_task_requirements

TOOLS = ("calculator", "read_only_file_search", "read_only_file_reader")


class RequirementDefaultTests(unittest.TestCase):
    def check(self, goal: str, *, required: list[str], grounded: bool, forbidden: list[str] = []) -> None:
        result = infer_task_requirements(goal, TOOLS)
        self.assertEqual(result.required_capabilities, required, goal)
        self.assertEqual(result.requires_grounding, grounded, goal)
        self.assertEqual(result.requires_citations, grounded, goal)
        self.assertEqual(result.forbidden_capabilities, forbidden, goal)
        self.assertEqual(result.requirement_policy_version, INFERRED_POLICY_VERSION)

    def test_questions_require_grounded_retrieval(self) -> None:
        self.check("Who maintains the payroll export job?", required=["read_only_file_search"], grounded=True)
        self.check("Summarize the onboarding notes", required=["read_only_file_search"], grounded=True)

    def test_arithmetic_requires_calculator_not_grounding(self) -> None:
        self.check("What is 12 squared?", required=["calculator"], grounded=False)
        self.check("Compute 19 * 7", required=["calculator"], grounded=False)
        self.check("Compute 19 * 7 and cite the result", required=["calculator"], grounded=True)

    def test_dates_and_ranges_are_not_arithmetic(self) -> None:
        search = ["read_only_file_search"]
        self.check("When did the 2024-03-15 audit start?", required=search, grounded=True)
        self.check("Is 03/15/2024 the same day as 15 March 2024?", required=search, grounded=True)
        self.check("What do lines 12-18 of the charter say?", required=search, grounded=True)
        # A real subtraction is still arithmetic.
        self.check("What is 2025 - 2019?", required=["calculator"], grounded=False)

    def test_combining_document_quantities_needs_search_then_calculator(self) -> None:
        both = ["read_only_file_search", "calculator"]
        self.check("How many samples did Atlas and Beacon process combined?", required=both, grounded=True)
        self.check("By how many hours did the Beacon trial beat Atlas?", required=both, grounded=True)
        self.check("How many more samples did Atlas process than Beacon?", required=both, grounded=True)
        self.check("How many samples were processed across all three trials?", required=both, grounded=True)
        self.check("What is the difference between the two trial durations?", required=both, grounded=True)

    def test_combining_stated_numbers_needs_only_the_calculator(self) -> None:
        self.check("What is 120 + 90 combined?", required=["calculator"], grounded=False)

    def test_combining_without_a_calculator_still_searches(self) -> None:
        result = infer_task_requirements(
            "How many samples did Atlas and Beacon process combined?", ["read_only_file_search"]
        )
        self.assertEqual(result.required_capabilities, ["read_only_file_search"])
        self.assertTrue(result.requires_grounding)

    def test_plan_requests_need_nothing(self) -> None:
        self.check("Draft a four-step plan to evaluate vendors", required=[], grounded=False)

    def test_destructive_verbs_are_forbidden_even_without_research(self) -> None:
        self.check("Erase notes.md.", required=[], grounded=False, forbidden=["delete_file"])
        self.check(
            "Find the vendor list and then rename it",
            required=["read_only_file_search"],
            grounded=True,
            forbidden=["move_file"],
        )

    def test_no_tools_means_no_required_capabilities(self) -> None:
        result = infer_task_requirements("Who maintains the payroll export job?", [])
        self.assertEqual(result.required_capabilities, [])
        self.assertFalse(result.requires_grounding)


if __name__ == "__main__":
    unittest.main()
