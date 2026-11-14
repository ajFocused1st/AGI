"""Ranking behavior of read_only_file_search (step 9), on a non-benchmark fixture."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from agi_tools import ReadOnlyFileSearch, ReadOnlyFileSearchConfig, SearchRoot
from agi_tools.file_search import ReadOnlyFileSearchInput


class RankingTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self) -> None:
        self._temporary = tempfile.TemporaryDirectory()
        root = Path(self._temporary.name)
        (root / "harbor.md").write_text(
            "# Harbor Depot\n\n"
            "[H1] Harbor Depot opened in 2019.\n"
            "During its first audit, Harbor shipped 400 crates.\n"
            "That audit lasted 2 days.\n",
            encoding="utf-8",
        )
        (root / "summit.md").write_text(
            "# Summit Depot\n\n"
            "Summit Depot shipped 250 crates in its first audit.\n"
            "That audit lasted 3 days.\n",
            encoding="utf-8",
        )
        self.tool = ReadOnlyFileSearch(
            ReadOnlyFileSearchConfig(roots=(SearchRoot(label="docs", path=root),))
        )

    def tearDown(self) -> None:
        self._temporary.cleanup()

    async def search(self, query: str) -> list[str]:
        output = await self.tool.run(ReadOnlyFileSearchInput(query=query))
        return [match.snippet for match in output.matches]

    async def test_question_line_carries_its_solution(self) -> None:
        (Path(self._temporary.name) / "exercises.txt").write_text(
            "Which gauge readings are consistent with the meter?\n"
            "(a) 2 bar, 2.0 bar\n"
            "Solution\n"
            "The meter can give 2.0 bar.\n"
            "Which gauge readings fit the dial?\n"
            "(a) 4 bar\n",
            encoding="utf-8",
        )
        output = await self.tool.run(ReadOnlyFileSearchInput(query="gauge readings consistent meter"))
        match = output.matches[0]
        self.assertEqual(
            match.snippet,
            "Which gauge readings are consistent with the meter? (a) 2 bar, 2.0 bar Solution "
            "The meter can give 2.0 bar.",
        )
        self.assertEqual((match.line_start, match.line_end), (1, 4))  # stops before next question

    async def test_statement_lines_get_no_extra_context(self) -> None:
        output = await self.tool.run(ReadOnlyFileSearchInput(query="When did Harbor Depot open?"))
        self.assertEqual(output.matches[0].line_end, output.matches[0].line_start)

    async def test_heading_context_matches_lines_that_omit_the_subject(self) -> None:
        self.assertIn("That audit lasted 3 days.", await self.search("How long did the Summit audit last?"))

    async def test_specific_question_returns_only_the_best_line(self) -> None:
        self.assertEqual(await self.search("When did Harbor Depot open?"), ["[H1] Harbor Depot opened in 2019."])

    async def test_comparison_keeps_both_sides_and_both_facts(self) -> None:
        snippets = await self.search("Compare Harbor and Summit crates shipped and audit durations")
        for expected in (
            "During its first audit, Harbor shipped 400 crates.",
            "That audit lasted 2 days.",
            "Summit Depot shipped 250 crates in its first audit.",
            "That audit lasted 3 days.",
        ):
            self.assertIn(expected, snippets)

    async def test_fact_question_with_no_matching_terms_returns_nothing(self) -> None:
        self.assertEqual(await self.search("What is the Zephyr warranty period?"), [])

    async def test_summary_request_returns_an_overview(self) -> None:
        self.assertGreaterEqual(len(await self.search("Summarize the local documents")), 4)

    async def test_matches_carry_their_document_title(self) -> None:
        output = await self.tool.run(ReadOnlyFileSearchInput(query="How long did the Summit audit last?"))
        self.assertEqual(output.matches[0].document_title, "Summit Depot")

    async def test_content_lines_rank_before_headings(self) -> None:
        snippets = await self.search("Harbor")
        self.assertFalse(snippets[0].startswith("#"))
        self.assertIn("# Harbor Depot", snippets)


if __name__ == "__main__":
    unittest.main()
