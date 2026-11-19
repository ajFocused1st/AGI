"""Passage retrieval and ranking of read_only_file_search, on non-benchmark fixtures."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from agi_tools import ReadOnlyFileSearch, ReadOnlyFileSearchConfig, SearchRoot
from agi_tools.file_search import FileSearchOutput, ReadOnlyFileSearchInput


class RankingTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self) -> None:
        self._temporary = tempfile.TemporaryDirectory()
        self.root = Path(self._temporary.name)
        (self.root / "harbor.md").write_text(
            "# Harbor Depot\n\n"
            "[H1] Harbor Depot opened in 2019.\n"
            "During its first audit, Harbor shipped 400 crates.\n"
            "That audit lasted 2 days.\n",
            encoding="utf-8",
        )
        (self.root / "summit.md").write_text(
            "# Summit Depot\n\n"
            "Summit Depot shipped 250 crates in its first audit.\n"
            "That audit lasted 3 days.\n",
            encoding="utf-8",
        )
        self.tool = self.make_tool()

    def tearDown(self) -> None:
        self._temporary.cleanup()

    def make_tool(self, **overrides) -> ReadOnlyFileSearch:
        return ReadOnlyFileSearch(
            ReadOnlyFileSearchConfig(roots=(SearchRoot(label="docs", path=self.root),), **overrides)
        )

    async def run_search(self, query: str, tool: ReadOnlyFileSearch | None = None) -> FileSearchOutput:
        return await (tool or self.tool).run(ReadOnlyFileSearchInput(query=query))

    async def search(self, query: str) -> list[str]:
        return [match.snippet for match in (await self.run_search(query)).matches]

    # --- a match brings the text around it ---------------------------------------

    async def test_short_section_is_returned_whole_with_its_line_span(self) -> None:
        match = (await self.run_search("When did Harbor Depot open?")).matches[0]
        self.assertEqual(
            match.snippet,
            "[H1] Harbor Depot opened in 2019. During its first audit, Harbor shipped 400 crates. "
            "That audit lasted 2 days.",
        )
        self.assertEqual((match.line_start, match.line_end, match.line_number), (3, 5, 3))

    async def test_plain_text_first_line_is_the_document_title(self) -> None:
        (self.root / "ferry.txt").write_text(
            "NORTHWIND FERRY COOPERATIVE\nInternal reference\n\nThe Archive ships 90 crates a week.\n",
            encoding="utf-8",
        )
        (self.root / "prose.txt").write_text("This file starts with a sentence. It has no title.\n", encoding="utf-8")
        titles = {m.relative_path: m.document_title for m in (await self.run_search("crates week sentence")).matches}
        self.assertEqual(titles.get("ferry.txt"), "NORTHWIND FERRY COOPERATIVE")
        self.assertIsNone(titles.get("prose.txt"))
        # Markdown documents keep their heading as the title.
        harbor = (await self.run_search("When did Harbor Depot open?")).matches[0]
        self.assertEqual(harbor.document_title, "Harbor Depot")

    async def test_question_carries_its_solution(self) -> None:
        (self.root / "exercises.txt").write_text(
            "Which gauge readings are consistent with the meter?\n"
            "(a) 2 bar, 2.0 bar\n"
            "Solution\n"
            "The meter can give 2.0 bar.\n",
            encoding="utf-8",
        )
        match = (await self.run_search("gauge readings consistent meter")).matches[0]
        self.assertEqual(
            match.snippet,
            "Which gauge readings are consistent with the meter? (a) 2 bar, 2.0 bar Solution "
            "The meter can give 2.0 bar.",
        )
        self.assertEqual((match.line_start, match.line_end), (1, 4))

    async def test_wrapped_sentence_is_read_across_lines(self) -> None:
        # PDF-extracted text: the answer ("is geology") sits on the next line.
        (self.root / "wrapped.txt").write_text(
            "Intro text ends here.\n"
            "The branch of science that studies rocks and minerals\n"
            "is geology. Another branch studies oceans.\n",
            encoding="utf-8",
        )
        match = (await self.run_search("branch science studies rocks minerals")).matches[0]
        self.assertIn("studies rocks and minerals is geology.", match.snippet)

    async def test_quotation_carries_its_attribution(self) -> None:
        (self.root / "quote.txt").write_text(
            "“Measure twice,\n"
            "cut once.”\n"
            "Ada Fielding,\n"
            "Welsh Carpenter (1801–1870)\n",
            encoding="utf-8",
        )
        match = (await self.run_search("measure twice")).matches[0]
        self.assertIn("cut once.” Ada Fielding, Welsh Carpenter (1801–1870)", match.snippet)

    # --- long documents are divided into passages ---------------------------------

    async def test_long_paragraph_is_divided_at_sentence_ends(self) -> None:
        sentences = [f"Fact {number} says the {name} valve was checked." for number, name in enumerate(
            ["amber", "birch", "cedar", "delta", "ember", "flint", "grove", "heron"] * 8, start=1
        )]
        (self.root / "long.txt").write_text(" ".join(sentences) + "\n", encoding="utf-8")
        output = await self.run_search("Fact 40 valve")
        self.assertGreater(len(output.matches), 1)
        for match in output.matches:
            if match.relative_path != "long.txt":
                continue
            self.assertLessEqual(len(match.snippet), 1_200)
            self.assertTrue(match.snippet.startswith("Fact "), match.snippet[:40])
            self.assertTrue(match.snippet.endswith("was checked."), match.snippet[-40:])
            self.assertEqual((match.line_start, match.line_end), (1, 1))
        self.assertIn("Fact 40 says", output.matches[0].snippet)

    async def test_wrapped_long_text_reports_the_lines_each_passage_spans(self) -> None:
        words = " ".join(
            f"Sentence {number} about the {name} reactor ends here."
            for number, name in enumerate(["argon", "boron", "cobalt", "xenon"] * 20, start=1)
        ).split()
        lines = [" ".join(words[index : index + 6]) for index in range(0, len(words), 6)]
        (self.root / "wrapped_long.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
        output = await self.run_search("Sentence 77 reactor")
        best = output.matches[0]
        self.assertIn("Sentence 77 about", best.snippet)
        self.assertGreater(best.line_end, best.line_start)
        covered = " ".join(lines[best.line_start - 1 : best.line_end])
        self.assertIn(best.snippet, covered)  # the reported lines contain the passage

    async def test_paragraphs_separated_by_blank_lines_stay_separate_passages(self) -> None:
        paragraphs = [
            f"SECTION {number}\n" + f"The {name} ferry carries {number * 11} cars on each crossing. " * 12
            for number, name in enumerate(["osprey", "kestrel", "plover", "gannet"], start=1)
        ]
        (self.root / "sections.txt").write_text("\n\n".join(paragraphs) + "\n", encoding="utf-8")
        best = (await self.run_search("kestrel ferry cars")).matches[0]
        self.assertTrue(best.snippet.startswith("SECTION 2 The kestrel ferry"))
        self.assertNotIn("osprey", best.snippet)
        self.assertNotIn("plover", best.snippet)

    async def test_short_question_before_a_break_is_repeated_with_its_answer(self) -> None:
        filler = "The archive room is cleaned weekly by the night crew. " * 14
        (self.root / "faq.txt").write_text(
            filler.strip() + "\n\nWhat colour is the Tolland beacon?\n\n"
            "It is violet, and it has been since the refit. " + filler,
            encoding="utf-8",
        )
        snippets = await self.search("Tolland beacon colour")
        self.assertTrue(
            any("What colour is the Tolland beacon? It is violet" in snippet for snippet in snippets),
            snippets,
        )

    # --- ranking -----------------------------------------------------------------

    async def test_heading_context_matches_passages_that_omit_the_subject(self) -> None:
        output = await self.run_search("How long did the Summit audit last?")
        self.assertEqual(output.matches[0].relative_path, "summit.md")
        self.assertIn("That audit lasted 3 days.", output.matches[0].snippet)

    async def test_best_passage_comes_first(self) -> None:
        output = await self.run_search("When did Harbor Depot open?")
        self.assertEqual(output.matches[0].relative_path, "harbor.md")
        self.assertIn("[H1] Harbor Depot opened in 2019.", output.matches[0].snippet)

    async def test_function_words_do_not_match(self) -> None:
        (self.root / "to.txt").write_text("We went to the shop to buy it.\n", encoding="utf-8")
        output = await self.run_search("How to get to it?")
        self.assertEqual(output.matches, [])

    async def test_weak_matches_are_dropped(self) -> None:
        (self.root / "noise.txt").write_text(
            "The depot canteen menu changes weekly and nobody minds.\n", encoding="utf-8"
        )
        paths = {match.relative_path for match in (await self.run_search("Harbor Depot crates shipped audit")).matches}
        self.assertIn("harbor.md", paths)
        self.assertNotIn("noise.txt", paths)

    async def test_results_stop_at_the_character_budget(self) -> None:
        for number in range(6):
            (self.root / f"pier{number}.txt").write_text(
                f"Pier {number} report. " + "The pier lamp was inspected and the pier rail was painted. " * 10,
                encoding="utf-8",
            )
        output = await self.run_search("pier lamp rail", self.make_tool(max_result_chars=1_500))
        self.assertTrue(output.matches)
        self.assertLessEqual(sum(len(match.snippet) for match in output.matches[1:]), 1_500)
        self.assertLess(len(output.matches), 6)
        self.assertTrue(output.truncated)

    async def test_comparison_keeps_both_sides_and_both_facts(self) -> None:
        text = " ".join(await self.search("Compare Harbor and Summit crates shipped and audit durations"))
        for expected in (
            "During its first audit, Harbor shipped 400 crates.",
            "That audit lasted 2 days.",
            "Summit Depot shipped 250 crates in its first audit.",
            "That audit lasted 3 days.",
        ):
            self.assertIn(expected, text)

    async def test_fact_question_with_no_matching_terms_returns_nothing(self) -> None:
        self.assertEqual(await self.search("What is the Zephyr warranty period?"), [])

    async def test_summary_request_returns_an_overview(self) -> None:
        text = " ".join(await self.search("Summarize the local documents"))
        self.assertIn("Harbor Depot opened in 2019.", text)
        self.assertIn("Summit Depot shipped 250 crates", text)

    async def test_matches_carry_their_document_title(self) -> None:
        output = await self.run_search("How long did the Summit audit last?")
        self.assertEqual(output.matches[0].document_title, "Summit Depot")

    async def test_passages_rank_before_headings(self) -> None:
        snippets = await self.search("Harbor")
        self.assertFalse(snippets[0].startswith("#"))
        self.assertIn("# Harbor Depot", snippets)


if __name__ == "__main__":
    unittest.main()
