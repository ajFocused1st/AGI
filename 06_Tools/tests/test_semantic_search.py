"""Meaning-based ranking in read_only_file_search, with a fake embedder (no network)."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from agi_tools import (
    EmbeddingUnavailableError,
    OllamaPassageEmbedder,
    ReadOnlyFileSearch,
    ReadOnlyFileSearchConfig,
    SearchRoot,
)
from agi_tools.file_search import ReadOnlyFileSearchInput

CHARTER = "The organization adopted its first charter on 2023-04-09 at the harbor hall."
LIGHTING = "The spring review showed that new terminal lighting improved night boarding."
FLEET = "The fleet has three vessels; Morrow Bell seats sixty-two passengers."


class FakeEmbedder:
    """Scores passages by which marker words they contain; can be made to fail."""

    def __init__(self, scores: dict[str, float], fail: bool = False) -> None:
        self.scores = scores
        self.fail = fail
        self.calls = 0

    def ensure(self, texts, *, deadline=None) -> int:
        return 0

    def similarities(self, query, texts, *, deadline=None):
        self.calls += 1
        if self.fail:
            raise EmbeddingUnavailableError("model is not running")
        return [max((s for word, s in self.scores.items() if word in text), default=0.2) for text in texts]


class MeaningRankingTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self) -> None:
        self._temporary = tempfile.TemporaryDirectory()
        root = Path(self._temporary.name)
        # Separate sections, so each statement is its own passage.
        (root / "notes.md").write_text(
            f"# Charter\n\n{CHARTER}\n\n# Lighting\n\n{LIGHTING}\n\n# Fleet\n\n{FLEET}\n", encoding="utf-8"
        )
        self.config = ReadOnlyFileSearchConfig(roots=(SearchRoot(label="docs", path=root),))

    def tearDown(self) -> None:
        self._temporary.cleanup()

    async def search(self, query: str, embedder=None):
        tool = ReadOnlyFileSearch(self.config, embedder)
        return await tool.run(ReadOnlyFileSearchInput(query=query))

    async def test_a_reworded_question_finds_its_passage_by_meaning(self) -> None:
        # No word of the question appears in the charter passage.
        question = "When was the founding document approved?"
        keyword_only = await self.search(question)
        self.assertNotIn(CHARTER, [m.snippet for m in keyword_only.matches])
        self.assertEqual(keyword_only.ranking, "keyword")

        found = await self.search(question, FakeEmbedder({"charter": 0.9}))
        self.assertEqual(found.matches[0].snippet, CHARTER)
        self.assertEqual(found.ranking, "keyword+meaning")

    async def test_keyword_best_match_stays_first(self) -> None:
        # The meaning model prefers the fleet passage, but the exact-word match leads.
        found = await self.search(
            "What did the review find about terminal lighting?", FakeEmbedder({"fleet": 0.9, "lighting": 0.7})
        )
        snippets = [m.snippet for m in found.matches]
        self.assertEqual(snippets[0], LIGHTING)
        self.assertIn(FLEET, snippets)

    async def test_weak_meaning_matches_are_left_out(self) -> None:
        # Within 0.15 of the best meaning score only: the 0.2 background passages are dropped.
        found = await self.search("When was the founding document approved?", FakeEmbedder({"charter": 0.9}))
        self.assertEqual([m.snippet for m in found.matches], [CHARTER])

    async def test_an_unavailable_model_falls_back_to_keywords(self) -> None:
        embedder = FakeEmbedder({}, fail=True)
        found = await self.search("What did the review find about terminal lighting?", embedder)
        self.assertEqual(found.matches[0].snippet, LIGHTING)
        self.assertEqual(found.ranking, "keyword")
        self.assertEqual(embedder.calls, 1)


class EmbedderCacheTests(unittest.TestCase):
    def test_passages_are_embedded_once_and_cached_on_disk(self) -> None:
        with tempfile.TemporaryDirectory() as folder:
            cache = Path(folder) / "vectors.json"
            sent: list[list[str]] = []

            def fake_embed(self, inputs):
                sent.append(list(inputs))
                return [[1.0, float(len(text) % 7)] for text in inputs]

            original = OllamaPassageEmbedder._embed
            OllamaPassageEmbedder._embed = fake_embed  # type: ignore[method-assign]
            try:
                first = OllamaPassageEmbedder(base_url="http://x", model="nomic-embed-text", cache_path=cache)
                self.assertEqual(first.ensure(["alpha passage", "beta passage"]), 2)
                self.assertEqual(sent[0], ["search_document: alpha passage", "search_document: beta passage"])
                # A new instance reads the cache: nothing is embedded again.
                second = OllamaPassageEmbedder(base_url="http://x", model="nomic-embed-text", cache_path=cache)
                self.assertEqual(second.ensure(["alpha passage", "beta passage"]), 0)
                scores = second.similarities("question", ["alpha passage", "gamma passage"])
                self.assertEqual(sent[-1], ["search_query: question"])
                self.assertIsNotNone(scores[0])
                self.assertIsNotNone(scores[1])  # newly embedded on demand
                # A cache written for another model is ignored.
                other = OllamaPassageEmbedder(base_url="http://x", model="other-model", cache_path=cache)
                self.assertEqual(other.ensure(["alpha passage"]), 1)
            finally:
                OllamaPassageEmbedder._embed = original  # type: ignore[method-assign]
            self.assertEqual(json.loads(cache.read_text(encoding="utf-8"))["model"], "other-model")

    def test_an_unreachable_server_raises_a_clear_error(self) -> None:
        embedder = OllamaPassageEmbedder(
            base_url="http://127.0.0.1:9", model="nomic-embed-text", cache_path=Path(tempfile.gettempdir()) / "x.json",
            timeout_seconds=2,
        )
        with self.assertRaises(EmbeddingUnavailableError):
            embedder._embed(["text"])


if __name__ == "__main__":
    unittest.main()
