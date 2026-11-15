from __future__ import annotations

import hashlib
import os
import stat
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from agi_contracts import ToolRequest
from agi_tools import (
    FileSearchMatch,
    ReadOnlyFileSearch,
    ReadOnlyFileSearchConfig,
    SearchRoot,
    ToolRegistry,
)
from pydantic import ValidationError


class ReadOnlyFileSearchTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self) -> None:
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary_directory.name)
        self.notes_bytes = (
            "Project Atlas launched on 2025-02-14.\n"
            "Atlas processed 120 samples in four hours.\n"
            "Mina Chen owns the Project Atlas review.\n"
        ).encode()
        (self.root / "notes.md").write_bytes(self.notes_bytes)
        (self.root / "credentials.json").write_text(
            '{"sentinel": "Project Atlas secret"}',
            encoding="utf-8",
        )
        (self.root / ".hidden.md").write_text(
            "Project Atlas hidden",
            encoding="utf-8",
        )
        (self.root / "ignored.py").write_text(
            "Project Atlas executable source",
            encoding="utf-8",
        )
        nested = self.root / "nested"
        nested.mkdir()
        (nested / "comparison.txt").write_text(
            "Project Beacon processed 90 samples.\n",
            encoding="utf-8",
        )
        (nested / "beacon.md").write_text(
            "Imani Cole owns the Beacon review.\n",
            encoding="utf-8",
        )
        (nested / "timing.md").write_text(
            "The Beacon trial completed in 3 hours.\n",
            encoding="utf-8",
        )

    def tearDown(self) -> None:
        self.temporary_directory.cleanup()

    def make_tool(self, **overrides) -> ReadOnlyFileSearch:
        values = {
            "roots": (SearchRoot(label="fixtures", path=self.root),),
            "max_file_size_bytes": 1_024,
            "max_matches": 10,
        }
        values.update(overrides)
        return ReadOnlyFileSearch(ReadOnlyFileSearchConfig(**values))

    async def execute(self, tool: ReadOnlyFileSearch, query: str, max_matches: int = 10):
        registry = ToolRegistry()
        registry.register(tool)
        request = ToolRequest(
            tool_name="read_only_file_search",
            arguments={"query": query, "max_matches": max_matches},
        )
        return await registry.execute(
            request,
            granted_permissions={"filesystem.search.readonly"},
        )

    async def test_returns_only_relative_reproducible_citations(self) -> None:
        decision, result = await self.execute(self.make_tool(), "2025-02-14")

        self.assertTrue(decision.allowed)
        self.assertTrue(result.success)
        self.assertEqual(len(result.output["matches"]), 1)
        match = result.output["matches"][0]
        self.assertEqual(match["source_id"], "fixtures:notes.md")
        self.assertEqual(match["relative_path"], "notes.md")
        self.assertEqual(match["content_sha256"], hashlib.sha256(self.notes_bytes).hexdigest())
        # A short document is one passage: the citation spans all of its lines.
        self.assertEqual((match["line_start"], match["line_end"]), (1, 3))
        self.assertEqual(match["line_number"], 1)
        self.assertIn("Project Atlas launched on 2025-02-14.", match["snippet"])
        self.assertNotIn(str(self.root), str(result.output))

    async def test_hidden_sensitive_and_disallowed_files_are_skipped(self) -> None:
        _, result = await self.execute(self.make_tool(), "Project Atlas")

        self.assertTrue(result.success)
        self.assertEqual(
            {match["relative_path"] for match in result.output["matches"]},
            {"notes.md"},
        )
        self.assertGreaterEqual(result.output["skipped_entries"], 3)

    async def test_term_fallback_matches_inflected_fact_query(self) -> None:
        _, result = await self.execute(self.make_tool(), "Project Atlas launch date")

        self.assertTrue(result.success)
        self.assertEqual(result.output["matches"][0]["relative_path"], "notes.md")
        self.assertIn("Project Atlas launched on 2025-02-14.", result.output["matches"][0]["snippet"])

    async def test_term_fallback_matches_owner_and_owns_variants(self) -> None:
        _, result = await self.execute(self.make_tool(), "Who owns Project Atlas?")

        self.assertTrue(result.success)
        self.assertEqual(len(result.output["matches"]), 1)
        self.assertEqual(result.output["matches"][0]["relative_path"], "notes.md")

    async def test_term_fallback_matches_owner_query_to_owns_line(self) -> None:
        _, result = await self.execute(self.make_tool(), "Beacon review owner")

        self.assertTrue(result.success)
        self.assertEqual(result.output["matches"][0]["relative_path"], "nested/beacon.md")

    async def test_term_fallback_matches_completion_to_completed_line(self) -> None:
        _, result = await self.execute(self.make_tool(), "completion time Beacon")

        self.assertTrue(result.success)
        self.assertIn(
            "nested/timing.md",
            {match["relative_path"] for match in result.output["matches"]},
        )
        self.assertIn(
            "The Beacon trial completed in 3 hours.",
            {match["snippet"] for match in result.output["matches"]},
        )

    async def test_generic_benchmark_query_returns_bounded_document_lines(self) -> None:
        _, result = await self.execute(self.make_tool(), "documents", max_matches=2)

        self.assertTrue(result.success)
        self.assertEqual(len(result.output["matches"]), 2)
        self.assertTrue(result.output["truncated"])

    async def test_partial_subject_fallback_returns_relevant_lines(self) -> None:
        _, result = await self.execute(self.make_tool(), "approved budget for Project Atlas")

        self.assertTrue(result.success)
        self.assertGreaterEqual(len(result.output["matches"]), 1)
        self.assertEqual(result.output["matches"][0]["relative_path"], "notes.md")

    async def test_match_oversize_and_binary_caps_are_enforced(self) -> None:
        (self.root / "large.md").write_text(
            "oversize sentinel " + ("x" * 2_000),
            encoding="utf-8",
        )
        (self.root / "binary.md").write_bytes(b"binary sentinel\x00payload")
        tool = self.make_tool(max_file_size_bytes=1_024, max_matches=1)

        # Three documents mention Beacon equally; the host cap of 1 wins over the request for 100.
        _, ordinary_result = await self.execute(tool, "Beacon", max_matches=100)
        _, oversized_result = await self.execute(tool, "oversize sentinel")
        _, binary_result = await self.execute(tool, "binary sentinel")

        self.assertTrue(ordinary_result.success)
        self.assertEqual(len(ordinary_result.output["matches"]), 1)
        self.assertTrue(ordinary_result.output["matches"][0]["relative_path"].startswith("nested/"))
        self.assertTrue(ordinary_result.output["truncated"])
        self.assertGreaterEqual(ordinary_result.output["skipped_entries"], 4)
        self.assertEqual(oversized_result.output["matches"], [])
        self.assertEqual(binary_result.output["matches"], [])

    async def test_reparse_or_symlink_escape_is_not_followed(self) -> None:
        with tempfile.TemporaryDirectory() as outside_directory:
            outside = Path(outside_directory) / "outside.md"
            outside.write_text("outside sentinel", encoding="utf-8")
            link = self.root / "linked.md"
            try:
                os.symlink(outside, link)
            except (NotImplementedError, OSError):
                self.skipTest("symlink creation is unavailable for this user")

            _, result = await self.execute(self.make_tool(), "outside sentinel")

        self.assertTrue(result.success)
        self.assertEqual(result.output["matches"], [])

    def test_reparse_and_symlink_metadata_are_always_rejected(self) -> None:
        tool = self.make_tool()
        reparse_stat = SimpleNamespace(
            st_mode=stat.S_IFREG,
            st_file_attributes=0x400,
        )
        symlink_stat = SimpleNamespace(
            st_mode=stat.S_IFLNK,
            st_file_attributes=0,
        )
        windows_hidden_stat = SimpleNamespace(
            st_mode=stat.S_IFREG,
            st_file_attributes=0x2,
        )

        self.assertTrue(tool._entry_is_rejected("junction.md", reparse_stat))
        self.assertTrue(tool._entry_is_rejected("link.md", symlink_stat))
        self.assertTrue(tool._entry_is_rejected("hidden.md", windows_hidden_stat))

    def test_model_cannot_inject_path_or_traversal(self) -> None:
        schema = self.make_tool().definition.input_schema

        self.assertEqual(set(schema["properties"]), {"query", "max_matches"})
        for extra in (
            {"root": "C:\\"},
            {"path": "../../outside.md"},
            {"relative_path": "../outside.md"},
        ):
            with self.subTest(extra=extra), self.assertRaises(ValidationError):
                self.make_tool().input_model.model_validate({"query": "Atlas", **extra})

        with self.assertRaises(ValidationError):
            FileSearchMatch(
                source_id="fixtures:../outside.md",
                root_label="fixtures",
                relative_path="../outside.md",
                content_sha256="0" * 64,
                line_start=1,
                line_end=1,
                line_number=1,
                snippet="outside",
            )

    def test_filesystem_root_cannot_be_configured(self) -> None:
        with self.assertRaises(ValueError):
            ReadOnlyFileSearch(
                ReadOnlyFileSearchConfig(
                    roots=(SearchRoot(label="drive", path=Path(self.root.anchor)),)
                )
            )

        with self.assertRaises(ValidationError):
            SearchRoot(label="network", path=r"\\server\share\documents")

    def test_similarly_prefixed_directory_is_not_contained(self) -> None:
        sibling_prefix = self.root.with_name(self.root.name + "-outside")

        self.assertFalse(ReadOnlyFileSearch._is_within(sibling_prefix, self.root))


if __name__ == "__main__":
    unittest.main()
