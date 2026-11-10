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
    ReadOnlyFileReader,
    ReadOnlyFileReaderConfig,
    ReadOnlyFileReaderInput,
    SearchRoot,
    ToolRegistry,
)
from pydantic import ValidationError


class ReadOnlyFileReaderTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self) -> None:
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary_directory.name)
        self.notes_text = (
            "Project Atlas launched on 2025-02-14.\n"
            "Atlas processed 120 samples in four hours.\n"
            "Mina Chen owns the Project Atlas review.\n"
        )
        self.notes_bytes = self.notes_text.encode("utf-8")
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
            "Project Beacon processed 90 samples.\nBeacon finished cleanly.\n",
            encoding="utf-8",
        )

    def tearDown(self) -> None:
        self.temporary_directory.cleanup()

    def make_tool(self, **overrides) -> ReadOnlyFileReader:
        values = {
            "roots": (SearchRoot(label="fixtures", path=self.root),),
            "max_file_size_bytes": 1_024,
            "max_lines": 200,
        }
        values.update(overrides)
        return ReadOnlyFileReader(ReadOnlyFileReaderConfig(**values))

    async def execute(self, tool: ReadOnlyFileReader, arguments: dict):
        registry = ToolRegistry()
        registry.register(tool)
        request = ToolRequest(
            tool_name="read_only_file_reader",
            arguments=arguments,
        )
        return await registry.execute(
            request,
            granted_permissions={"filesystem.read.readonly"},
        )

    async def test_reads_only_requested_relative_line_range(self) -> None:
        _, result = await self.execute(
            self.make_tool(),
            {
                "root_label": "fixtures",
                "relative_path": "notes.md",
                "line_start": 2,
                "line_end": 3,
            },
        )

        self.assertTrue(result.success)
        self.assertEqual(result.output["source_id"], "fixtures:notes.md")
        self.assertEqual(result.output["relative_path"], "notes.md")
        self.assertEqual(result.output["content_sha256"], hashlib.sha256(self.notes_bytes).hexdigest())
        self.assertEqual(result.output["line_start"], 2)
        self.assertEqual(result.output["line_end"], 3)
        self.assertEqual(result.output["total_lines"], 3)
        self.assertFalse(result.output["truncated"])
        self.assertEqual(
            result.output["content"],
            "Atlas processed 120 samples in four hours.\n"
            "Mina Chen owns the Project Atlas review.",
        )
        self.assertNotIn(str(self.root), str(result.output))

    async def test_explicit_line_range_is_capped_and_marked_truncated(self) -> None:
        _, result = await self.execute(
            self.make_tool(max_lines=1),
            {
                "root_label": "fixtures",
                "relative_path": "notes.md",
                "line_start": 1,
                "line_end": 3,
            },
        )

        self.assertTrue(result.success)
        self.assertEqual(result.output["line_end"], 1)
        self.assertTrue(result.output["truncated"])
        self.assertEqual(result.output["content"], "Project Atlas launched on 2025-02-14.")

    async def test_omitted_line_end_is_still_bounded_and_marked_truncated(self) -> None:
        _, result = await self.execute(
            self.make_tool(max_lines=2),
            {
                "root_label": "fixtures",
                "relative_path": "notes.md",
                "line_start": 1,
            },
        )

        self.assertTrue(result.success)
        self.assertEqual(result.output["line_end"], 2)
        self.assertTrue(result.output["truncated"])
        self.assertEqual(len(result.output["content"].splitlines()), 2)

    async def test_hidden_sensitive_and_disallowed_files_are_rejected(self) -> None:
        for relative_path in (".hidden.md", "credentials.json", "ignored.py"):
            with self.subTest(relative_path=relative_path):
                _, result = await self.execute(
                    self.make_tool(),
                    {
                        "root_label": "fixtures",
                        "relative_path": relative_path,
                    },
                )

                self.assertFalse(result.success)

    async def test_unknown_root_is_rejected(self) -> None:
        _, result = await self.execute(
            self.make_tool(),
            {
                "root_label": "missing",
                "relative_path": "notes.md",
            },
        )

        self.assertFalse(result.success)

    async def test_missing_permission_denies_without_reading(self) -> None:
        registry = ToolRegistry()
        registry.register(self.make_tool())

        decision, result = await registry.execute(
            ToolRequest(
                tool_name="read_only_file_reader",
                arguments={
                    "root_label": "fixtures",
                    "relative_path": "notes.md",
                },
            ),
            granted_permissions=set(),
        )

        self.assertFalse(decision.allowed)
        self.assertFalse(result.success)

    async def test_reparse_or_symlink_escape_is_not_followed(self) -> None:
        with tempfile.TemporaryDirectory() as outside_directory:
            outside = Path(outside_directory) / "outside.md"
            outside.write_text("outside sentinel", encoding="utf-8")
            link = self.root / "linked.md"
            try:
                os.symlink(outside, link)
            except (NotImplementedError, OSError):
                self.skipTest("symlink creation is unavailable for this user")

            _, result = await self.execute(
                self.make_tool(),
                {
                    "root_label": "fixtures",
                    "relative_path": "linked.md",
                },
            )

        self.assertFalse(result.success)

    def test_model_cannot_inject_absolute_path_or_traversal(self) -> None:
        for relative_path in ("../outside.md", "/outside.md", r"C:\outside.md", "nested//file.md"):
            with self.subTest(relative_path=relative_path), self.assertRaises(ValidationError):
                ReadOnlyFileReaderInput(
                    root_label="fixtures",
                    relative_path=relative_path,
                )

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

    def test_filesystem_root_cannot_be_configured(self) -> None:
        with self.assertRaises(ValueError):
            ReadOnlyFileReader(
                ReadOnlyFileReaderConfig(
                    roots=(SearchRoot(label="drive", path=Path(self.root.anchor)),)
                )
            )


if __name__ == "__main__":
    unittest.main()
