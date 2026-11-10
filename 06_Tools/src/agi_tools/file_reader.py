"""Read-only line-range access within administrator-configured document roots."""

from __future__ import annotations

import asyncio
import hashlib
import os
import stat
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from agi_contracts import RiskLevel
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from .base import BaseTool, ToolInputError
from .file_search import (
    DEFAULT_ALLOWED_EXTENSIONS,
    DEFAULT_SENSITIVE_NAMES,
    DEFAULT_SENSITIVE_SUFFIXES,
    WINDOWS_HIDDEN_ATTRIBUTE,
    WINDOWS_REPARSE_ATTRIBUTE,
    SearchRoot,
)


class ReadOnlyFileReaderConfig(BaseModel):
    """Hard limits and roots controlled by the host application."""

    model_config = ConfigDict(extra="forbid", frozen=True, validate_default=True)

    roots: tuple[SearchRoot, ...] = Field(min_length=1, max_length=16)
    allowed_extensions: frozenset[str] = DEFAULT_ALLOWED_EXTENSIONS
    max_file_size_bytes: int = Field(default=1_048_576, ge=1, le=10_485_760)
    max_lines: int = Field(default=200, ge=1, le=2_000)
    sensitive_names: frozenset[str] = DEFAULT_SENSITIVE_NAMES
    sensitive_suffixes: frozenset[str] = DEFAULT_SENSITIVE_SUFFIXES

    @field_validator("allowed_extensions", mode="before")
    @classmethod
    def normalize_extensions(cls, value: Any) -> frozenset[str]:
        if isinstance(value, str):
            raise ValueError("allowed extensions must be a collection")
        normalized = frozenset(str(item).casefold() for item in value)
        if not normalized or any(
            not item.startswith(".") or item in {".", ".*"} for item in normalized
        ):
            raise ValueError("allowed extensions must be explicit dotted suffixes")
        return normalized

    @field_validator("sensitive_names", "sensitive_suffixes", mode="before")
    @classmethod
    def normalize_sensitive_values(cls, value: Any) -> frozenset[str]:
        if isinstance(value, str):
            raise ValueError("sensitive values must be a collection")
        return frozenset(str(item).casefold() for item in value)

    @model_validator(mode="after")
    def labels_must_be_unique(self) -> "ReadOnlyFileReaderConfig":
        labels = [root.label.casefold() for root in self.roots]
        if len(labels) != len(set(labels)):
            raise ValueError("reader root labels must be unique")
        return self


class ReadOnlyFileReaderInput(BaseModel):
    """Model-controlled read input; root paths remain host-controlled."""

    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    root_label: str = Field(pattern=r"^[A-Za-z0-9][A-Za-z0-9_-]{0,63}$")
    relative_path: str = Field(min_length=1, max_length=512)
    line_start: int = Field(default=1, ge=1)
    line_end: int | None = Field(default=None, ge=1)

    @field_validator("relative_path")
    @classmethod
    def relative_path_must_be_safe(cls, value: str) -> str:
        if any(character in value for character in ("\x00", "\r", "\n")):
            raise ValueError("relative path must be a single path")
        normalized = value.replace("\\", "/")
        parts = normalized.split("/")
        if (
            not normalized
            or normalized.startswith("/")
            or ".." in parts
            or any(not part for part in parts)
        ):
            raise ValueError("relative path must stay within the configured root")
        if len(normalized) >= 2 and normalized[1] == ":":
            raise ValueError("relative path must not contain a drive prefix")
        return normalized

    @model_validator(mode="after")
    def line_bounds_must_be_ordered(self) -> "ReadOnlyFileReaderInput":
        if self.line_end is not None and self.line_end < self.line_start:
            raise ValueError("line_end cannot precede line_start")
        return self


class ReadOnlyFileReaderOutput(BaseModel):
    """Bounded file content with reproducible source metadata."""

    model_config = ConfigDict(extra="forbid")

    source_id: str
    root_label: str
    relative_path: str
    content_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    line_start: int = Field(ge=1)
    line_end: int = Field(ge=1)
    content: str
    total_lines: int = Field(ge=0)
    truncated: bool


@dataclass(frozen=True, slots=True)
class _PreparedRoot:
    label: str
    path: Path


class ReadOnlyFileReader(BaseTool[ReadOnlyFileReaderInput, ReadOnlyFileReaderOutput]):
    """Read safe text line ranges without exposing arbitrary filesystem access."""

    name = "read_only_file_reader"
    description = (
        "Read a bounded line range from an approved local document. Roots are fixed by the host."
    )
    risk_level = RiskLevel.LOW
    permissions_required = ("filesystem.read.readonly",)
    human_approval_required = False
    timeout_seconds = 5.0
    input_model = ReadOnlyFileReaderInput
    output_model = ReadOnlyFileReaderOutput

    def __init__(self, config: ReadOnlyFileReaderConfig) -> None:
        self.config = config
        self._roots = self._prepare_roots(config.roots)

    async def run(self, tool_input: ReadOnlyFileReaderInput) -> ReadOnlyFileReaderOutput:
        return await asyncio.to_thread(self._read_sync, tool_input)

    def _read_sync(self, tool_input: ReadOnlyFileReaderInput) -> ReadOnlyFileReaderOutput:
        root = self._roots.get(tool_input.root_label.casefold())
        if root is None:
            raise ToolInputError("unknown configured root label")
        candidate = root.path.joinpath(*tool_input.relative_path.split("/"))
        safe_candidate = self._safe_resolved_file(candidate, root)
        if safe_candidate is None:
            raise ToolInputError("file is unavailable or outside the configured root")
        if safe_candidate.suffix.casefold() not in self.config.allowed_extensions:
            raise ToolInputError("file extension is not readable by this tool")

        file_content = self._read_bounded_file(safe_candidate)
        if file_content is None:
            raise ToolInputError("file is not readable as bounded UTF-8 text")
        content, content_sha256 = file_content
        lines = content.splitlines()
        total_lines = len(lines)
        requested_end = tool_input.line_end or total_lines
        if total_lines == 0:
            selected: list[str] = []
            actual_end = tool_input.line_start
        else:
            capped_end = min(
                requested_end,
                tool_input.line_start + self.config.max_lines - 1,
                total_lines,
            )
            selected = lines[tool_input.line_start - 1 : min(capped_end, total_lines)]
            actual_end = tool_input.line_start + len(selected) - 1 if selected else tool_input.line_start
        truncated = requested_end > actual_end
        relative_path = safe_candidate.relative_to(root.path).as_posix()
        return ReadOnlyFileReaderOutput(
            source_id=f"{root.label}:{relative_path}",
            root_label=root.label,
            relative_path=relative_path,
            content_sha256=content_sha256,
            line_start=tool_input.line_start,
            line_end=actual_end,
            content="\n".join(selected),
            total_lines=total_lines,
            truncated=truncated,
        )

    def _prepare_roots(self, roots: tuple[SearchRoot, ...]) -> dict[str, _PreparedRoot]:
        prepared: dict[str, _PreparedRoot] = {}
        prepared_paths: list[Path] = []
        for configured_root in roots:
            try:
                configured_path = Path(os.path.abspath(os.fspath(configured_root.path.expanduser())))
                configured_stat = configured_path.lstat()
                if self._entry_is_rejected(configured_path.name, configured_stat):
                    raise ValueError(
                        "configured reader root is hidden, sensitive, or reparse-backed: "
                        f"{configured_root.label}"
                    )
                path = configured_path.resolve(strict=True)
                root_stat = path.lstat()
            except ValueError:
                raise
            except (FileNotFoundError, OSError) as exc:
                raise ValueError(
                    f"configured reader root is unavailable: {configured_root.label}"
                ) from exc
            if not path.is_dir():
                raise ValueError(f"configured reader root is not a directory: {configured_root.label}")
            if path == Path(path.anchor):
                raise ValueError("filesystem roots cannot be configured as reader roots")
            if self._entry_is_rejected(path.name, root_stat):
                raise ValueError(
                    f"configured reader root is hidden, sensitive, or reparse-backed: "
                    f"{configured_root.label}"
                )
            for existing in prepared_paths:
                if self._is_within(path, existing) or self._is_within(existing, path):
                    raise ValueError("configured reader roots must not overlap")
            prepared_paths.append(path)
            prepared[configured_root.label.casefold()] = _PreparedRoot(
                label=configured_root.label,
                path=path,
            )
        return prepared

    def _safe_resolved_file(self, candidate: Path, root: _PreparedRoot) -> Path | None:
        try:
            resolved = candidate.resolve(strict=True)
        except (FileNotFoundError, OSError):
            return None
        if not self._is_within(resolved, root.path) or not resolved.is_file():
            return None
        try:
            relative_parts = resolved.relative_to(root.path).parts
        except ValueError:
            return None
        cursor = root.path
        for part in relative_parts:
            cursor = cursor / part
            try:
                cursor_stat = cursor.lstat()
            except (FileNotFoundError, PermissionError, OSError):
                return None
            if self._entry_is_rejected(part, cursor_stat):
                return None
        return resolved

    def _read_bounded_file(self, path: Path) -> tuple[str, str] | None:
        try:
            expected_stat = path.lstat()
            if expected_stat.st_size > self.config.max_file_size_bytes:
                return None
            with path.open("rb") as file_handle:
                opened_stat = os.fstat(file_handle.fileno())
                if not stat.S_ISREG(opened_stat.st_mode) or self._stat_is_reparse(opened_stat):
                    return None
                if (
                    expected_stat.st_ino
                    and opened_stat.st_ino
                    and (
                        expected_stat.st_ino != opened_stat.st_ino
                        or expected_stat.st_dev != opened_stat.st_dev
                    )
                ):
                    return None
                raw = file_handle.read(self.config.max_file_size_bytes + 1)
        except (FileNotFoundError, PermissionError, OSError):
            return None
        if len(raw) > self.config.max_file_size_bytes or b"\x00" in raw:
            return None
        try:
            content = raw.decode("utf-8-sig", errors="strict")
        except UnicodeDecodeError:
            return None
        return content, hashlib.sha256(raw).hexdigest()

    def _entry_is_rejected(self, name: str, entry_stat: os.stat_result) -> bool:
        folded = name.casefold()
        if name.startswith("."):
            return True
        sensitive_names = self.config.sensitive_names | DEFAULT_SENSITIVE_NAMES
        sensitive_suffixes = self.config.sensitive_suffixes | DEFAULT_SENSITIVE_SUFFIXES
        if folded in sensitive_names:
            return True
        if any(folded.endswith(suffix) for suffix in sensitive_suffixes):
            return True
        if self._stat_is_hidden(entry_stat) or self._stat_is_reparse(entry_stat):
            return True
        return False

    @staticmethod
    def _stat_is_hidden(entry_stat: os.stat_result) -> bool:
        attributes = int(getattr(entry_stat, "st_file_attributes", 0))
        return bool(attributes & WINDOWS_HIDDEN_ATTRIBUTE)

    @staticmethod
    def _stat_is_reparse(entry_stat: os.stat_result) -> bool:
        attributes = int(getattr(entry_stat, "st_file_attributes", 0))
        return stat.S_ISLNK(entry_stat.st_mode) or bool(attributes & WINDOWS_REPARSE_ATTRIBUTE)

    @staticmethod
    def _is_within(candidate: Path, root: Path) -> bool:
        candidate_text = os.path.normcase(os.path.abspath(os.fspath(candidate)))
        root_text = os.path.normcase(os.path.abspath(os.fspath(root)))
        try:
            return os.path.commonpath((candidate_text, root_text)) == root_text
        except ValueError:
            return False
