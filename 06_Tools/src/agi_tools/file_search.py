"""Read-only literal text search within administrator-configured roots."""

from __future__ import annotations

import asyncio
import hashlib
import os
import re
import stat
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter
from typing import Any

from agi_contracts import RiskLevel
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    field_validator,
    model_validator,
)

from .base import BaseTool, ToolInputError


DEFAULT_ALLOWED_EXTENSIONS = frozenset({".csv", ".json", ".md", ".rst", ".txt", ".yaml", ".yml"})
DEFAULT_SENSITIVE_NAMES = frozenset(
    {
        ".env",
        ".git",
        ".hg",
        ".ssh",
        ".svn",
        "credentials.json",
        "secrets",
        "tokens.json",
    }
)
DEFAULT_SENSITIVE_SUFFIXES = frozenset({".cer", ".crt", ".der", ".key", ".p12", ".pem", ".pfx"})
WINDOWS_HIDDEN_ATTRIBUTE = getattr(stat, "FILE_ATTRIBUTE_HIDDEN", 0x2)
QUERY_STOPWORDS = frozenset(
    {
        "a",
        "an",
        "and",
        "date",
        "fact",
        "facts",
        "find",
        "for",
        "in",
        "is",
        "of",
        "on",
        "the",
        "what",
        "when",
        "who",
    }
)
WINDOWS_REPARSE_ATTRIBUTE = getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0x400)


class SearchRoot(BaseModel):
    """A trusted root supplied by application configuration, never tool input."""

    model_config = ConfigDict(extra="forbid", frozen=True, str_strip_whitespace=True)

    label: str = Field(pattern=r"^[A-Za-z0-9][A-Za-z0-9_-]{0,63}$")
    path: Path

    @field_validator("path")
    @classmethod
    def path_must_be_absolute(cls, value: Path) -> Path:
        path_text = os.fspath(value)
        if path_text.startswith(("\\\\", "//")):
            raise ValueError("UNC and device search roots are not allowed")
        if not value.is_absolute():
            raise ValueError("search root path must be absolute")
        return value


class ReadOnlyFileSearchConfig(BaseModel):
    """Hard limits and roots controlled by the host application."""

    model_config = ConfigDict(extra="forbid", frozen=True, validate_default=True)

    roots: tuple[SearchRoot, ...] = Field(min_length=1, max_length=16)
    allowed_extensions: frozenset[str] = DEFAULT_ALLOWED_EXTENSIONS
    max_file_size_bytes: int = Field(default=1_048_576, ge=1, le=10_485_760)
    max_matches: int = Field(default=50, ge=1, le=100)
    max_files_scanned: int = Field(default=1_000, ge=1, le=10_000)
    max_directories_scanned: int = Field(default=1_000, ge=1, le=10_000)
    max_snippet_chars: int = Field(default=400, ge=80, le=1_000)
    case_sensitive: bool = False
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
    def labels_must_be_unique(self) -> "ReadOnlyFileSearchConfig":
        labels = [root.label.casefold() for root in self.roots]
        if len(labels) != len(set(labels)):
            raise ValueError("search root labels must be unique")
        return self


class ReadOnlyFileSearchInput(BaseModel):
    """Model-controlled search input; deliberately contains no path or root."""

    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    query: str = Field(min_length=1, max_length=200)
    max_matches: int = Field(default=20, ge=1, le=100)

    @field_validator("query")
    @classmethod
    def query_must_be_plain_text(cls, value: str) -> str:
        if any(character in value for character in ("\x00", "\r", "\n")):
            raise ValueError("query must be a single line")
        return value


class FileSearchMatch(BaseModel):
    """One line containing the literal query."""

    model_config = ConfigDict(extra="forbid")

    source_id: str
    root_label: str
    relative_path: str
    content_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    line_start: int = Field(ge=1)
    line_end: int = Field(ge=1)
    line_number: int = Field(ge=1)
    snippet: str

    @field_validator("relative_path")
    @classmethod
    def relative_path_must_stay_relative(cls, value: str) -> str:
        pure_parts = value.replace("\\", "/").split("/")
        if not value or value.startswith(("/", "\\")) or ".." in pure_parts:
            raise ValueError("match path must be a safe relative path")
        if len(value) >= 2 and value[1] == ":":
            raise ValueError("match path must not contain a drive prefix")
        return value.replace("\\", "/")

    @model_validator(mode="after")
    def line_bounds_must_be_consistent(self) -> "FileSearchMatch":
        if self.line_end < self.line_start:
            raise ValueError("line end cannot precede line start")
        if self.line_number != self.line_start:
            raise ValueError("line number must equal line start")
        return self


class FileSearchOutput(BaseModel):
    """Bounded search results with no configured absolute paths."""

    model_config = ConfigDict(extra="forbid")

    query: str
    matches: list[FileSearchMatch]
    scanned_files: int = Field(ge=0)
    skipped_entries: int = Field(ge=0)
    truncated: bool


@dataclass(frozen=True, slots=True)
class _PreparedRoot:
    label: str
    path: Path


class ReadOnlyFileSearch(BaseTool[ReadOnlyFileSearchInput, FileSearchOutput]):
    """Search safe text files without exposing arbitrary filesystem access."""

    name = "read_only_file_search"
    description = (
        "Find literal text in approved local documents. Search roots are fixed by the host."
    )
    risk_level = RiskLevel.LOW
    permissions_required = ("filesystem.search.readonly",)
    human_approval_required = False
    timeout_seconds = 10.0
    input_model = ReadOnlyFileSearchInput
    output_model = FileSearchOutput

    def __init__(self, config: ReadOnlyFileSearchConfig) -> None:
        self.config = config
        self._roots = self._prepare_roots(config.roots)

    async def run(self, tool_input: ReadOnlyFileSearchInput) -> FileSearchOutput:
        return await asyncio.to_thread(self._search_sync, tool_input)

    def _search_sync(self, tool_input: ReadOnlyFileSearchInput) -> FileSearchOutput:
        deadline = perf_counter() + self.timeout_seconds
        effective_limit = min(tool_input.max_matches, self.config.max_matches)
        needle = tool_input.query if self.config.case_sensitive else tool_input.query.casefold()
        query_terms = _query_terms(tool_input.query, case_sensitive=self.config.case_sensitive)
        generic_query = _is_generic_document_query(query_terms)
        matches: list[FileSearchMatch] = []
        fallback_matches: list[FileSearchMatch] = []
        scanned_files = 0
        skipped_entries = 0
        directories_scanned = 0
        truncated = False

        for root in self._roots:
            pending_directories = [root.path]
            while pending_directories:
                if perf_counter() >= deadline:
                    raise ToolInputError("file search timed out")
                if directories_scanned >= self.config.max_directories_scanned:
                    truncated = True
                    pending_directories.clear()
                    break

                directory = pending_directories.pop()
                directories_scanned += 1
                try:
                    with os.scandir(directory) as iterator:
                        entries = sorted(iterator, key=lambda entry: entry.name.casefold())
                except (FileNotFoundError, NotADirectoryError, PermissionError, OSError):
                    skipped_entries += 1
                    continue

                child_directories: list[Path] = []
                for entry in entries:
                    if perf_counter() >= deadline:
                        raise ToolInputError("file search timed out")
                    try:
                        entry_stat = entry.stat(follow_symlinks=False)
                    except (FileNotFoundError, PermissionError, OSError):
                        skipped_entries += 1
                        continue

                    if self._entry_is_rejected(entry.name, entry_stat):
                        skipped_entries += 1
                        continue

                    try:
                        if entry.is_dir(follow_symlinks=False):
                            child = Path(entry.path)
                            if self._is_within(child, root.path):
                                child_directories.append(child)
                            else:
                                skipped_entries += 1
                            continue
                        if not entry.is_file(follow_symlinks=False):
                            skipped_entries += 1
                            continue
                    except OSError:
                        skipped_entries += 1
                        continue

                    candidate = Path(entry.path)
                    if candidate.suffix.casefold() not in self.config.allowed_extensions:
                        skipped_entries += 1
                        continue
                    if entry_stat.st_size > self.config.max_file_size_bytes:
                        skipped_entries += 1
                        continue
                    if scanned_files >= self.config.max_files_scanned:
                        truncated = True
                        break

                    safe_candidate = self._safe_resolved_file(candidate, root, entry_stat)
                    if safe_candidate is None:
                        skipped_entries += 1
                        continue

                    file_content = self._read_bounded_file(safe_candidate, entry_stat)
                    if file_content is None:
                        skipped_entries += 1
                        continue
                    content, content_sha256 = file_content
                    scanned_files += 1

                    relative_path = safe_candidate.relative_to(root.path).as_posix()
                    for line_number, line in enumerate(content.splitlines(), start=1):
                        haystack = line if self.config.case_sensitive else line.casefold()
                        location = haystack.find(needle)
                        term_match = _line_matches_terms(haystack, query_terms)
                        generic_match = generic_query and line.strip()
                        if location < 0 and not term_match and not generic_match:
                            fallback_location = _partial_term_location(haystack, query_terms)
                            if fallback_location >= 0 and len(fallback_matches) < effective_limit:
                                fallback_matches.append(
                                    FileSearchMatch(
                                        source_id=f"{root.label}:{relative_path}",
                                        root_label=root.label,
                                        relative_path=relative_path,
                                        content_sha256=content_sha256,
                                        line_start=line_number,
                                        line_end=line_number,
                                        line_number=line_number,
                                        snippet=self._snippet(
                                            line, fallback_location, len(tool_input.query)
                                        ),
                                    )
                                )
                            continue
                        if location < 0 and query_terms:
                            location = _first_term_location(haystack, query_terms)
                        matches.append(
                            FileSearchMatch(
                                source_id=f"{root.label}:{relative_path}",
                                root_label=root.label,
                                relative_path=relative_path,
                                content_sha256=content_sha256,
                                line_start=line_number,
                                line_end=line_number,
                                line_number=line_number,
                                snippet=self._snippet(line, location, len(tool_input.query)),
                            )
                        )
                        if len(matches) >= effective_limit:
                            truncated = True
                            return FileSearchOutput(
                                query=tool_input.query,
                                matches=matches,
                                scanned_files=scanned_files,
                                skipped_entries=skipped_entries,
                                truncated=truncated,
                            )

                if truncated:
                    break
                # Reverse push order so the next directory popped is lexical-first.
                pending_directories.extend(reversed(child_directories))

            if truncated:
                break

        return FileSearchOutput(
            query=tool_input.query,
            matches=matches or fallback_matches,
            scanned_files=scanned_files,
            skipped_entries=skipped_entries,
            truncated=truncated,
        )

    def _prepare_roots(self, roots: tuple[SearchRoot, ...]) -> tuple[_PreparedRoot, ...]:
        prepared: list[_PreparedRoot] = []
        for configured_root in roots:
            try:
                configured_path = Path(
                    os.path.abspath(os.fspath(configured_root.path.expanduser()))
                )
                configured_stat = configured_path.lstat()
                if self._entry_is_rejected(configured_path.name, configured_stat):
                    raise ValueError(
                        "configured search root is hidden, sensitive, or reparse-backed: "
                        f"{configured_root.label}"
                    )
                path = configured_path.resolve(strict=True)
                root_stat = path.lstat()
            except ValueError:
                raise
            except (FileNotFoundError, OSError) as exc:
                raise ValueError(
                    f"configured search root is unavailable: {configured_root.label}"
                ) from exc
            if not path.is_dir():
                raise ValueError(
                    f"configured search root is not a directory: {configured_root.label}"
                )
            if path == Path(path.anchor):
                raise ValueError("filesystem roots cannot be configured as search roots")
            if self._entry_is_rejected(path.name, root_stat):
                raise ValueError(
                    f"configured search root is hidden, sensitive, or reparse-backed: "
                    f"{configured_root.label}"
                )
            for existing in prepared:
                if self._is_within(path, existing.path) or self._is_within(existing.path, path):
                    raise ValueError("configured search roots must not overlap")
            prepared.append(_PreparedRoot(label=configured_root.label, path=path))
        return tuple(prepared)

    def _safe_resolved_file(
        self,
        candidate: Path,
        root: _PreparedRoot,
        entry_stat: os.stat_result,
    ) -> Path | None:
        if self._stat_is_reparse(entry_stat):
            return None
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

    def _read_bounded_file(
        self,
        path: Path,
        expected_stat: os.stat_result,
    ) -> tuple[str, str] | None:
        try:
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

    def _snippet(self, line: str, match_start: int, query_length: int) -> str:
        clean_line = " ".join(line.strip().split())
        if len(clean_line) <= self.config.max_snippet_chars:
            return clean_line

        # The whitespace normalization above can shift the original match. Find it
        # again when possible and otherwise use the original offset conservatively.
        query_length = min(query_length, self.config.max_snippet_chars)
        start = min(match_start, max(0, len(clean_line) - 1))
        left_context = max(20, (self.config.max_snippet_chars - query_length) // 2)
        window_start = max(0, start - left_context)
        window_end = min(len(clean_line), window_start + self.config.max_snippet_chars)
        window_start = max(0, window_end - self.config.max_snippet_chars)
        snippet = clean_line[window_start:window_end]
        if window_start:
            snippet = "..." + snippet[3:]
        if window_end < len(clean_line):
            snippet = snippet[:-3] + "..."
        return snippet

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
        """Case-normalized containment that handles Windows drive boundaries."""

        candidate_text = os.path.normcase(os.path.abspath(os.fspath(candidate)))
        root_text = os.path.normcase(os.path.abspath(os.fspath(root)))
        try:
            return os.path.commonpath((candidate_text, root_text)) == root_text
        except ValueError:
            # Different Windows drives have no common path.
            return False


def _query_terms(query: str, *, case_sensitive: bool) -> tuple[str, ...]:
    text = query if case_sensitive else query.casefold()
    terms = [
        term
        for term in re.findall(r"[A-Za-z0-9]+", text)
        if len(term) >= 2 and term not in QUERY_STOPWORDS
    ]
    return tuple(dict.fromkeys(terms))


def _line_matches_terms(haystack: str, query_terms: tuple[str, ...]) -> bool:
    if not query_terms:
        return False
    line_terms = re.findall(r"[A-Za-z0-9]+", haystack)
    return all(
        any(
            line_term == query_term
            or line_term.startswith(query_term)
            or query_term.startswith(line_term)
            or _stem(line_term) == _stem(query_term)
            for line_term in line_terms
        )
        for query_term in query_terms
    )


def _partial_term_location(haystack: str, query_terms: tuple[str, ...]) -> int:
    weak_terms = {"details", "document", "documents", "file", "files", "local"}
    locations = [
        location
        for term in query_terms
        if term not in weak_terms
        for candidate in (term, _stem(term))
        if (location := haystack.find(candidate)) >= 0
    ]
    return min(locations) if locations else -1


def _first_term_location(haystack: str, query_terms: tuple[str, ...]) -> int:
    locations = [haystack.find(term) for term in query_terms if haystack.find(term) >= 0]
    return min(locations) if locations else 0


_STEM_SUFFIXES = ("ations", "ation", "ions", "ion", "ings", "ing", "ers", "er", "ies", "ied", "ed", "es", "s", "e")


def _stem(term: str) -> str:
    """Generic light suffix stripping; no word-specific rules."""

    for suffix in _STEM_SUFFIXES:
        if len(term) > len(suffix) + 2 and term.endswith(suffix):
            if suffix == "s" and term.endswith(("ss", "us", "is", "as")):
                return term
            if suffix in {"ies", "ied"}:
                return term[: -len(suffix)] + "y"
            return term[: -len(suffix)]
    return term


def _is_generic_document_query(query_terms: tuple[str, ...]) -> bool:
    if not query_terms:
        return False
    generic_terms = {
        "document",
        "documents",
        "local",
        "summarize",
        "summary",
    }
    return all(term in generic_terms for term in query_terms)
