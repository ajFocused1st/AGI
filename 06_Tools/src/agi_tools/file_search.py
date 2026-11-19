"""Read-only passage search within administrator-configured roots."""

from __future__ import annotations

import asyncio
import bisect
import hashlib
import math
import os
import re
import stat
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from collections.abc import Callable
from time import perf_counter
from typing import Any, Literal

from agi_contracts import RiskLevel
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    field_validator,
    model_validator,
)

from .base import BaseTool, ToolInputError
from .semantic import EmbeddingUnavailableError, OllamaPassageEmbedder


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
    "a an and date fact facts find for in is of on the what when who "
    # Function words: they occur everywhere, so matching them says nothing about relevance.
    "am as at be been being but by can could do had has have he her his if it may me my no "
    "nor not or our she should so to too us very we would you your".split()
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
    # Prose is returned as sentence-aligned passages of about this many characters.
    passage_chars: int = Field(default=800, ge=200, le=2_000)
    # Total snippet characters one search may return (the best match is always returned).
    max_result_chars: int = Field(default=4_000, ge=200, le=20_000)
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
    """One matching unit: a fact line, or a passage of prose spanning several lines."""

    model_config = ConfigDict(extra="forbid")

    source_id: str
    root_label: str
    relative_path: str
    content_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    line_start: int = Field(ge=1)
    line_end: int = Field(ge=1)
    line_number: int = Field(ge=1)
    snippet: str
    document_title: str | None = Field(default=None, max_length=200)

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
    # "keyword+meaning" when an embedding model also ranked the passages.
    ranking: Literal["keyword", "keyword+meaning"] = "keyword"


@dataclass(frozen=True, slots=True)
class _PreparedRoot:
    label: str
    path: Path


class ReadOnlyFileSearch(BaseTool[ReadOnlyFileSearchInput, FileSearchOutput]):
    """Search safe text files without exposing arbitrary filesystem access."""

    name = "read_only_file_search"
    description = (
        "Find the passages of approved local documents most relevant to a plain-language "
        "query. Search roots are fixed by the host."
    )
    risk_level = RiskLevel.LOW
    permissions_required = ("filesystem.search.readonly",)
    human_approval_required = False
    timeout_seconds = 10.0
    input_model = ReadOnlyFileSearchInput
    output_model = FileSearchOutput

    def __init__(
        self,
        config: ReadOnlyFileSearchConfig,
        embedder: OllamaPassageEmbedder | None = None,
    ) -> None:
        self.config = config
        self._roots = self._prepare_roots(config.roots)
        self._embedder = embedder

    async def run(self, tool_input: ReadOnlyFileSearchInput) -> FileSearchOutput:
        return await asyncio.to_thread(self._search_sync, tool_input)

    def warm_up(self) -> int:
        """Embed every passage not yet cached, so searches only embed their query.

        Raises EmbeddingUnavailableError when the embedding model cannot be used.
        Returns how many passages were newly embedded (0 without an embedder).
        """

        if self._embedder is None:
            return 0
        documents, *_ = self._scan(perf_counter() + 3600)
        texts = [u.text for u in _units(documents, self.config.passage_chars) if not u.is_heading]
        return self._embedder.ensure(texts)

    def _search_sync(self, tool_input: ReadOnlyFileSearchInput) -> FileSearchOutput:
        deadline = perf_counter() + self.timeout_seconds
        effective_limit = min(tool_input.max_matches, self.config.max_matches)
        documents, scanned_files, skipped_entries, truncated = self._scan(deadline)
        meaning_used = False

        def meaning(passages: list[_Unit]) -> list[float | None] | None:
            nonlocal meaning_used
            assert self._embedder is not None
            try:
                # Leave a second for ranking; passages not embedded by then rank by keyword only.
                scores = self._embedder.similarities(
                    tool_input.query, [unit.text for unit in passages], deadline=deadline - 1.0
                )
            except EmbeddingUnavailableError:
                return None  # the keyword ranking still answers
            meaning_used = True
            return scores

        ranked, more_available = _rank_units(
            documents,
            tool_input.query,
            limit=effective_limit,
            case_sensitive=self.config.case_sensitive,
            passage_chars=self.config.passage_chars,
            max_result_chars=self.config.max_result_chars,
            meaning=meaning if self._embedder is not None else None,
        )
        matches = [
            FileSearchMatch(
                source_id=f"{unit.document.root_label}:{unit.document.relative_path}",
                root_label=unit.document.root_label,
                relative_path=unit.document.relative_path,
                content_sha256=unit.document.content_sha256,
                line_start=unit.line_start,
                line_end=unit.line_end,
                line_number=unit.line_start,
                snippet=unit.text,
                document_title=unit.heading_text,
            )
            for unit in ranked
        ]
        return FileSearchOutput(
            query=tool_input.query,
            matches=matches,
            scanned_files=scanned_files,
            skipped_entries=skipped_entries,
            truncated=truncated or more_available,
            ranking="keyword+meaning" if meaning_used else "keyword",
        )

    def _scan(self, deadline: float) -> tuple[list[_ScannedDocument], int, int, bool]:
        """Every safe, allowed document under the roots, read within the limits."""

        documents: list[_ScannedDocument] = []
        scanned_files = 0
        skipped_entries = 0
        directories_scanned = 0
        truncated = False

        # Traversal and every file-safety check are unchanged; only matching moved below.
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
                    documents.append(
                        _ScannedDocument(
                            root_label=root.label,
                            relative_path=safe_candidate.relative_to(root.path).as_posix(),
                            content_sha256=content_sha256,
                            lines=content.splitlines(),
                        )
                    )

                if truncated:
                    break
                # Reverse push order so the next directory popped is lexical-first.
                pending_directories.extend(reversed(child_directories))

            if truncated:
                break

        return documents, scanned_files, skipped_entries, truncated

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


@dataclass(frozen=True, slots=True)
class _ScannedDocument:
    root_label: str
    relative_path: str
    content_sha256: str
    lines: list[str]


@dataclass(slots=True)
class _Unit:
    """A retrievable unit: one markdown heading line, or one passage of a section."""

    document: _ScannedDocument
    line_start: int
    line_end: int
    text: str
    is_heading: bool
    terms: list[str]
    context_terms: set[str]
    document_terms: set[str]
    heading_text: str | None = None
    coverage: int = 0
    score: float = 0.0


# Words that carry no retrieval signal in natural-language questions and instructions.
_INSTRUCTION_WORDS = frozenset(
    "after against before below beside beyond during per since through toward under until "
    "upon via within without "
    "about all also and any are cite cited citing did does doing done from give how into "
    "its many much only please show source sources supporting tell than that the their them "
    "then there these this those using use was were what when where which while who whom "
    "whose why will with".split()
)
# Corpus-level words: on their own they ask for an overview rather than a fact.
_GENERIC_TERMS = frozenset(
    "available content contents document documents file files local overview".split()
)
_SUMMARY_INTENT = frozenset("summarize summarise summary summaries overview".split())
# Comparisons need breadth: each side's facts rarely repeat every query term.
_COMPARE_INTENT = frozenset("compare comparing comparison contrast versus vs difference differences between".split())
# Closing punctuation, optionally followed by a closing quote or bracket.
_ENDS_SENTENCE = re.compile(r"[.!?][\"'”’)\]]*$")
# The same, inside running text, followed by the start of another sentence.
_SENTENCE_BREAK = re.compile(r"[.!?][\"'”’)\]]*\s+(?=[\"“‘(\[]?[A-Z0-9])")
_BM25_K1 = 1.2
_BM25_B = 0.75
_CONTEXT_WEIGHT = 0.5
# Passages scoring below this share of the best passage are noise, not evidence.
_MIN_RELATIVE_SCORE = 0.35
# A passage that follows a break repeats up to this much of the text before it, so a
# title, a question, or the sentence naming the subject is not cut off from what follows.
_CARRY_CHARS = 200
# Where a section may be divided, best first.
_BREAK_PARAGRAPH = 0  # after a blank line
_BREAK_LINE_SENTENCE = 1  # a new line after a finished sentence
_BREAK_SENTENCE = 2  # between two sentences on one line
_BREAK_LINE = 3  # a new line in the middle of a sentence


def _tokens(text: str) -> list[str]:
    return [token for token in re.findall(r"[a-z0-9]+", text.casefold()) if len(token) >= 2]


def _content_terms(text: str) -> list[str]:
    return [
        _stem(token)
        for token in _tokens(text)
        if token not in QUERY_STOPWORDS and token not in _INSTRUCTION_WORDS
    ]


def _query_profile(query: str) -> tuple[list[str], bool, bool]:
    """Stemmed content terms (ordered, unique), overview intent, and comparison intent."""

    raw = _tokens(query)
    summary_intent = any(token in _SUMMARY_INTENT for token in raw)
    compare_intent = any(token in _COMPARE_INTENT for token in raw)
    terms = [
        _stem(token)
        for token in raw
        if token not in QUERY_STOPWORDS
        and token not in _INSTRUCTION_WORDS
        and token not in _GENERIC_TERMS
        and token not in _SUMMARY_INTENT
        and token not in _COMPARE_INTENT
    ]
    return list(dict.fromkeys(terms)), summary_intent, compare_intent


MeaningScorer = Callable[[list[_Unit]], "list[float | None] | None"]


def _rank_units(
    documents: list[_ScannedDocument],
    query: str,
    *,
    limit: int,
    case_sensitive: bool,
    passage_chars: int,
    max_result_chars: int,
    meaning: MeaningScorer | None = None,
) -> tuple[list[_Unit], bool]:
    """Keyword ranking, interleaved with meaning ranking when a scorer is available.

    Keyword (BM25) ranking finds the passages that use the question's words; meaning
    ranking finds those that say the same thing in other words. Alternating the two
    lists (best keyword match, best meaning match, next keyword match, ...) kept every
    original question's answer in the top three and found reworded questions' answers
    that keywords missed. When no passage shares a word with the question, the meaning
    matches are used alone. Comparison and overview requests keep their keyword logic.
    """

    units = _units(documents, passage_chars)
    keyword, mode = _keyword_order(units, query, case_sensitive)
    if meaning is None or mode in {"compare", "overview"}:
        return _bounded(keyword, limit, max_result_chars)
    by_meaning = _meaning_order(units, meaning)
    if not by_meaning:
        return _bounded(keyword, limit, max_result_chars)
    if mode == "none":
        return _bounded(by_meaning, limit, max_result_chars)
    return _bounded(_interleave(keyword, by_meaning), limit, max_result_chars)


# Meaning matches kept: at most this many, within this similarity of the best one.
_MEANING_CANDIDATES = 10
_MEANING_MARGIN = 0.15


def _meaning_order(units: list[_Unit], meaning: MeaningScorer) -> list[_Unit]:
    passages = [unit for unit in units if not unit.is_heading]
    scores = meaning(passages) if passages else None
    if not scores:
        return []
    scored = [(score, index) for index, score in enumerate(scores) if score is not None]
    if not scored:
        return []
    scored.sort(key=lambda pair: (-pair[0], pair[1]))
    best = scored[0][0]
    return [
        passages[index] for score, index in scored[:_MEANING_CANDIDATES] if score >= best - _MEANING_MARGIN
    ]


def _interleave(first: list[_Unit], second: list[_Unit]) -> list[_Unit]:
    merged: list[_Unit] = []
    seen: set[int] = set()
    for position in range(max(len(first), len(second))):
        for ordered in (first, second):
            if position < len(ordered) and id(ordered[position]) not in seen:
                seen.add(id(ordered[position]))
                merged.append(ordered[position])
    return merged


def _keyword_order(units: list[_Unit], query: str, case_sensitive: bool) -> tuple[list[_Unit], str]:
    """Passages ordered by BM25 with heading and file-name context, and how they were chosen.

    Documents are searched as passages (see ``_passages``), so a match brings the
    sentences around it: the rest of a wrapped sentence, the answer under a question,
    the name under a quotation. A passage also matches on its document context (nearest
    markdown heading and file name), so "The trial completed in 3 hours." under
    "# Project Beacon" still answers a Beacon query.

    Modes: "ranked" (normal), "compare" (both sides of a comparison), "overview" (no
    content words, or an overview request matching nothing: the documents' openings),
    and "none" (a fact question no passage shares a word with).
    """

    terms, summary_intent, compare_intent = _query_profile(query)
    if not terms:
        return units, "overview"
    term_set = set(terms)
    document_frequency = {
        term: sum(1 for unit in units if term in unit.context_terms or term in unit.terms)
        for term in term_set
    }
    total = max(1, len(units))
    average_length = sum(len(unit.terms) for unit in units) / total or 1.0
    needle = " ".join(query.split()) if case_sensitive else " ".join(query.casefold().split())
    for unit in units:
        counts: dict[str, int] = {}
        for token in unit.terms:
            if token in term_set:
                counts[token] = counts.get(token, 0) + 1
        present = {term for term in term_set if term in counts or term in unit.context_terms}
        unit.coverage = len(present)
        length_norm = 1 - _BM25_B + _BM25_B * len(unit.terms) / average_length
        score = 0.0
        for term in present:
            frequency = counts.get(term, 0) + (_CONTEXT_WEIGHT if term in unit.context_terms else 0.0)
            df = document_frequency[term]
            idf = math.log(1 + (total - df + 0.5) / (df + 0.5))
            score += idf * frequency * (_BM25_K1 + 1) / (frequency + _BM25_K1 * length_norm)
        haystack = unit.text if case_sensitive else unit.text.casefold()
        if len(needle) >= 3 and needle in haystack:
            score += 1.0  # literal phrase bonus
        unit.score = score
    matching = [unit for unit in units if unit.coverage]
    if not matching:
        # Nothing matches: an overview request gets the documents' openings; a fact question gets none.
        return (units, "overview") if summary_intent else ([], "none")
    if compare_intent:
        compared = _comparison_units(units, term_set)
        if compared:
            return compared, "compare"
    # Headings are navigation, not facts: rank them after every passage.
    matching.sort(key=lambda unit: (unit.is_heading, -unit.score))  # stable: ties keep document order
    floor = matching[0].score * _MIN_RELATIVE_SCORE
    return [unit for unit in matching if unit.score >= floor], "ranked"


def _bounded(units: list[_Unit], limit: int, max_chars: int) -> tuple[list[_Unit], bool]:
    """Leading units within the count and character limits (never fewer than one)."""

    selected: list[_Unit] = []
    used = 0
    for unit in units[:limit]:
        if selected and used + len(unit.text) > max_chars:
            break
        selected.append(unit)
        used += len(unit.text)
    return selected, len(units) > len(selected)


def _comparison_units(units: list[_Unit], terms: set[str]) -> list[_Unit]:
    """For comparisons: every passage in a compared subject's documents that states at
    least one compared attribute.

    Subjects are query terms that name a document (its heading or file name); attributes
    are the remaining terms. Each side's facts sit in its own document and repeat few of
    the query's terms, so ranking by score alone would favour one side.
    """

    subjects = {term for term in terms if any(term in unit.document_terms for unit in units)}
    attributes = terms - subjects
    if len(subjects) < 2 or not attributes:
        return []
    selected = [
        unit
        for unit in units
        if unit.document_terms & subjects and set(unit.terms) & attributes and not unit.is_heading
    ]
    selected.sort(key=lambda unit: (-unit.coverage, -unit.score))
    return selected


def _plain_title(lines: list[str]) -> str | None:
    first = next((" ".join(line.split()) for line in lines if line.strip()), "")
    if not first or first.startswith("#") or len(first) > 80 or _ENDS_SENTENCE.search(first):
        return None
    return first


def _units(documents: list[_ScannedDocument], passage_chars: int) -> list[_Unit]:
    units: list[_Unit] = []
    for document in documents:
        name_terms = set(_content_terms(PurePosixPath(document.relative_path).stem))
        heading_terms: set[str] = set()
        # A plain-text document's short first line ("NORTHWIND FERRY COOPERATIVE") is its
        # title until a markdown heading says otherwise. It is reported as provenance only;
        # it does not change ranking.
        heading_text: str | None = _plain_title(document.lines)
        section: list[tuple[int, str]] = []
        # A trailing heading marker closes the last section.
        for line_number, line in [*enumerate(document.lines, start=1), (0, "#")]:
            text = " ".join(line.split())
            if not text.startswith("#"):
                section.append((line_number, text))
                continue
            for line_start, line_end, passage in _passages(section, passage_chars):
                terms = _content_terms(passage)
                units.append(
                    _Unit(
                        document=document,
                        line_start=line_start,
                        line_end=line_end,
                        text=passage,
                        is_heading=False,
                        terms=terms,
                        context_terms=(heading_terms | name_terms) - set(terms),
                        document_terms=heading_terms | name_terms,
                        heading_text=heading_text,
                    )
                )
            section = []
            if line_number == 0:
                break
            heading_text = text.lstrip("#").strip()[:200] or None
            terms = _content_terms(text)
            heading_terms = set(terms)
            units.append(
                _Unit(
                    document=document,
                    line_start=line_number,
                    line_end=line_number,
                    text=text,
                    is_heading=True,
                    terms=terms,
                    context_terms=name_terms - heading_terms,
                    document_terms=heading_terms | name_terms,
                    heading_text=heading_text,
                )
            )
    return units


def _passages(lines: list[tuple[int, str]], target: int) -> list[tuple[int, int, str]]:
    """Divide one section into passages: (first line, last line, text).

    The section's lines are joined into running text, so line wrapping (as in text
    extracted from a PDF) does not matter. A section up to one and a half times
    ``target`` characters stays whole. A longer one is divided into passages of about
    ``target`` characters at the best break available near that length: a blank line,
    else the end of a sentence, else a line break, else a space.
    """

    parts: list[str] = []
    line_offsets: list[int] = []
    line_numbers: list[int] = []
    breaks: dict[int, int] = {}
    length = 0
    after_blank = False
    for line_number, text in lines:
        if not text:
            after_blank = True
            continue
        if parts:
            length += 1  # the joining space
            if after_blank:
                breaks[length] = _BREAK_PARAGRAPH
            elif _ENDS_SENTENCE.search(parts[-1]):
                breaks[length] = _BREAK_LINE_SENTENCE
            else:
                breaks[length] = _BREAK_LINE
        line_offsets.append(length)
        line_numbers.append(line_number)
        parts.append(text)
        length += len(text)
        after_blank = False
    if not parts:
        return []
    joined = " ".join(parts)
    for found in _SENTENCE_BREAK.finditer(joined):
        breaks.setdefault(found.end(), _BREAK_SENTENCE)
    ordered = sorted(breaks)

    spans: list[tuple[int, int]] = []
    start = 0
    hard_limit = target * 3 // 2
    while len(joined) - start > hard_limit:
        remaining = len(joined) - start
        aim = start + remaining // math.ceil(remaining / target)
        in_range = ordered[
            bisect.bisect_left(ordered, start + target // 2) : bisect.bisect_right(ordered, start + hard_limit)
        ]
        end, kind = 0, _BREAK_LINE
        for wanted in (_BREAK_PARAGRAPH, _BREAK_LINE_SENTENCE, _BREAK_SENTENCE, _BREAK_LINE):
            candidates = [offset for offset in in_range if breaks[offset] == wanted]
            if candidates:
                end, kind = min(candidates, key=lambda offset: abs(offset - aim)), wanted
                break
        if not end:
            space = joined.rfind(" ", start + target // 2, start + target)
            end = space + 1 if space >= 0 else start + target
        spans.append((start, end))
        start = _carry_start(ordered, breaks, start, end, kind)
    spans.append((start, len(joined)))

    passages: list[tuple[int, int, str]] = []
    for span_start, span_end in spans:
        text = joined[span_start:span_end].strip()
        if not text:
            continue
        first = bisect.bisect_right(line_offsets, span_start) - 1
        last = bisect.bisect_right(line_offsets, span_start + len(joined[span_start:span_end].rstrip()) - 1) - 1
        passages.append((line_numbers[first], line_numbers[last], text))
    return passages


def _carry_start(ordered: list[int], breaks: dict[int, int], start: int, end: int, kind: int) -> int:
    """Where the passage after a break begins: at the break, or up to ``_CARRY_CHARS`` earlier.

    After a blank line, a short closing paragraph (a title, a question) is repeated
    whole; a long one is not. After a sentence break, the last sentence or two are.
    """

    if kind == _BREAK_PARAGRAPH:
        index = bisect.bisect_left(ordered, end) - 1
        while index >= 0 and breaks[ordered[index]] != _BREAK_PARAGRAPH:
            index -= 1
        paragraph = ordered[index] if index >= 0 else 0
        return paragraph if paragraph > start and end - paragraph <= _CARRY_CHARS else end
    if kind in (_BREAK_LINE_SENTENCE, _BREAK_SENTENCE):
        index = bisect.bisect_left(ordered, end - _CARRY_CHARS)
        while index < len(ordered) and ordered[index] < end:
            if ordered[index] > start and breaks[ordered[index]] != _BREAK_LINE:
                return ordered[index]
            index += 1
    return end


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
