"""Governed, low-risk tool implementations."""

from .base import BaseTool, ToolInputError
from .calculator import Calculator, CalculatorInput, CalculatorOutput
from .file_search import (
    FileSearchMatch,
    FileSearchOutput,
    ReadOnlyFileSearch,
    ReadOnlyFileSearchConfig,
    ReadOnlyFileSearchInput,
    SearchRoot,
)
from .file_reader import (
    ReadOnlyFileReader,
    ReadOnlyFileReaderConfig,
    ReadOnlyFileReaderInput,
    ReadOnlyFileReaderOutput,
)
from .registry import DecisionRecorder, ToolRegistry
from .semantic import EmbeddingUnavailableError, OllamaPassageEmbedder

__all__ = [
    "BaseTool",
    "Calculator",
    "CalculatorInput",
    "CalculatorOutput",
    "DecisionRecorder",
    "EmbeddingUnavailableError",
    "FileSearchMatch",
    "FileSearchOutput",
    "OllamaPassageEmbedder",
    "ReadOnlyFileSearch",
    "ReadOnlyFileSearchConfig",
    "ReadOnlyFileSearchInput",
    "ReadOnlyFileReader",
    "ReadOnlyFileReaderConfig",
    "ReadOnlyFileReaderInput",
    "ReadOnlyFileReaderOutput",
    "SearchRoot",
    "ToolInputError",
    "ToolRegistry",
]
