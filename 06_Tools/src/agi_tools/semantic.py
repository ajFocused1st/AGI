"""Meaning-based passage similarity from a local Ollama embedding model.

Keyword search misses a passage that answers a question in different words ("founding
document" for "charter"). An embedding model places texts with similar meaning close
together, so the search tool can also rank passages by meaning. Passage vectors are
computed once per passage text and cached on disk; a search embeds only its query.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import threading
import urllib.error
import urllib.request
from collections.abc import Sequence
from pathlib import Path
from time import perf_counter

_BATCH = 64


class EmbeddingUnavailableError(RuntimeError):
    """The embedding model could not be reached or returned something unusable."""


class OllamaPassageEmbedder:
    """Embeds passages and queries with an installed Ollama embedding model."""

    def __init__(
        self,
        *,
        base_url: str,
        model: str,
        cache_path: Path,
        timeout_seconds: float = 120.0,
        keep_alive: str = "30m",
    ) -> None:
        if not model.strip():
            raise ValueError("embedding model name must not be empty")
        self.model = model.strip()
        self._url = base_url.rstrip("/") + "/api/embed"
        self._cache_path = cache_path
        self._timeout = timeout_seconds
        self._keep_alive = keep_alive
        # nomic-embed-text is trained with task prefixes; other models take plain text.
        nomic = "nomic" in self.model.casefold()
        self._query_prefix = "search_query: " if nomic else ""
        self._document_prefix = "search_document: " if nomic else ""
        self._vectors: dict[str, list[float]] | None = None
        self._lock = threading.Lock()

    # --- public ---------------------------------------------------------------------

    def ensure(self, texts: Sequence[str], *, deadline: float | None = None) -> int:
        """Embed every passage not yet cached; returns how many were newly embedded."""

        with self._lock:
            vectors = self._load()
            missing = list(dict.fromkeys(t for t in texts if _key(t) not in vectors))
            for start in range(0, len(missing), _BATCH):
                if deadline is not None and perf_counter() >= deadline:
                    break
                batch = missing[start : start + _BATCH]
                for text, vector in zip(batch, self._embed([self._document_prefix + t for t in batch])):
                    vectors[_key(text)] = vector
            if missing:
                self._save(vectors)
            return len(missing)

    def similarities(
        self, query: str, texts: Sequence[str], *, deadline: float | None = None
    ) -> list[float | None]:
        """Cosine similarity of the query to each passage (None where not yet embedded)."""

        self.ensure(texts, deadline=deadline)
        (query_vector,) = self._embed([self._query_prefix + query])
        vectors = self._load()
        return [
            _cosine(query_vector, vectors[_key(text)]) if _key(text) in vectors else None
            for text in texts
        ]

    # --- internals ------------------------------------------------------------------

    def _embed(self, inputs: list[str]) -> list[list[float]]:
        body = json.dumps(
            {"model": self.model, "input": inputs, "keep_alive": self._keep_alive}
        ).encode("utf-8")
        request = urllib.request.Request(
            self._url, data=body, headers={"Content-Type": "application/json"}, method="POST"
        )
        try:
            with urllib.request.urlopen(request, timeout=self._timeout) as response:
                payload = json.load(response)
        except (urllib.error.URLError, TimeoutError, OSError, ValueError) as exc:
            raise EmbeddingUnavailableError(
                f"embedding model {self.model!r} is unavailable: {exc}"
            ) from exc
        embeddings = payload.get("embeddings") if isinstance(payload, dict) else None
        if (
            not isinstance(embeddings, list)
            or len(embeddings) != len(inputs)
            or not all(isinstance(v, list) and v for v in embeddings)
        ):
            raise EmbeddingUnavailableError(
                f"embedding model {self.model!r} returned an unexpected response"
            )
        return [[float(x) for x in vector] for vector in embeddings]

    def _load(self) -> dict[str, list[float]]:
        if self._vectors is None:
            self._vectors = {}
            try:
                stored = json.loads(self._cache_path.read_text(encoding="utf-8"))
                if isinstance(stored, dict) and stored.get("model") == self.model:
                    self._vectors = {
                        str(k): [float(x) for x in v] for k, v in stored.get("vectors", {}).items()
                    }
            except (OSError, ValueError):
                pass  # no cache yet, or an unreadable one: start empty
        return self._vectors

    def _save(self, vectors: dict[str, list[float]]) -> None:
        self._cache_path.parent.mkdir(parents=True, exist_ok=True)
        temporary = self._cache_path.with_suffix(".tmp")
        temporary.write_text(
            json.dumps(
                {"model": self.model, "vectors": {k: [round(x, 6) for x in v] for k, v in vectors.items()}}
            ),
            encoding="utf-8",
        )
        os.replace(temporary, self._cache_path)


def _key(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _cosine(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    norm = math.sqrt(sum(x * x for x in a)) * math.sqrt(sum(y * y for y in b))
    return dot / norm if norm else 0.0
