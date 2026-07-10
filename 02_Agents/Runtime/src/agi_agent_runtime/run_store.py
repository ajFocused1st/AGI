"""Atomic, redacted JSON persistence for run audit records."""

from __future__ import annotations

import asyncio
import json
import os
from pathlib import Path
from typing import Any
from uuid import uuid4

from agi_contracts import RunRecord


_SENSITIVE_KEYS = frozenset(
    {
        "api_key",
        "apikey",
        "authorization",
        "credential",
        "credentials",
        "password",
        "private_key",
        "secret",
        "token",
    }
)


class AtomicJsonRunStore:
    """Write each run exactly once beneath a configured runtime log root."""

    def __init__(self, root: str | Path, *, forbidden_root: str | Path | None = None) -> None:
        self.root = Path(root).expanduser().resolve()
        if forbidden_root is not None:
            forbidden = Path(forbidden_root).expanduser().resolve()
            if _is_within(self.root, forbidden):
                raise ValueError("run storage must be outside the source workspace")

    async def save(self, record: RunRecord) -> Path:
        return await asyncio.to_thread(self._save_sync, record)

    def _save_sync(self, record: RunRecord) -> Path:
        run_directory = self.root / str(record.run_id)
        destination = run_directory / "run.json"
        run_directory.mkdir(parents=True, exist_ok=True)
        if destination.exists():
            raise FileExistsError(f"run record already exists: {record.run_id}")

        payload = _redact(record.model_dump(mode="json"))
        encoded = json.dumps(
            payload,
            ensure_ascii=False,
            allow_nan=False,
            indent=2,
            sort_keys=True,
        ) + "\n"
        temporary = run_directory / f".{record.run_id}.{uuid4().hex}.tmp"
        try:
            with temporary.open("x", encoding="utf-8", newline="\n") as handle:
                handle.write(encoded)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temporary, destination)
        finally:
            temporary.unlink(missing_ok=True)
        return destination


def _redact(value: Any, *, key: str | None = None) -> Any:
    if key is not None and key.casefold() in _SENSITIVE_KEYS:
        return "[REDACTED]"
    if isinstance(value, dict):
        return {item_key: _redact(item, key=item_key) for item_key, item in value.items()}
    if isinstance(value, list):
        return [_redact(item) for item in value]
    return value


def _is_within(candidate: Path, root: Path) -> bool:
    candidate_text = os.path.normcase(os.path.abspath(candidate))
    root_text = os.path.normcase(os.path.abspath(root))
    try:
        return os.path.commonpath((candidate_text, root_text)) == root_text
    except ValueError:
        return False
