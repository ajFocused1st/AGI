"""Deterministic network-free model adapter for integration tests."""

from __future__ import annotations

from collections import deque
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

from agi_contracts import ModelRequest, ModelResponse, ModelUsage

from .adapter import BaseModelAdapter, validate_structured_output
from .errors import OllamaResponseValidationError


@dataclass(frozen=True, slots=True)
class ScriptedReply:
    """One provider-neutral reply consumed in FIFO order."""

    content: str
    structured_output: Mapping[str, Any] | None = None
    model: str | None = None
    usage: ModelUsage | None = None
    raw_metadata: Mapping[str, Any] = field(default_factory=dict)


class ScriptedModelAdapter(BaseModelAdapter):
    """Records requests and returns a finite deterministic response script."""

    def __init__(self, replies: Iterable[ScriptedReply]) -> None:
        self._replies = deque(replies)
        self.requests: list[ModelRequest] = []

    @property
    def remaining_replies(self) -> int:
        return len(self._replies)

    async def generate(self, request: ModelRequest) -> ModelResponse:
        self.requests.append(request.model_copy(deep=True))
        if not self._replies:
            raise RuntimeError("ScriptedModelAdapter reply script is exhausted")
        reply = self._replies.popleft()

        structured_output = (
            dict(reply.structured_output)
            if reply.structured_output is not None
            else None
        )
        if request.response_schema is not None:
            if structured_output is None:
                structured_output = validate_structured_output(
                    reply.content,
                    request.response_schema,
                )
            else:
                # Re-encode so the same strict JSON/schema path is exercised as
                # the real adapter, including the object-only invariant.
                import json

                try:
                    encoded = json.dumps(structured_output, allow_nan=False)
                except (TypeError, ValueError) as exc:
                    raise OllamaResponseValidationError(
                        "Scripted structured output is not strict JSON"
                    ) from exc
                structured_output = validate_structured_output(
                    encoded,
                    request.response_schema,
                )

        metadata = {"provider": "scripted", **dict(reply.raw_metadata)}
        return ModelResponse(
            request_id=request.request_id,
            model=reply.model or request.model,
            content=reply.content,
            structured_output=structured_output,
            usage=reply.usage
            or ModelUsage(
                input_tokens=None,
                output_tokens=None,
                total_tokens=None,
                estimated_cost=None,
            ),
            raw_metadata=metadata,
            created_at=datetime.now(timezone.utc),
        )
