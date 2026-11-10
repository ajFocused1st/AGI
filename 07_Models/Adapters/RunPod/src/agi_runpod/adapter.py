"""RunPod Serverless vLLM implementation of the shared model boundary."""

from __future__ import annotations

import asyncio
import json
from collections.abc import Awaitable, Callable, Mapping
from datetime import datetime, timezone
from time import perf_counter
from typing import Any

from agi_contracts import ModelRequest, ModelResponse, ModelUsage
from jsonschema import Draft202012Validator
from jsonschema.exceptions import SchemaError, ValidationError

from .config import RunPodConfig
from .errors import (
    RunPodAdapterError, RunPodConnectionError, RunPodHTTPError, RunPodProtocolError,
    RunPodRequestError, RunPodResponseValidationError, RunPodTimeoutError,
)
from .telemetry import http_status_category, structured_output_telemetry
from .transport import AsyncHttpTransport, HttpResponse, UrllibHttpTransport

_RETRYABLE = frozenset({408, 429, 500, 502, 503, 504})


class RunPodAdapter:
    def __init__(
        self, config: RunPodConfig, *, transport: AsyncHttpTransport | None = None,
        sleeper: Callable[[float], Awaitable[None]] = asyncio.sleep,
    ) -> None:
        self._config = config
        self._transport = transport or UrllibHttpTransport()
        self._sleeper = sleeper

    @property
    def config(self) -> RunPodConfig:
        return self._config

    async def generate(self, request: ModelRequest) -> ModelResponse:
        payload = self._build_payload(request)
        for validation_attempt in range(self._config.structured_output_retries + 1):
            started = perf_counter()
            response = await self._post_with_retries(payload)
            latency_ms = (perf_counter() - started) * 1000
            try:
                return self._parse_success(
                    request, response, latency_ms=latency_ms,
                    attempt=validation_attempt,
                )
            except RunPodResponseValidationError as exc:
                if request.response_schema is None or validation_attempt >= self._config.structured_output_retries:
                    raise
                payload = self._build_payload(request, corrective_error=str(exc))
        raise AssertionError("bounded validation loop ended unexpectedly")

    async def _post_with_retries(self, payload: Mapping[str, Any]) -> HttpResponse:
        last_error: RunPodAdapterError | None = None
        for attempt in range(self._config.max_retries + 1):
            try:
                response = await asyncio.wait_for(
                    self._transport.post_json(
                        url=f"{self._config.base_url}/chat/completions",
                        payload=payload,
                        authorization=f"Bearer {self._config.api_key.get_secret_value()}",
                        timeout_seconds=self._config.timeout_seconds,
                        max_response_bytes=self._config.max_response_bytes,
                    ), timeout=self._config.timeout_seconds,
                )
                if not 200 <= response.status_code < 300:
                    raise self._http_error(response)
                return response
            except asyncio.CancelledError:
                raise
            except asyncio.TimeoutError as exc:
                last_error = RunPodTimeoutError("Timed out waiting for RunPod")
                last_error.__cause__ = exc
            except (RunPodTimeoutError, RunPodConnectionError) as exc:
                last_error = exc
            except RunPodHTTPError as exc:
                last_error = exc
                if exc.status_code not in _RETRYABLE:
                    raise
            if attempt >= self._config.max_retries:
                assert last_error is not None
                raise last_error
            delay = min(self._config.retry_backoff_seconds * (2**attempt), 5.0)
            if delay:
                await self._sleeper(delay)
        raise AssertionError("bounded retry loop ended unexpectedly")

    def _build_payload(self, request: ModelRequest, *, corrective_error: str | None = None) -> dict[str, Any]:
        if request.model != self._config.model:
            raise RunPodRequestError("Requested model is not the configured RunPod model")
        if request.max_tokens > self._config.max_tokens_per_request:
            raise RunPodRequestError("Requested tokens exceed the RunPod adapter ceiling")
        if sum(len(item.content) for item in request.messages) > self._config.max_prompt_characters:
            raise RunPodRequestError("Prompt exceeds the RunPod adapter ceiling")
        messages = [
            {"role": str(getattr(item.role, "value", item.role)), "content": item.content}
            for item in request.messages
        ]
        payload: dict[str, Any] = {
            "model": self._config.model, "messages": messages,
            "temperature": request.temperature, "max_tokens": request.max_tokens,
            "stream": False,
        }
        if request.seed is not None:
            payload["seed"] = request.seed
        if request.response_schema is not None:
            _schema_validator(request.response_schema)
            payload["response_format"] = {
                "type": "json_schema",
                "json_schema": {"name": "agi_response", "strict": True, "schema": request.response_schema},
            }
            messages.insert(0, _json_instruction(request.response_schema))
            if corrective_error is not None:
                messages.append({
                    "role": "user",
                    "content": "The previous response failed host validation. Return only one object matching the supplied schema. No prose or Markdown.",
                })
        try:
            json.dumps(payload, ensure_ascii=False, allow_nan=False)
        except (TypeError, ValueError) as exc:
            raise RunPodRequestError("Request is not strict JSON serializable") from exc
        return payload

    def _http_error(self, response: HttpResponse) -> RunPodHTTPError:
        detail = _safe_error_detail(response.body)
        secret = self._config.api_key.get_secret_value()
        if secret:
            detail = detail.replace(secret, "[REDACTED]")
        return RunPodHTTPError(response.status_code, http_status_category(response.status_code), detail)

    @staticmethod
    def _parse_success(request: ModelRequest, response: HttpResponse, *, latency_ms: float, attempt: int) -> ModelResponse:
        try:
            decoded = _strict_loads(response.body.decode("utf-8"))
            choice = decoded["choices"][0]
            message = choice["message"]
            content = message.get("content") or ""
        except (UnicodeDecodeError, json.JSONDecodeError, ValueError, KeyError, IndexError, TypeError) as exc:
            raise RunPodProtocolError("RunPod returned an invalid chat-completions envelope") from exc
        if not isinstance(decoded, dict) or not isinstance(choice, dict) or not isinstance(message, dict) or not isinstance(content, str):
            raise RunPodProtocolError("RunPod returned invalid response field types")
        structured = None
        if request.response_schema is not None:
            structured = _validate_output(content, request.response_schema)
        usage_raw = decoded.get("usage") if isinstance(decoded.get("usage"), dict) else {}
        usage = ModelUsage(
            input_tokens=_optional_nonnegative_int(usage_raw.get("prompt_tokens")),
            output_tokens=_optional_nonnegative_int(usage_raw.get("completion_tokens")),
            total_tokens=_optional_nonnegative_int(usage_raw.get("total_tokens")),
        )
        native_calls = _untrusted_tool_calls(message.get("tool_calls"))
        metadata: dict[str, Any] = {
            "provider": "runpod", "latency_ms": latency_ms,
            "finish_reason": choice.get("finish_reason"),
            "http_status_category": http_status_category(response.status_code),
            "structured_output": structured_output_telemetry(attempt, valid=True),
        }
        if native_calls:
            metadata["untrusted_tool_call_drafts"] = native_calls
        return ModelResponse(
            request_id=request.request_id, model=str(decoded.get("model") or request.model),
            content=content, structured_output=structured, usage=usage,
            raw_metadata={key: value for key, value in metadata.items() if value is not None},
            created_at=datetime.now(timezone.utc),
        )


def _validate_output(content: str, schema: Mapping[str, Any]) -> dict[str, Any]:
    try:
        parsed = _strict_loads(content)
    except (json.JSONDecodeError, ValueError) as exc:
        raise RunPodResponseValidationError("Model output was not strict JSON") from exc
    if not isinstance(parsed, dict):
        raise RunPodResponseValidationError("Structured output must be an object")
    try:
        _schema_validator(schema).validate(parsed)
    except ValidationError as exc:
        path = ".".join(str(part) for part in exc.absolute_path) or "<root>"
        raise RunPodResponseValidationError(f"Structured output failed schema validation at {path}") from exc
    return parsed


def _schema_validator(schema: Mapping[str, Any]) -> Draft202012Validator:
    _reject_external_references(schema)
    try:
        Draft202012Validator.check_schema(schema)
    except SchemaError as exc:
        raise RunPodRequestError("Invalid response JSON Schema") from exc
    return Draft202012Validator(schema)


def _reject_external_references(value: Any) -> None:
    if isinstance(value, Mapping):
        for keyword in ("$ref", "$dynamicRef", "$recursiveRef"):
            reference = value.get(keyword)
            if isinstance(reference, str) and not reference.startswith("#"):
                raise RunPodRequestError("External JSON Schema references are not allowed")
        for child in value.values():
            _reject_external_references(child)
    elif isinstance(value, list):
        for child in value:
            _reject_external_references(child)


def _json_instruction(schema: Mapping[str, Any]) -> dict[str, str]:
    return {"role": "system", "content": "Return exactly one JSON object matching this schema: " + json.dumps(schema, separators=(",", ":"))}


def _strict_loads(value: str) -> Any:
    return json.loads(value, parse_constant=lambda item: (_ for _ in ()).throw(ValueError(item)), object_pairs_hook=_unique_object)


def _unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("Duplicate JSON key")
        result[key] = value
    return result


def _optional_nonnegative_int(value: Any) -> int | None:
    return value if isinstance(value, int) and not isinstance(value, bool) and value >= 0 else None


def _untrusted_tool_calls(value: Any) -> list[dict[str, Any]]:
    if not isinstance(value, list):
        return []
    drafts: list[dict[str, Any]] = []
    for item in value[:100]:
        function = item.get("function") if isinstance(item, dict) else None
        if not isinstance(function, dict) or not isinstance(function.get("name"), str):
            continue
        arguments = function.get("arguments", {})
        if isinstance(arguments, str):
            try:
                arguments = _strict_loads(arguments)
            except (json.JSONDecodeError, ValueError):
                arguments = {}
        if isinstance(arguments, dict):
            drafts.append({"tool_name": function["name"], "arguments": arguments, "purpose": None, "related_step_id": None})
    return drafts


def _safe_error_detail(body: bytes) -> str:
    text = body.decode("utf-8", errors="replace")[:4000]
    try:
        decoded = _strict_loads(text)
    except (json.JSONDecodeError, ValueError):
        return "provider rejected the request"
    if isinstance(decoded, dict):
        error = decoded.get("error")
        if isinstance(error, dict) and isinstance(error.get("message"), str):
            return error["message"][:500]
        if isinstance(error, str):
            return error[:500]
    return "provider rejected the request"
