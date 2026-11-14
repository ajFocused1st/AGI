"""Generic model boundary and its local Ollama implementation."""

from __future__ import annotations

import asyncio
import json
from abc import ABC, abstractmethod
from collections.abc import Awaitable, Callable, Mapping
from datetime import datetime, timezone
from typing import Any, Protocol, runtime_checkable

from agi_contracts import ModelRequest, ModelResponse, ModelUsage
from jsonschema import Draft202012Validator
from jsonschema.exceptions import SchemaError, ValidationError
from pydantic import BaseModel, ConfigDict, Field, ValidationError as PydanticValidationError

from .config import OllamaConfig
from .errors import (
    OllamaAdapterError,
    OllamaConnectionError,
    OllamaHTTPError,
    OllamaModelNotFoundError,
    OllamaProtocolError,
    OllamaRequestError,
    OllamaResponseValidationError,
    OllamaTimeoutError,
)
from .transport import AsyncHttpTransport, HttpResponse, UrllibHttpTransport

_RETRYABLE_STATUS_CODES = frozenset({408, 429, 500, 502, 503, 504})


@runtime_checkable
class ModelAdapter(Protocol):
    """Provider-neutral model interface consumed by the agent runtime."""

    async def generate(self, request: ModelRequest) -> ModelResponse:
        """Generate one response for a validated shared request."""


class BaseModelAdapter(ABC):
    """Optional nominal base for implementations that prefer an ABC."""

    @abstractmethod
    async def generate(self, request: ModelRequest) -> ModelResponse:
        """Generate one response for a validated shared request."""


class _OllamaMessage(BaseModel):
    model_config = ConfigDict(extra="ignore")

    role: str | None = None
    content: str


class _OllamaChatEnvelope(BaseModel):
    model_config = ConfigDict(extra="allow")

    model: str
    created_at: str | None = None
    message: _OllamaMessage
    done: bool
    done_reason: str | None = None
    total_duration: int | None = Field(default=None, ge=0)
    load_duration: int | None = Field(default=None, ge=0)
    prompt_eval_count: int | None = Field(default=None, ge=0)
    prompt_eval_duration: int | None = Field(default=None, ge=0)
    eval_count: int | None = Field(default=None, ge=0)
    eval_duration: int | None = Field(default=None, ge=0)


class OllamaAdapter(BaseModelAdapter):
    """Finite, loopback-only adapter for Ollama's non-streaming chat API."""

    def __init__(
        self,
        config: OllamaConfig | None = None,
        *,
        transport: AsyncHttpTransport | None = None,
        sleeper: Callable[[float], Awaitable[None]] = asyncio.sleep,
    ) -> None:
        self._config = config or OllamaConfig()
        self._transport = transport or UrllibHttpTransport()
        self._sleeper = sleeper
        self._schema_format_supported = True

    @property
    def config(self) -> OllamaConfig:
        return self._config

    async def generate(self, request: ModelRequest) -> ModelResponse:
        payload = self._build_payload(request)
        structured_error: OllamaResponseValidationError | None = None

        for validation_attempt in range(self._config.structured_output_retries + 1):
            schema_mode_fallback = False
            try:
                response = await self._post_with_retries(payload)
            except OllamaHTTPError as exc:
                if not _is_grammar_parse_error(exc) or request.response_schema is None:
                    raise
                self._schema_format_supported = False
                payload = dict(payload)
                payload["format"] = "json"
                schema_mode_fallback = True
                response = await self._post_with_retries(payload)
            try:
                return self._parse_success(
                    request,
                    response,
                    structured_output_telemetry=_structured_output_telemetry(
                        request=request,
                        validation_attempt=validation_attempt,
                        final_valid=True,
                        schema_mode_fallback=schema_mode_fallback,
                        first_error=structured_error,
                    ),
                )
            except OllamaResponseValidationError as exc:
                structured_error = exc
                if (
                    request.response_schema is None
                    or validation_attempt >= self._config.structured_output_retries
                ):
                    raise
                payload = self._build_payload(
                    request,
                    corrective_error=str(exc),
                )

        assert structured_error is not None
        raise structured_error

    async def _post_with_retries(self, payload: Mapping[str, Any]) -> HttpResponse:
        last_error: OllamaAdapterError | None = None

        for attempt in range(self._config.max_retries + 1):
            try:
                response = await asyncio.wait_for(
                    self._transport.post_json(
                        url=f"{self._config.base_url}/api/chat",
                        payload=payload,
                        timeout_seconds=self._config.timeout_seconds,
                        max_response_bytes=self._config.max_response_bytes,
                    ),
                    timeout=self._config.timeout_seconds,
                )
                if not 200 <= response.status_code < 300:
                    raise self._http_error(response)
                return response
            except asyncio.CancelledError:
                raise
            except asyncio.TimeoutError as exc:
                last_error = OllamaTimeoutError("Timed out waiting for local Ollama")
                last_error.__cause__ = exc
            except (OllamaTimeoutError, OllamaConnectionError) as exc:
                last_error = exc
            except OllamaHTTPError as exc:
                last_error = exc
                if exc.status_code not in _RETRYABLE_STATUS_CODES:
                    raise

            if attempt >= self._config.max_retries:
                assert last_error is not None
                raise last_error
            delay = min(
                self._config.retry_backoff_seconds * (2**attempt),
                5.0,
            )
            if delay:
                await self._sleeper(delay)

        raise AssertionError("bounded retry loop ended unexpectedly")

    def _build_payload(
        self,
        request: ModelRequest,
        *,
        corrective_error: str | None = None,
    ) -> dict[str, Any]:
        model = request.model.strip()
        if not model:
            raise OllamaRequestError("Model name must not be blank")
        if model not in self._config.allowed_models:
            configured = ", ".join(self._config.allowed_models) or "<none>"
            raise OllamaRequestError(
                f"Model {model!r} is not allowed by this adapter; configured models: "
                f"{configured}"
            )
        if request.max_tokens > self._config.max_tokens_per_request:
            raise OllamaRequestError(
                f"Requested {request.max_tokens} tokens exceeds adapter ceiling "
                f"of {self._config.max_tokens_per_request}"
            )
        prompt_characters = sum(len(message.content) for message in request.messages)
        if prompt_characters > self._config.max_prompt_characters:
            raise OllamaRequestError(
                f"Prompt contains {prompt_characters} characters; adapter ceiling is "
                f"{self._config.max_prompt_characters}"
            )

        options: dict[str, Any] = {
            "temperature": request.temperature,
            "num_predict": request.max_tokens,
        }
        if request.seed is not None:
            options["seed"] = request.seed
        if self._config.context_tokens is not None:
            options["num_ctx"] = self._config.context_tokens

        messages = [
            {
                "role": str(getattr(message.role, "value", message.role)),
                "content": message.content,
            }
            for message in request.messages
        ]
        if request.response_schema is not None:
            messages.insert(0, _json_mode_instruction(request.response_schema))
            if corrective_error is not None:
                messages.append(
                    {
                        "role": "user",
                        "content": (
                            "Your previous response could not be used by the host: "
                            f"{corrective_error}. Return only one schema-valid JSON "
                            "object now. No Markdown, no prose, no code fence."
                        ),
                    }
                )

        payload: dict[str, Any] = {
            "model": model,
            "messages": messages,
            "stream": False,
            "options": options,
        }
        if self._config.think is not None:
            payload["think"] = self._config.think
        if request.response_schema is not None:
            _schema_validator(request.response_schema)
            payload["format"] = (
                request.response_schema
                if self._schema_format_supported
                else "json"
            )
        try:
            json.dumps(payload, ensure_ascii=False, allow_nan=False)
        except (TypeError, ValueError) as exc:
            raise OllamaRequestError("Model request is not strict JSON serializable") from exc
        return payload

    @staticmethod
    def _http_error(response: HttpResponse) -> OllamaHTTPError:
        detail = _error_detail(response.body)
        if response.status_code == 404:
            return OllamaModelNotFoundError(
                response.status_code,
                f"{detail} (model was not pulled automatically)",
            )
        return OllamaHTTPError(response.status_code, detail)

    @staticmethod
    def _parse_success(
        request: ModelRequest,
        response: HttpResponse,
        *,
        structured_output_telemetry: dict[str, Any],
    ) -> ModelResponse:
        try:
            decoded = _strict_json_loads(response.body.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError, RecursionError, ValueError) as exc:
            raise OllamaProtocolError("Ollama returned malformed JSON") from exc
        if not isinstance(decoded, dict):
            raise OllamaProtocolError("Ollama response envelope must be a JSON object")
        try:
            envelope = _OllamaChatEnvelope.model_validate(decoded)
        except PydanticValidationError as exc:
            raise OllamaProtocolError("Ollama response envelope failed validation") from exc
        if not envelope.done:
            raise OllamaProtocolError("Ollama returned an incomplete non-streaming response")

        structured_output: dict[str, Any] | None = None
        if request.response_schema is not None:
            structured_output, coercions = parse_structured_output(
                envelope.message.content,
                request.response_schema,
            )
            if coercions:
                structured_output_telemetry = {
                    **structured_output_telemetry,
                    "coercions": coercions[:32],
                }

        input_tokens = envelope.prompt_eval_count
        output_tokens = envelope.eval_count
        total_tokens = (
            input_tokens + output_tokens
            if input_tokens is not None and output_tokens is not None
            else None
        )
        usage = ModelUsage(
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            total_tokens=total_tokens,
            estimated_cost=None,
        )
        raw_metadata = {
            key: value
            for key, value in {
                "provider": "ollama",
                "provider_created_at": envelope.created_at,
                "done_reason": envelope.done_reason,
                "total_duration_ns": envelope.total_duration,
                "load_duration_ns": envelope.load_duration,
                "prompt_eval_duration_ns": envelope.prompt_eval_duration,
                "eval_duration_ns": envelope.eval_duration,
            }.items()
            if value is not None
        }
        raw_metadata["structured_output"] = structured_output_telemetry
        return ModelResponse(
            request_id=request.request_id,
            model=envelope.model,
            content=envelope.message.content,
            structured_output=structured_output,
            usage=usage,
            raw_metadata=raw_metadata,
            created_at=datetime.now(timezone.utc),
        )


def validate_structured_output(content: str, schema: Mapping[str, Any]) -> dict[str, Any]:
    """Parse strict JSON and validate it against a Draft 2020-12 schema."""

    validator = _schema_validator(schema)
    try:
        parsed = _strict_json_loads(content)
    except (json.JSONDecodeError, RecursionError, ValueError) as exc:
        raise OllamaResponseValidationError(
            "Model output was not strict JSON; "
            f"preview={_content_preview(content)!r}"
        ) from exc
    if not isinstance(parsed, dict):
        raise OllamaResponseValidationError(
            "Structured model output must be a JSON object"
        )
    try:
        validator.validate(parsed)
    except ValidationError as exc:
        path = ".".join(str(part) for part in exc.absolute_path) or "<root>"
        raise OllamaResponseValidationError(
            f"Structured model output failed schema validation at {path}: {exc.message}"
        ) from exc
    return parsed


def parse_structured_output(
    content: str, schema: Mapping[str, Any]
) -> tuple[dict[str, Any], list[str]]:
    """Parse, apply conservative schema-driven coercion, then validate strictly.

    Local grammar-constrained decoding does not enforce ``additionalProperties: false``
    or ``minLength``, so small models emit echoed prompt keys or ``""`` for optional
    fields. Coercion only (a) drops keys where the schema forbids extra properties and
    (b) maps ``""`` to ``null`` where the schema allows ``null``. It never invents values;
    everything else must still pass the unchanged strict validator. Changes are returned
    for telemetry.
    """

    try:
        parsed = _strict_json_loads(content)
    except (json.JSONDecodeError, RecursionError, ValueError):
        return validate_structured_output(content, schema), []
    if not isinstance(parsed, dict):
        return validate_structured_output(content, schema), []
    changes: list[str] = []
    coerced = _coerce(parsed, schema, schema, "", changes)
    if not changes:
        return validate_structured_output(content, schema), []
    encoded = json.dumps(coerced, ensure_ascii=False, allow_nan=False)
    return validate_structured_output(encoded, schema), changes


def _resolve(node: Mapping[str, Any], root: Mapping[str, Any]) -> Mapping[str, Any]:
    reference = node.get("$ref")
    if isinstance(reference, str) and reference.startswith("#/"):
        target: Any = root
        for part in reference[2:].split("/"):
            target = target.get(part, {}) if isinstance(target, Mapping) else {}
        return target if isinstance(target, Mapping) else node
    return node


def _allows_null(node: Mapping[str, Any], root: Mapping[str, Any]) -> bool:
    node = _resolve(node, root)
    if node.get("type") == "null" or (isinstance(node.get("type"), list) and "null" in node["type"]):
        return True
    return any(_allows_null(option, root) for option in node.get("anyOf", []) if isinstance(option, Mapping))


def _object_branch(node: Mapping[str, Any], root: Mapping[str, Any]) -> Mapping[str, Any] | None:
    node = _resolve(node, root)
    if node.get("type") == "object" or "properties" in node:
        return node
    for option in node.get("anyOf", []):
        if isinstance(option, Mapping):
            branch = _object_branch(option, root)
            if branch is not None:
                return branch
    return None


def _array_items(node: Mapping[str, Any], root: Mapping[str, Any]) -> Mapping[str, Any] | None:
    node = _resolve(node, root)
    if node.get("type") == "array" and isinstance(node.get("items"), Mapping):
        return node["items"]
    for option in node.get("anyOf", []):
        if isinstance(option, Mapping):
            items = _array_items(option, root)
            if items is not None:
                return items
    return None


def _coerce(
    value: Any,
    node: Mapping[str, Any],
    root: Mapping[str, Any],
    path: str,
    changes: list[str],
) -> Any:
    if value == "" and _allows_null(node, root):
        changes.append(f"empty_to_null:{path or '<root>'}")
        return None
    if isinstance(value, dict):
        branch = _object_branch(node, root)
        if branch is None:
            return value
        properties = branch.get("properties", {})
        result: dict[str, Any] = {}
        for key, item in value.items():
            if key not in properties:
                if branch.get("additionalProperties") is False:
                    changes.append(f"dropped_key:{path + '.' if path else ''}{key[:40]}")
                    continue
                result[key] = item
                continue
            result[key] = _coerce(item, properties[key], root, f"{path + '.' if path else ''}{key}", changes)
        # A required list the model left out entirely means "none": an empty list adds no
        # content, and the host still enforces any capability the task requires.
        for key in branch.get("required", []):
            if key in result or key not in properties:
                continue
            target = _resolve(properties[key], root)
            if target.get("type") == "array" and not target.get("minItems"):
                result[key] = []
                changes.append(f"missing_list_to_empty:{path + '.' if path else ''}{key}")
        return result
    if isinstance(value, list):
        items = _array_items(node, root)
        if items is None:
            return value
        coerced = [_coerce(item, items, root, f"{path}[{index}]", changes) for index, item in enumerate(value)]
        return _drop_incomplete_items(coerced, items, _resolve(node, root), root, path, changes)
    return value


def _drop_incomplete_items(
    values: list[Any],
    items: Mapping[str, Any],
    array_node: Mapping[str, Any],
    root: Mapping[str, Any],
    path: str,
    changes: list[str],
) -> list[Any]:
    """Drop object items missing a required property, only while the array still meets
    its ``minItems``. Small models sometimes emit a stray fragment (for example a plan
    step with no description); dropping it never invents content, and an array that would
    fall below its minimum is left for the strict validator to reject."""

    branch = _object_branch(items, root)
    required = branch.get("required", []) if branch is not None else []
    if not required:
        return values
    kept = [
        value
        for value in values
        if not isinstance(value, dict) or all(key in value for key in required)
    ]
    minimum = array_node.get("minItems", 0) if isinstance(array_node, Mapping) else 0
    if len(kept) == len(values) or len(kept) < max(1, minimum):
        return values
    for index, value in enumerate(values):
        if isinstance(value, dict) and not all(key in value for key in required):
            changes.append(f"dropped_incomplete_item:{path}[{index}]")
    return kept


def _schema_validator(schema: Mapping[str, Any]) -> Draft202012Validator:
    _reject_external_references(schema)
    try:
        encoded = json.dumps(schema, allow_nan=False, separators=(",", ":"))
    except (TypeError, ValueError) as exc:
        raise OllamaRequestError("Response JSON Schema is not strict JSON") from exc
    if len(encoded.encode("utf-8")) > 256 * 1024:
        raise OllamaRequestError("Response JSON Schema exceeds 262144 bytes")
    try:
        Draft202012Validator.check_schema(schema)
    except SchemaError as exc:
        raise OllamaRequestError(f"Invalid response JSON Schema: {exc.message}") from exc
    return Draft202012Validator(schema)


def _json_mode_instruction(schema: Mapping[str, Any]) -> dict[str, str]:
    encoded_schema = json.dumps(
        schema,
        ensure_ascii=False,
        allow_nan=False,
        separators=(",", ":"),
    )
    return {
        "role": "system",
        "content": (
            "You are returning data to a strict JSON parser. Return exactly one JSON "
            "object matching this JSON Schema. Do not include Markdown, prose, code "
            f"fences, or comments. JSON Schema: {encoded_schema}"
        ),
    }


def _content_preview(content: str, *, limit: int = 500) -> str:
    compact = " ".join(content.strip().split())
    if len(compact) <= limit:
        return compact
    return compact[: limit - 3] + "..."


def _structured_output_telemetry(
    *,
    request: ModelRequest,
    validation_attempt: int,
    final_valid: bool,
    schema_mode_fallback: bool = False,
    first_error: OllamaResponseValidationError | None = None,
) -> dict[str, Any]:
    if request.response_schema is None:
        return {
            "first_attempt_valid": True,
            "repair_attempted": False,
            "repair_succeeded": False,
            "final_valid": True,
            "attempt_count": 1,
        }
    telemetry = {
        "first_attempt_valid": validation_attempt == 0 and final_valid,
        "repair_attempted": validation_attempt > 0,
        "repair_succeeded": validation_attempt > 0 and final_valid,
        "final_valid": final_valid,
        "attempt_count": validation_attempt + 1,
    }
    if schema_mode_fallback:
        telemetry["schema_mode_fallback"] = True
    if first_error is not None:
        telemetry["first_error"] = _structured_error_telemetry(first_error)
    return telemetry


def _structured_error_telemetry(error: OllamaResponseValidationError) -> dict[str, str]:
    message = str(error)
    if "not strict JSON" in message:
        return {"category": "not_json"}
    if "must be a JSON object" in message:
        return {"category": "not_object"}
    marker = "failed schema validation at "
    if marker in message:
        remainder = message.split(marker, 1)[1]
        path = remainder.split(":", 1)[0].strip()
        return {"category": "schema_validation", "path": path[:128]}
    return {"category": "unknown"}


def _is_grammar_parse_error(error: OllamaHTTPError) -> bool:
    return (
        error.status_code == 400
        and "failed to parse grammar" in error.detail.casefold()
    )


def _strict_json_loads(value: str) -> Any:
    return json.loads(
        value,
        parse_constant=_reject_json_constant,
        object_pairs_hook=_reject_duplicate_keys,
    )


def _reject_json_constant(constant: str) -> Any:
    raise ValueError(f"Non-standard JSON constant: {constant}")


def _reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"Duplicate JSON object key: {key}")
        result[key] = value
    return result


def _reject_external_references(value: Any) -> None:
    """Prevent JSON Schema validation from resolving anything over a network."""

    if isinstance(value, Mapping):
        for keyword in ("$ref", "$dynamicRef", "$recursiveRef"):
            reference = value.get(keyword)
            if isinstance(reference, str) and not reference.startswith("#"):
                raise OllamaRequestError(
                    "External JSON Schema references are not allowed"
                )
        for child in value.values():
            _reject_external_references(child)
    elif isinstance(value, list):
        for child in value:
            _reject_external_references(child)


def _error_detail(body: bytes) -> str:
    text = body.decode("utf-8", errors="replace")
    try:
        decoded = _strict_json_loads(text)
    except (json.JSONDecodeError, RecursionError, ValueError):
        return text.strip()[:1000] or "empty error response"
    if isinstance(decoded, dict) and isinstance(decoded.get("error"), str):
        return decoded["error"][:1000]
    return text.strip()[:1000] or "empty error response"
