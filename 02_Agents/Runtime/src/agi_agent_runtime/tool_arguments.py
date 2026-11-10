"""Pre-dispatch validation for model-proposed tool arguments."""

from __future__ import annotations

import hashlib
import json
from pathlib import PurePath
from typing import Any

from agi_contracts import TaskRequirements, ToolArgumentValidationResult, ToolDefinition
from agi_contracts.models import JsonObject


def validate_tool_arguments(
    definition: ToolDefinition,
    arguments: JsonObject,
    requirements: TaskRequirements,
    *,
    repair_attempted: bool = False,
) -> ToolArgumentValidationResult:
    schema = definition.input_schema
    properties = schema.get("properties", {})
    required = set(schema.get("required", []))
    supplied = set(arguments)
    unknown = sorted(supplied - set(properties))
    missing = sorted(required - supplied)
    schema_errors: list[str] = []
    semantic_errors: list[str] = []
    for name in missing:
        schema_errors.append(f"required argument is missing: {name}")
    for name in unknown:
        schema_errors.append(f"unsupported argument: {name}")
    for name in supplied & set(properties):
        _validate_value(name, arguments[name], properties[name], schema_errors)
    if definition.name == "read_only_file_search":
        _validate_search_semantics(arguments, requirements, semantic_errors)
    if definition.name == "calculator":
        expression = arguments.get("expression")
        allowed = frozenset("0123456789.+-*/%() \t")
        if not isinstance(expression, str) or not expression.strip() or any(
            character not in allowed for character in expression
        ):
            semantic_errors.append("calculator expression must contain arithmetic only")
    messages = schema_errors + semantic_errors
    valid = not messages
    return ToolArgumentValidationResult(
        tool_name=definition.name,
        valid=valid,
        schema_errors=schema_errors,
        semantic_errors=semantic_errors,
        unknown_arguments=unknown,
        missing_arguments=missing,
        repair_attempted=repair_attempted,
        repair_succeeded=repair_attempted and valid,
        final_valid=valid,
        dispatch_allowed=valid,
        validation_messages=messages,
    )


def redact_arguments(arguments: JsonObject) -> JsonObject:
    """Persist shape and stable fingerprints, never unrestricted values."""

    redacted: JsonObject = {}
    for name, value in arguments.items():
        encoded = json.dumps(value, sort_keys=True, ensure_ascii=False).encode("utf-8")
        redacted[name] = {
            "type": type(value).__name__,
            "length": len(value) if isinstance(value, (str, list, dict)) else None,
            "sha256": hashlib.sha256(encoded).hexdigest(),
        }
    return redacted


def _validate_value(
    name: str, value: Any, schema: dict[str, Any], errors: list[str]
) -> None:
    expected = schema.get("type")
    valid_type = {
        "string": isinstance(value, str),
        "integer": isinstance(value, int) and not isinstance(value, bool),
        "number": isinstance(value, (int, float)) and not isinstance(value, bool),
        "boolean": isinstance(value, bool),
        "object": isinstance(value, dict),
        "array": isinstance(value, list),
    }.get(expected, True)
    if not valid_type:
        errors.append(f"argument {name} must be {expected}")
        return
    if isinstance(value, str):
        if len(value) < int(schema.get("minLength", 0)):
            errors.append(f"argument {name} is too short")
        if len(value) > int(schema.get("maxLength", len(value))):
            errors.append(f"argument {name} is too long")
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        if "minimum" in schema and value < schema["minimum"]:
            errors.append(f"argument {name} is below minimum")
        if "maximum" in schema and value > schema["maximum"]:
            errors.append(f"argument {name} exceeds maximum")


def _validate_search_semantics(
    arguments: JsonObject,
    requirements: TaskRequirements,
    errors: list[str],
) -> None:
    query = arguments.get("query")
    if isinstance(query, str) and not query.strip():
        errors.append("search query must not be empty or whitespace")
        return
    if not isinstance(query, str):
        return
    normalized = " ".join(query.casefold().split())
    path = PurePath(query)
    if ".." in path.parts or query.startswith(("/", "\\")) or ":\\" in query:
        errors.append("search query must not be a filesystem path")
    sensitive = (".env", "credentials", "password", "private key", "secrets")
    if any(term in normalized for term in sensitive):
        errors.append("search query targets sensitive-file material")
    vague = {
        "summarize the documents",
        "search the documents",
        "find relevant information",
        "local benchmark documents",
    }
    if normalized in vague:
        errors.append("search query is a vague task instruction")
    if requirements.task_subjects and not any(
        subject.casefold() in normalized for subject in requirements.task_subjects
    ):
        errors.append("search query does not identify a required task subject")
