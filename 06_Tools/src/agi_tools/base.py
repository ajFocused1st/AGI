"""Common execution boundary for tools."""

from __future__ import annotations

import asyncio
import json
from abc import ABC, abstractmethod
from datetime import UTC, datetime
from time import perf_counter
from typing import ClassVar, Generic, TypeVar

from agi_contracts import RiskLevel, ToolDefinition, ToolRequest, ToolResult
from pydantic import BaseModel, ValidationError


InputT = TypeVar("InputT", bound=BaseModel)
OutputT = TypeVar("OutputT", bound=BaseModel)


class ToolInputError(ValueError):
    """A safe, user-correctable tool input error."""


def _validation_error(exc: ValidationError) -> str:
    """Return validation details without echoing potentially sensitive input."""

    details = [
        {
            "location": ".".join(str(part) for part in error["loc"]),
            "message": error["msg"],
            "type": error["type"],
        }
        for error in exc.errors(include_input=False, include_url=False)
    ]
    return f"invalid tool arguments: {json.dumps(details, separators=(',', ':'))}"


class BaseTool(ABC, Generic[InputT, OutputT]):
    """Validates a request and returns the shared ``ToolResult`` contract.

    Permission checks deliberately do not live here. Callers execute tools through
    :class:`agi_tools.ToolRegistry`, which records a decision before invoking this
    internal execution method.
    """

    name: ClassVar[str]
    description: ClassVar[str]
    risk_level: ClassVar[RiskLevel] = RiskLevel.LOW
    permissions_required: ClassVar[tuple[str, ...]] = ()
    human_approval_required: ClassVar[bool] = False
    timeout_seconds: ClassVar[float] = 5.0
    input_model: ClassVar[type[BaseModel]]
    output_model: ClassVar[type[BaseModel]]

    @property
    def definition(self) -> ToolDefinition:
        """Return the machine-readable definition exposed to planners."""

        return ToolDefinition(
            name=self.name,
            description=self.description,
            input_schema=self.input_model.model_json_schema(),
            output_schema=self.output_model.model_json_schema(),
            risk_level=self.risk_level,
            permissions_required=list(self.permissions_required),
            human_approval_required=self.human_approval_required,
            timeout_seconds=self.timeout_seconds,
        )

    async def _execute_request(self, request: ToolRequest) -> ToolResult:
        """Execute a request after the registry has granted permission."""

        started_at = datetime.now(UTC)
        started_clock = perf_counter()
        success = False
        output: dict[str, object] = {}
        error: str | None = None

        try:
            if request.tool_name != self.name:
                raise ToolInputError("request tool name does not match the selected tool")
            tool_input = self.input_model.model_validate(request.arguments)
            raw_output = await asyncio.wait_for(
                self.run(tool_input),
                timeout=self.timeout_seconds,
            )
            tool_output = self.output_model.model_validate(raw_output)
            output = tool_output.model_dump(mode="json")
            success = True
        except ValidationError as exc:
            error = _validation_error(exc)
        except ToolInputError as exc:
            error = str(exc)
        except TimeoutError:
            error = "tool execution timed out"
        except Exception:
            # Tool failures may cross trust boundaries. Do not disclose local paths,
            # implementation details, or input data through an exception string.
            error = "tool execution failed"

        completed_at = datetime.now(UTC)
        duration_ms = max(0.0, (perf_counter() - started_clock) * 1000.0)
        return ToolResult(
            request_id=request.request_id,
            tool_name=self.name,
            success=success,
            output=output,
            error=error,
            started_at=started_at,
            completed_at=completed_at,
            duration_ms=duration_ms,
        )

    def preflight(self, tool_input: InputT) -> list[str]:
        """Side-effect-free checks run before dispatch; return reasons the call would fail."""

        return []

    @abstractmethod
    async def run(self, tool_input: InputT) -> OutputT:
        """Perform the tool-specific operation on validated input."""
