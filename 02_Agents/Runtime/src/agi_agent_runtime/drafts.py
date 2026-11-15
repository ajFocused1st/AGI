"""Structured model outputs that the runtime validates before use."""

from __future__ import annotations

from typing import Any
from uuid import UUID

from agi_contracts.models import Description, Identifier, JsonObject, LongText, ShortText
from pydantic import BaseModel, ConfigDict, Field, model_validator


class PlanStepDraft(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    step_id: str | int | None = Field(
        default=None,
        description="Ignored by the host. Present only to tolerate simple model numbering.",
    )
    description: Description
    success_criteria: ShortText | None = None


_TOOL_NAME_ALIASES = ("name", "tool")
_NO_TOOL_NAME = frozenset({"", "null", "none"})


class ToolCallDraft(BaseModel):
    """A proposed tool call, read tolerantly.

    Small models misspell the envelope, not the intent: they write ``name`` or ``tool``
    for ``tool_name``, the text "null" for no tool, and arguments beside ``arguments``
    instead of inside it. Rejecting or silently dropping those keys loses the call, so
    they are accepted here: aliases are mapped, and unknown keys are kept (``extra``)
    for the host to match against the named tool's argument schema. Nothing is invented,
    and every recovered call still passes the same permission and argument validation.
    """

    model_config = ConfigDict(extra="allow", str_strip_whitespace=True)

    tool_name: Identifier | None = None
    arguments: JsonObject = Field(default_factory=dict)
    purpose: ShortText | None = None
    related_step_id: str | int | None = None

    @model_validator(mode="before")
    @classmethod
    def _accept_common_spellings(cls, value: Any) -> Any:
        if not isinstance(value, dict):
            return value
        data = dict(value)
        for alias in _TOOL_NAME_ALIASES:
            spelled = data.pop(alias, None)
            if isinstance(spelled, str) and not isinstance(data.get("tool_name"), str):
                data["tool_name"] = spelled
        name = data.get("tool_name")
        if isinstance(name, str) and name.strip().casefold() in _NO_TOOL_NAME:
            data["tool_name"] = None
        if data.get("arguments") is None:
            data["arguments"] = {}
        return data

    def is_empty(self) -> bool:
        """True when the entry proposes nothing (a model's way of writing "no call")."""

        return not self.tool_name and not self.arguments and not self.model_extra


class PlanDraft(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    plan_id: Any | None = Field(
        default=None,
        description="Ignored by the host. The runtime assigns durable identifiers.",
    )
    goal: LongText | None = None
    steps: list[PlanStepDraft] = Field(min_length=1, max_length=100)
    tool_calls: list[ToolCallDraft] = Field(max_length=100)


class FollowUpDraft(BaseModel):
    """Additional tool calls proposed after observing results; empty means sufficient."""

    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    tool_calls: list[ToolCallDraft] = Field(default_factory=list, max_length=4)


class GroundedClaimDraft(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    text: str = Field(min_length=1, max_length=16_384)
    evidence_ids: list[UUID] = Field(default_factory=list, max_length=128)


class AnswerDraft(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    answer: str | None = Field(default=None, min_length=1, max_length=1_000_000)
    cited_evidence_ids: list[UUID] = Field(default_factory=list, max_length=512)
    insufficient_evidence: bool = False
    claims: list[GroundedClaimDraft] = Field(default_factory=list, max_length=256)
    uncertainty: str | None = Field(default=None, min_length=1, max_length=16_384)
    summary: str | None = Field(default=None, min_length=1, max_length=65_536)


class CritiqueDraft(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    critique: str = Field(min_length=1, max_length=100_000)
    suggestions: list[str] = Field(default_factory=list, max_length=32)
