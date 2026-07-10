"""Structured model outputs that the runtime validates before use."""

from __future__ import annotations

from typing import Any
from uuid import UUID

from agi_contracts.models import Description, Identifier, JsonObject, LongText, ShortText
from pydantic import BaseModel, ConfigDict, Field


class PlanStepDraft(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    step_id: str | int | None = Field(
        default=None,
        description="Ignored by the host. Present only to tolerate simple model numbering.",
    )
    description: Description
    tool_name: Identifier | None = None
    tool_arguments: JsonObject = Field(default_factory=dict)
    success_criteria: ShortText | None = None


class PlanDraft(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    plan_id: Any | None = Field(
        default=None,
        description="Ignored by the host. The runtime assigns durable identifiers.",
    )
    goal: LongText | None = None
    steps: list[PlanStepDraft] = Field(min_length=1, max_length=100)


class AnswerDraft(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    answer: str = Field(min_length=1, max_length=1_000_000)
    cited_evidence_ids: list[UUID] = Field(default_factory=list, max_length=512)
    insufficient_evidence: bool = False


class CritiqueDraft(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    critique: str = Field(min_length=1, max_length=100_000)
    suggestions: list[str] = Field(default_factory=list, max_length=32)
