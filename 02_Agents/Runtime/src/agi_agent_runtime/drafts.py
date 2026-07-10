"""Structured model outputs that the runtime validates before use."""

from __future__ import annotations

from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field


class AnswerDraft(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    answer: str = Field(min_length=1, max_length=1_000_000)
    cited_evidence_ids: list[UUID] = Field(default_factory=list, max_length=512)
    insufficient_evidence: bool = False


class CritiqueDraft(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    critique: str = Field(min_length=1, max_length=100_000)
    suggestions: list[str] = Field(default_factory=list, max_length=32)
