"""Host-owned routing policies for adaptive reasoning runs."""

from __future__ import annotations

from enum import StrEnum
from types import MappingProxyType

from pydantic import BaseModel, ConfigDict, Field, field_validator


class ReasoningTier(StrEnum):
    FAST = "fast"
    STANDARD = "standard"
    DEEP = "deep"
    MAX = "max"


class BudgetPolicy(BaseModel):
    """Hard limits for one routed attempt."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    max_model_calls: int = Field(ge=1, le=64)
    max_tool_calls: int = Field(ge=0, le=128)
    max_input_tokens: int | None = Field(default=None, ge=1)
    max_output_tokens: int = Field(ge=1, le=131_072)
    max_total_tokens: int | None = Field(default=None, ge=1)
    max_wallclock_seconds: float = Field(gt=0.0, le=7200.0)

    @field_validator("max_total_tokens")
    @classmethod
    def total_must_cover_output(cls, value: int | None, info: object) -> int | None:
        if value is None:
            return value
        data = getattr(info, "data", {})
        output_limit = data.get("max_output_tokens")
        if output_limit is not None and value < output_limit:
            raise ValueError("max_total_tokens must be >= max_output_tokens")
        return value


class RoutingPolicy(BaseModel):
    """Explicit contract for a reasoning tier."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    tier: ReasoningTier
    budget: BudgetPolicy
    allow_tools: bool
    allow_retrieval: bool
    allow_critique: bool
    allow_revision: bool
    native_thinking: bool
    description: str = Field(min_length=1, max_length=512)


DEFAULT_ROUTING_POLICIES = MappingProxyType(
    {
        ReasoningTier.FAST: RoutingPolicy(
            tier=ReasoningTier.FAST,
            budget=BudgetPolicy(
                max_model_calls=1,
                max_tool_calls=0,
                max_input_tokens=4096,
                max_output_tokens=512,
                max_total_tokens=4608,
                max_wallclock_seconds=20.0,
            ),
            allow_tools=False,
            allow_retrieval=False,
            allow_critique=False,
            allow_revision=False,
            native_thinking=False,
            description="One direct model call for simple answers and cheap triage.",
        ),
        ReasoningTier.STANDARD: RoutingPolicy(
            tier=ReasoningTier.STANDARD,
            budget=BudgetPolicy(
                max_model_calls=5,
                max_tool_calls=8,
                max_input_tokens=8192,
                max_output_tokens=2048,
                max_total_tokens=12_288,
                max_wallclock_seconds=120.0,
            ),
            allow_tools=True,
            allow_retrieval=True,
            allow_critique=False,
            allow_revision=True,
            native_thinking=False,
            description="Bounded tool use and one repair/revision path.",
        ),
        ReasoningTier.DEEP: RoutingPolicy(
            tier=ReasoningTier.DEEP,
            budget=BudgetPolicy(
                max_model_calls=6,
                max_tool_calls=6,
                max_input_tokens=16_384,
                max_output_tokens=4096,
                max_total_tokens=24_576,
                max_wallclock_seconds=300.0,
            ),
            allow_tools=True,
            allow_retrieval=True,
            allow_critique=True,
            allow_revision=True,
            native_thinking=True,
            description="Sequential critique and revision for difficult tasks.",
        ),
        ReasoningTier.MAX: RoutingPolicy(
            tier=ReasoningTier.MAX,
            budget=BudgetPolicy(
                max_model_calls=10,
                max_tool_calls=12,
                max_input_tokens=32_768,
                max_output_tokens=8192,
                max_total_tokens=49_152,
                max_wallclock_seconds=900.0,
            ),
            allow_tools=True,
            allow_retrieval=True,
            allow_critique=True,
            allow_revision=True,
            native_thinking=True,
            description="Highest local budget for rare, explicitly expensive tasks.",
        ),
    }
)


def get_routing_policy(tier: ReasoningTier | str) -> RoutingPolicy:
    return DEFAULT_ROUTING_POLICIES[ReasoningTier(tier)]
