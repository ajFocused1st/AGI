"""One-call FAST path for cheap direct answers."""

from __future__ import annotations

from dataclasses import dataclass

from agi_contracts import AgentProfile, AgentRequest, ModelMessage, ModelRequest, ModelResponse

from .budgets import BudgetGovernor
from .ports import ModelAdapter
from .routing import ReasoningTier, get_routing_policy


@dataclass(frozen=True)
class FastPathResult:
    response: ModelResponse
    ledger: dict[str, object]


class FastPathRunner:
    """Runs exactly one direct model call under the FAST policy."""

    def __init__(self, *, model_adapter: ModelAdapter) -> None:
        self._model = model_adapter
        self._policy = get_routing_policy(ReasoningTier.FAST)

    async def run(self, request: AgentRequest, profile: AgentProfile) -> FastPathResult:
        governor = BudgetGovernor(self._policy)
        governor.reserve_model_call("fast_answer")
        model_request = ModelRequest(
            model=profile.model,
            messages=[
                ModelMessage(
                    role="system",
                    content=(
                        "Answer directly and briefly. Do not use tools, retrieval, hidden "
                        "critique, or multi-step repair. If the answer is uncertain, say so."
                    ),
                ),
                ModelMessage(role="user", content=request.goal),
            ],
            temperature=0,
            max_tokens=self._policy.budget.max_output_tokens,
            metadata={
                "phase": "fast_answer",
                "routing_tier": self._policy.tier.value,
                "native_thinking": self._policy.native_thinking,
            },
        )
        response = await self._model.generate(model_request)
        governor.record_model_usage("fast_answer", response.usage)
        return FastPathResult(response=response, ledger=governor.ledger.as_dict())
