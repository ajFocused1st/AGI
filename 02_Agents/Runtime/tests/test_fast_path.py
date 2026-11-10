from __future__ import annotations

import unittest

from agi_agent_runtime import FastPathRunner
from agi_contracts import AgentProfile, AgentRequest, ModelRequest, ModelResponse, ModelUsage


class OneReplyAdapter:
    def __init__(self) -> None:
        self.requests: list[ModelRequest] = []

    async def generate(self, request: ModelRequest) -> ModelResponse:
        self.requests.append(request)
        return ModelResponse(
            request_id=request.request_id,
            model=request.model,
            content="Direct answer.",
            usage=ModelUsage(input_tokens=12, output_tokens=3, total_tokens=15),
        )


class FastPathTests(unittest.IsolatedAsyncioTestCase):
    async def test_fast_path_makes_exactly_one_model_call(self) -> None:
        adapter = OneReplyAdapter()
        runner = FastPathRunner(model_adapter=adapter)

        result = await runner.run(
            AgentRequest(goal="Answer quickly."),
            AgentProfile(id="researcher", description="Test", model="scripted"),
        )

        self.assertEqual(result.response.content, "Direct answer.")
        self.assertEqual(len(adapter.requests), 1)
        self.assertEqual(adapter.requests[0].metadata["routing_tier"], "fast")
        self.assertEqual(adapter.requests[0].max_tokens, 512)
        self.assertEqual(result.ledger["model_attempts"], 1)
        self.assertEqual(result.ledger["tool_attempts"], 0)


if __name__ == "__main__":
    unittest.main()
