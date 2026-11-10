from __future__ import annotations

import asyncio
import json
import sys
import unittest
from collections import deque
from pathlib import Path
from typing import Any, Mapping

PROJECT_ROOT = Path(__file__).resolve().parents[1]
WORKSPACE_ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(WORKSPACE_ROOT / "01_Core" / "Contracts" / "src"))

from agi_contracts import MessageRole, ModelMessage, ModelRequest  # noqa: E402
from agi_runpod import HttpResponse, RunPodAdapter, RunPodConfig  # noqa: E402
from agi_runpod.errors import RunPodHTTPError, RunPodRequestError, RunPodResponseValidationError  # noqa: E402


class FakeTransport:
    def __init__(self, responses: list[HttpResponse]) -> None:
        self.responses = deque(responses)
        self.calls: list[dict[str, Any]] = []

    async def post_json(
        self, *, url: str, payload: Mapping[str, Any], authorization: str,
        timeout_seconds: float, max_response_bytes: int,
    ) -> HttpResponse:
        self.calls.append({"url": url, "payload": payload, "authorization": authorization})
        return self.responses.popleft()


def config(**updates: Any) -> RunPodConfig:
    return RunPodConfig(api_key="temporary-secret", endpoint_id="endpoint123", model="fixture-model", **updates)


def request_for(*, structured: bool = True) -> ModelRequest:
    schema = {
        "type": "object", "properties": {"answer": {"type": "string"}},
        "required": ["answer"], "additionalProperties": False,
    } if structured else None
    return ModelRequest(
        model="fixture-model", messages=[ModelMessage(role=MessageRole.USER, content="Question")],
        response_schema=schema, max_tokens=128,
    )


def response(content: str, *, status: int = 200) -> HttpResponse:
    envelope = {
        "id": "response-id", "model": "fixture-model",
        "choices": [{"message": {"role": "assistant", "content": content}, "finish_reason": "stop"}],
        "usage": {"prompt_tokens": 10, "completion_tokens": 4, "total_tokens": 14},
    }
    return HttpResponse(status, json.dumps(envelope).encode(), {})


class RunPodAdapterTests(unittest.TestCase):
    def test_config_derives_exact_https_endpoint_and_redacts_key(self) -> None:
        configured = config()
        self.assertEqual(configured.base_url, "https://api.runpod.ai/v2/endpoint123/openai/v1")
        self.assertNotIn("temporary-secret", repr(configured))
        with self.assertRaises(ValueError):
            RunPodConfig(api_key="x", endpoint_id="https://api.runpod.ai/bad")

    def test_structured_response_and_safe_telemetry(self) -> None:
        transport = FakeTransport([response('{"answer":"grounded"}')])
        result = asyncio.run(RunPodAdapter(config(), transport=transport).generate(request_for()))
        self.assertEqual(result.structured_output, {"answer": "grounded"})
        self.assertEqual(result.raw_metadata["provider"], "runpod")
        self.assertEqual(result.usage.total_tokens, 14)
        self.assertEqual(transport.calls[0]["authorization"], "Bearer temporary-secret")
        self.assertNotIn("temporary-secret", result.model_dump_json())
        self.assertEqual(len(transport.calls), 1)

    def test_one_bounded_structured_repair(self) -> None:
        transport = FakeTransport([response("not json"), response('{"answer":"fixed"}')])
        result = asyncio.run(RunPodAdapter(config(), transport=transport).generate(request_for()))
        self.assertEqual(len(transport.calls), 2)
        self.assertTrue(result.raw_metadata["structured_output"]["repair_succeeded"])

    def test_invalid_output_stops_after_one_repair(self) -> None:
        transport = FakeTransport([response("bad"), response("still bad")])
        with self.assertRaises(RunPodResponseValidationError):
            asyncio.run(RunPodAdapter(config(), transport=transport).generate(request_for()))
        self.assertEqual(len(transport.calls), 2)

    def test_authentication_error_is_typed_without_body_echo(self) -> None:
        body = b'{"error":{"message":"invalid key temporary-secret"}}'
        transport = FakeTransport([HttpResponse(401, body, {})])
        with self.assertRaises(RunPodHTTPError) as caught:
            asyncio.run(RunPodAdapter(config(), transport=transport).generate(request_for(structured=False)))
        self.assertEqual(caught.exception.category, "authentication")
        self.assertNotIn("temporary-secret", str(caught.exception))

    def test_external_schema_reference_is_rejected_before_transport(self) -> None:
        transport = FakeTransport([])
        model_request = request_for().model_copy(
            update={"response_schema": {"$ref": "https://example.com/schema.json"}}
        )
        with self.assertRaises(RunPodRequestError):
            asyncio.run(RunPodAdapter(config(), transport=transport).generate(model_request))
        self.assertEqual(transport.calls, [])


if __name__ == "__main__":
    unittest.main()
