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
from agi_ollama import (  # noqa: E402
    HttpResponse,
    OllamaAdapter,
    OllamaConfig,
    OllamaModelNotFoundError,
    OllamaRequestError,
    OllamaResponseValidationError,
    OllamaTimeoutError,
    ScriptedModelAdapter,
    ScriptedReply,
)


class FakeTransport:
    def __init__(self, responses: list[HttpResponse | Exception]) -> None:
        self.responses = deque(responses)
        self.calls: list[dict[str, Any]] = []

    async def post_json(
        self,
        *,
        url: str,
        payload: Mapping[str, Any],
        timeout_seconds: float,
        max_response_bytes: int,
    ) -> HttpResponse:
        self.calls.append(
            {
                "url": url,
                "payload": payload,
                "timeout_seconds": timeout_seconds,
                "max_response_bytes": max_response_bytes,
            }
        )
        result = self.responses.popleft()
        if isinstance(result, Exception):
            raise result
        return result


def request_for(*, structured: bool = False, max_tokens: int = 128) -> ModelRequest:
    schema = None
    if structured:
        schema = {
            "type": "object",
            "properties": {"answer": {"type": "string"}},
            "required": ["answer"],
            "additionalProperties": False,
        }
    return ModelRequest(
        model="fixture-model",
        messages=[
            ModelMessage(role=MessageRole.USER, content="What is in the fixture?")
        ],
        response_schema=schema,
        temperature=0,
        max_tokens=max_tokens,
        seed=7,
    )


def config(**overrides: Any) -> OllamaConfig:
    return OllamaConfig(allowed_models=("fixture-model",), **overrides)


def response(content: str, *, status: int = 200) -> HttpResponse:
    envelope = {
        "model": "fixture-model",
        "created_at": "2026-07-10T12:00:00Z",
        "message": {"role": "assistant", "content": content},
        "done": True,
        "done_reason": "stop",
        "prompt_eval_count": 11,
        "eval_count": 5,
    }
    return HttpResponse(status, json.dumps(envelope).encode(), {})


class OllamaConfigTests(unittest.TestCase):
    def test_remote_hosts_and_url_paths_are_rejected(self) -> None:
        bad_urls = (
            "http://example.com:11434",
            "https://127.0.0.1:11434",
            "http://localhost.evil:11434",
            "http://127.0.0.1:11434/api",
        )
        for value in bad_urls:
            with self.subTest(value=value), self.assertRaises(ValueError):
                OllamaConfig(base_url=value)

    def test_ipv4_and_ipv6_loopback_are_accepted(self) -> None:
        self.assertEqual(
            OllamaConfig(base_url="http://127.2.3.4:11434/").base_url,
            "http://127.2.3.4:11434",
        )
        self.assertEqual(
            OllamaConfig(base_url="http://[::1]:11434").base_url,
            "http://[::1]:11434",
        )


class OllamaAdapterTests(unittest.IsolatedAsyncioTestCase):
    async def test_generates_and_validates_structured_output(self) -> None:
        transport = FakeTransport([response('{"answer":"grounded"}')])
        adapter = OllamaAdapter(config(), transport=transport)
        model_request = request_for(structured=True)

        result = await adapter.generate(model_request)

        self.assertEqual(result.request_id, model_request.request_id)
        self.assertEqual(result.structured_output, {"answer": "grounded"})
        self.assertEqual(result.usage.total_tokens, 16)
        self.assertEqual(transport.calls[0]["url"], "http://127.0.0.1:11434/api/chat")
        payload = transport.calls[0]["payload"]
        self.assertFalse(payload["stream"])
        self.assertEqual(payload["format"], model_request.response_schema)
        self.assertEqual(payload["options"]["num_predict"], 128)
        self.assertEqual(payload["options"]["seed"], 7)

    async def test_schema_violation_is_rejected(self) -> None:
        transport = FakeTransport([response('{"wrong":"field"}')])
        adapter = OllamaAdapter(config(), transport=transport)
        with self.assertRaises(OllamaResponseValidationError):
            await adapter.generate(request_for(structured=True))

    async def test_external_schema_reference_is_rejected_before_transport(self) -> None:
        transport = FakeTransport([])
        adapter = OllamaAdapter(config(), transport=transport)
        model_request = request_for()
        model_request = model_request.model_copy(
            update={"response_schema": {"$ref": "https://example.com/schema.json"}}
        )
        with self.assertRaisesRegex(OllamaRequestError, "External"):
            await adapter.generate(model_request)
        self.assertEqual(transport.calls, [])

    async def test_duplicate_json_keys_are_rejected(self) -> None:
        transport = FakeTransport(
            [response('{"answer":"first","answer":"second"}')]
        )
        adapter = OllamaAdapter(config(), transport=transport)
        with self.assertRaisesRegex(OllamaResponseValidationError, "strict JSON"):
            await adapter.generate(request_for(structured=True))

    async def test_transient_failure_retries_only_to_bound(self) -> None:
        unavailable = HttpResponse(503, b'{"error":"busy"}', {})
        transport = FakeTransport([unavailable, unavailable])
        delays: list[float] = []

        async def record_sleep(delay: float) -> None:
            delays.append(delay)

        adapter = OllamaAdapter(
            config(max_retries=1, retry_backoff_seconds=0.2),
            transport=transport,
            sleeper=record_sleep,
        )
        with self.assertRaisesRegex(Exception, "HTTP 503"):
            await adapter.generate(request_for())
        self.assertEqual(len(transport.calls), 2)
        self.assertEqual(delays, [0.2])

    async def test_missing_model_is_not_retried_or_pulled(self) -> None:
        missing = HttpResponse(404, b'{"error":"model not found"}', {})
        transport = FakeTransport([missing])
        adapter = OllamaAdapter(config(max_retries=3), transport=transport)
        with self.assertRaises(OllamaModelNotFoundError) as raised:
            await adapter.generate(request_for())
        self.assertIn("not pulled automatically", str(raised.exception))
        self.assertEqual(len(transport.calls), 1)

    async def test_custom_transport_is_still_subject_to_deadline(self) -> None:
        class HangingTransport:
            async def post_json(self, **_: Any) -> HttpResponse:
                await asyncio.sleep(1)
                return response("late")

        adapter = OllamaAdapter(
            config(timeout_seconds=0.01, max_retries=0),
            transport=HangingTransport(),
        )
        with self.assertRaises(OllamaTimeoutError):
            await adapter.generate(request_for())

    async def test_token_ceiling_is_checked_before_transport(self) -> None:
        transport = FakeTransport([])
        adapter = OllamaAdapter(
            config(max_tokens_per_request=64),
            transport=transport,
        )
        with self.assertRaises(OllamaRequestError):
            await adapter.generate(request_for(max_tokens=65))
        self.assertEqual(transport.calls, [])

    async def test_model_allowlist_is_default_deny(self) -> None:
        transport = FakeTransport([])
        adapter = OllamaAdapter(transport=transport)
        with self.assertRaisesRegex(OllamaRequestError, "not allowed"):
            await adapter.generate(request_for())
        self.assertEqual(transport.calls, [])


class ScriptedAdapterTests(unittest.IsolatedAsyncioTestCase):
    async def test_scripted_adapter_is_fifo_and_network_free(self) -> None:
        adapter = ScriptedModelAdapter(
            [ScriptedReply(content='{"answer":"offline"}')]
        )
        model_request = request_for(structured=True)

        result = await adapter.generate(model_request)

        self.assertEqual(result.request_id, model_request.request_id)
        self.assertEqual(result.structured_output, {"answer": "offline"})
        self.assertEqual(adapter.remaining_replies, 0)
        self.assertEqual(adapter.requests[0].request_id, model_request.request_id)
        with self.assertRaisesRegex(RuntimeError, "exhausted"):
            await adapter.generate(model_request)


if __name__ == "__main__":
    unittest.main()
