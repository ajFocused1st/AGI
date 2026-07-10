from __future__ import annotations

import asyncio
import os
import unittest

from agi_contracts import ModelMessage, ModelRequest
from agi_ollama import OllamaAdapter, OllamaConfig


@unittest.skipUnless(
    os.environ.get("AGI_RUN_OLLAMA_SMOKE") == "1",
    "set AGI_RUN_OLLAMA_SMOKE=1 and OLLAMA_MODEL for an opt-in live test",
)
class LiveOllamaSmokeTest(unittest.TestCase):
    def test_local_structured_generation(self) -> None:
        model = os.environ.get("OLLAMA_MODEL", "").strip()
        if not model:
            self.skipTest("OLLAMA_MODEL is not configured")
        request = ModelRequest(
            model=model,
            messages=[ModelMessage(role="user", content="Return JSON with ok set to true.")],
            response_schema={
                "type": "object",
                "properties": {"ok": {"type": "boolean"}},
                "required": ["ok"],
                "additionalProperties": False,
            },
            temperature=0,
            max_tokens=32,
        )
        response = asyncio.run(
            OllamaAdapter(OllamaConfig(allowed_models=(model,))).generate(request)
        )
        self.assertEqual(response.structured_output, {"ok": True})


if __name__ == "__main__":
    unittest.main()
