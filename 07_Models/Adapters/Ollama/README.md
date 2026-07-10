# AGI Ollama Adapter

`agi-ollama` is the local Ollama implementation of the workspace's generic
model-adapter boundary. It accepts and returns the shared models from
`agi-contracts`; provider-specific response shapes do not escape this package.

Safety properties:

- only literal loopback endpoints (`localhost`, `127.0.0.0/8`, or `::1`) are
  accepted;
- model selection is default-deny and requires an explicit allowlist;
- every request has a finite timeout, token ceiling, prompt-size ceiling, and
  response-size ceiling;
- retry counts and backoff are bounded;
- structured output is validated against the requested JSON Schema;
- a missing model is reported as an error—the adapter never pulls a model;
- there is no remote-provider fallback.

```python
from agi_contracts import MessageRole, ModelMessage, ModelRequest
from agi_ollama import OllamaAdapter, OllamaConfig

adapter = OllamaAdapter(
    OllamaConfig(allowed_models=("your-installed-model",))
)
response = await adapter.generate(
    ModelRequest(
        model="your-installed-model",
        messages=[
            ModelMessage(
                role=MessageRole.USER,
                content="Return a short answer.",
            )
        ],
        max_tokens=256,
    )
)
```

`ScriptedModelAdapter` supplies deterministic replies for integration tests and
never performs network I/O.
