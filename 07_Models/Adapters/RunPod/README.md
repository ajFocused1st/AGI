# AGI RunPod adapter

An authenticated, bounded adapter for RunPod Serverless vLLM's
OpenAI-compatible chat-completions endpoint. The API key is accepted only at
runtime and is never included in requests, responses, telemetry, or errors.

The adapter is provider-specific. The agent runtime continues to depend only
on the shared `ModelAdapter` behavior and receives this implementation through
the project composition root.
