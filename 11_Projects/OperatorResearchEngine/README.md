# Operator Research Engine

This project is the first governed vertical slice of the AGI workspace. It accepts a research goal, creates a structured plan, invokes only approved low-risk tools, evaluates the grounded response, and atomically stores a reproducible run record under `C:\AGI_Runtime`.

## Boundaries

The project composes shared distributions; it does not copy them:

- `agi_contracts` — shared Pydantic contracts
- `agi_agent_runtime` — injected orchestration state machine and run store
- `agi_tools` — calculator, read-only file search, registry, and permission checks
- `agi_ollama` — provider-neutral interface and loopback-only Ollama adapter
- `agi_evaluation` — authoritative rule checks and separate advisory critique

The first release cannot execute terminals, mutate files, send messages, browse the web, pull models, or fall back to a remote provider.

## Development

Run `scripts\Setup-Development.ps1` once to create an ignored local environment and register the shared source-layout packages through a local `.pth` file. The offline setup performs no package or model downloads and does not modify global Python. Set `OLLAMA_MODEL` to the exact name of an already-installed Ollama model before using the live CLI.

Use `scripts\Test.ps1` for deterministic offline tests and `scripts\Run-Benchmark.ps1` for the ten-case benchmark. Generated records and results are written outside Git to `C:\AGI_Runtime`.

## Milestone

The milestone is complete only when one agent can plan, use an approved tool, return grounded citations, receive a rule-based evaluation, and save a schema-valid run record. The destructive-action, calculator-trace, citation, and record-validity benchmark cases are mandatory safety gates.

## RunPod provider

Provider choice is explicit through `MODEL_PROVIDER`; there is no silent local
or remote fallback. RunPod support lives in the separate `agi_runpod` adapter.

Set `RUNPOD_API_KEY` and `RUNPOD_ENDPOINT_ID` (the ID only, not a URL) in the
current PowerShell session, then use `scripts\Run-RunPod.ps1`. Keep the key out
of `.env`, YAML, scripts, command history, logs, reports, and Git. The optional
`RUNPOD_BASE_URL` is restricted to the matching HTTPS endpoint at
`api.runpod.ai`.
