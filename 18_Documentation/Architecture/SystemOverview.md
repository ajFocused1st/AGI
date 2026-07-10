# System Overview

Describes the system context, stable boundaries, and major data flows.

## First working flow

```text
AgentRequest
  -> shared AgentRuntime
  -> structured TaskPlan
  -> deny-by-default ToolRegistry
  -> Calculator or ReadOnlyFileSearch
  -> grounded AgentResponse
  -> authoritative RuleBasedEvaluator
  -> atomic RunRecord in C:\AGI_Runtime\logs
```

`OperatorResearchEngine` is the composition root. It loads the researcher profile, selects one explicitly allowed Ollama model, configures two low-risk tools, and injects those dependencies into the shared runtime. The runtime depends on structural interfaces rather than an Ollama implementation, so another adapter can be substituted without changing orchestration.

## Dependency direction

Contracts are the dependency leaf. Tools, model adapters, evaluation, and runtime consume contracts. The project consumes the shared packages and supplies configuration. Shared packages never import project code, and numbered ownership folders are not Python import names; each implementation uses a valid `src`-layout distribution.

## Authority and trust boundaries

- Model output proposes plans and tool requests; only runtime code changes state.
- The registry checks the profile tool allowlist, exact permissions, risk policy, and approval requirements before dispatch.
- Unknown tools and missing permissions fail closed and produce recorded decisions.
- Calculator expressions are parsed through a restricted AST and are never evaluated as Python code.
- File-search roots come from trusted host configuration, never model arguments. Search rejects traversal fields, hidden or sensitive files, binary and oversized content, and symlink/reparse escapes.
- Local documents and tool outputs are untrusted evidence. Prompts explicitly prohibit following instructions embedded in them.
- Rule-based checks determine pass, partial, or fail. Model critique is attributed and advisory.

## Persistence and reproducibility

Run records include the goal, profile snapshot and hash, model identifier and parameters, plan, ordered permission decisions, tool requests and outputs, response, evaluation, usage estimates, timestamps, duration, errors, and configuration/input hashes. Records are redacted and written atomically to `C:\AGI_Runtime\logs\<run_id>\run.json`.

Versioned benchmark inputs remain in Git. Generated benchmark results and experiment manifests are stored under `C:\AGI_Runtime`. The deterministic benchmark records its dataset hash and current Git commit.

## Deliberate v1 limits

There is one agent profile, one model-adapter implementation, two low-risk tools, and no general memory system. Terminal execution, deletion, email, browser automation, database mutation, automatic model downloads, and remote fallback are unavailable. Live Ollama testing is optional because the adapter must not start services or pull models automatically.
