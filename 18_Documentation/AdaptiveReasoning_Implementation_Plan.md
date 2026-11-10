# Adaptive reasoning implementation stages

The application will extend the existing C:\AGI shared packages. Model selection remains local Qwen3 8B through Ollama. All model roles execute sequentially. Runtime artifacts belong under C:\AGI_Runtime.

| Stage | Work | Completion evidence |
|---|---|---|
| 0. Baseline | Forty direct-Qwen diagnostic cases, independent graders, configuration and model hashes, latency and token logs | Tests pass; live baseline collected; failures and limitations reported |
| 1. Routing and budgets | Host precheck, optional model governor, explicit FAST/STANDARD/DEEP/MAX policy contracts, real one-call FAST path, decision ledger | No hidden routing calls; all attempts count; time/token/call budgets enforced |
| 2. Tools and verification | Reuse calculator and read-only retrieval; add task-specific evidence checks | Correctness checks distinguish execution success from answer support |
| 3. Adaptive reasoning | Structured assumptions, selective critic/revision, escalation, stop decisions, sequential candidates | Supported answers stop early; missing premises and exhausted budgets terminate honestly |
| 4. Memory and skills | SQLite provenance, active/superseded records, verification timestamps, selected procedural skills | Retrieval preserves history and uncertainty; only relevant skills enter context |
| 5. Comparison and tuning | Plain Qwen vs current runtime vs adaptive system; repeated and held-out evaluations | Accuracy, reliability, tokens, calls, latency and resource costs reported together |

Evaluation continues throughout development. Later stages do not begin automatically merely because their files or interface names exist.

## Stage 0 implementation

- Versioned cases: `10_Benchmarks/Definitions/qwen_baseline_v1.json`.
- Runner and graders: `10_Benchmarks/Runners/QwenBaseline`.
- Shared integration: optional context/thinking configuration in the existing Ollama adapter. Existing consumers retain their defaults.
- Live outputs: timestamped directories under `C:\AGI_Runtime\benchmark-results\qwen-baseline-v1`.
- Initial condition: one direct request per case, native thinking enabled, 4096 context, 2048 generated-token ceiling, temperature 0.6, seed 42, no tools, retries, critic, retrieval, or schema-constrained decoding.
- Those settings are an explicit bounded baseline, not a claim of optimal Qwen performance. Non-thinking and larger-budget comparisons must be separately labeled; never replace earlier results silently.

Forty synthetic cases cover arithmetic, logic, ambiguity, instructions, coding diagnosis/tracing, constrained planning, contradictions, constrained business reasoning, hallucination resistance, and tool selection. Every grader is deterministic. Grading rules stay outside model prompts. Strict requested answer format is part of correctness. Failures do not automatically establish a reasoning deficit.

This first suite does not measure generated-code execution, long-document retrieval, open-ended professional advice, or general intelligence. Fresh held-out tasks and broader rubrics must be added before general improvement claims. No generated model code is executed during Phase 0.

The tool registry, evaluator, existing benchmark baselines, RunPod work, and other in-progress files retain their existing roles. The direct-Qwen runner deliberately bypasses AgentRuntime so it measures the base condition.

## Stage 1 implementation

Initial routing and budget contracts now live in the shared runtime package:

- `02_Agents/Runtime/src/agi_agent_runtime/routing.py` defines FAST, STANDARD, DEEP, and MAX policies.
- `02_Agents/Runtime/src/agi_agent_runtime/budgets.py` counts model/tool attempts, token usage, and wall-clock limits.
- `02_Agents/Runtime/src/agi_agent_runtime/fast_path.py` provides a one-call FAST runner.

The FAST policy is deliberately strict: one model call, no tools, no retrieval, no critique, no revision, native thinking disabled, and a 512 generated-token ceiling. This preserves a cheap path for simple answers and prevents hidden routing calls from consuming budget.

`AgentRuntime.run(...)` now enforces the selected policy across planning, plan correction, tool argument repair, answer generation, answer correction, critique, and actual tool dispatch. The persisted run metadata includes the budget ledger, and budget exhaustion creates a failed run with a `budget_exceeded` error before surfacing a `RuntimeExecutionError`.

Host precheck now validates routing/profile/tool compatibility before model calls begin. OperatorResearchEngine exposes public routing selection through the validated `reasoning_tier` setting, `AGI_REASONING_TIER`, and the CLI `--tier fast|standard|deep|max` argument.

Next Stage 2 work: add the first verification path by routing concrete arithmetic through the calculator and preserving answer support evidence.

The first Stage 2 slice is implemented: successful calculator outputs now enter the evidence bundle as `tool_result` evidence with deterministic hashes. Answer drafts can cite those evidence IDs, and response requirements validate citation/grounding over calculator evidence.

Arithmetic verification now has a deterministic `calculator_evidence_consistency` rule. If an answer cites calculator evidence, the cited result value must appear in the answer; citing `37 * 14 = 518` while answering `519` fails the final evaluation.

Document verification now has a deterministic `document_evidence_support` rule. If an answer cites document evidence, the answer must share enough concrete terms with cited excerpts and must not introduce title-cased entities absent from those excerpts. This catches failures such as citing evidence that says `Mina Chen owns the Atlas review` while answering `Nora Patel owns the Atlas review`.

Calculator and document evidence-support failures now feed the existing bounded answer-repair loop. The runtime tries one correction with the failed verification messages included in `missing_response_requirements`, then the deterministic evaluator judges the repaired answer. If the correction still conflicts with evidence, the final run remains failed.

The live reliability benchmark now includes calculator/document verification cases and reports `calculator_evidence_consistency_rate` plus `document_evidence_support_rate`, with provisional quality gates set to 1.0 for both.

Next Stage 2 work: run a live one-iteration benchmark with the local Qwen model, then inspect verification metrics and repair rates before deciding whether Stage 2 needs tuning.
