# Adaptive reasoning current sprint

Current stage: Stage 2, tools and verification.

Completed:

- Stage 0 direct Qwen baseline runner and reports.
- Explicit FAST/STANDARD/DEEP/MAX routing policies.
- Budget governor for model attempts, tool attempts, token totals, and wall-clock limits.
- One-call FAST path runner with no tools, retrieval, critique, revision, or native thinking.
- Budget governor wired into `AgentRuntime.run(...)` for planning, plan correction, tool argument repair, answer generation, answer correction, critique, and actual tool dispatch.
- Run records now persist the budget ledger in `metadata["budget"]`.
- Budget exhaustion now persists a failed run with a `budget_exceeded` error before raising `RuntimeExecutionError`.
- Host precheck validates routing/profile/tool compatibility before model calls begin.
- OperatorResearchEngine exposes public routing selection through `reasoning_tier`, `AGI_REASONING_TIER`, and the CLI `--tier` argument.
- Stage 2 started: successful calculator results now become host-owned evidence with deterministic hashes.
- Calculator evidence can be cited by answer drafts and validated by response requirements.
- Deterministic evaluation now includes a critical `calculator_evidence_consistency` check, so an answer citing calculator evidence fails if it omits the cited result value.
- Deterministic evaluation now includes a critical `document_evidence_support` check, so document-cited answers fail when they invent title-cased entities or lack enough concrete overlap with cited excerpts.
- Calculator/document evidence-support failures now feed the existing bounded answer-repair loop. The runtime tries one correction, then the deterministic evaluator judges the repaired answer.
- Live reliability benchmark now includes calculator/document verification cases and reports `calculator_evidence_consistency_rate` plus `document_evidence_support_rate`.
- Ollama structured-output handling now falls back from schema grammar mode to JSON mode when local Ollama rejects the schema with `failed to parse grammar`. Host-side schema validation still remains authoritative.
- OperatorResearchEngine standard-tier Ollama runs now pass `think=false`, matching the routing policy and preventing Qwen reasoning text from leaking into structured JSON responses.
- Live benchmark controlled runtime failures now preserve `run_id`, read persisted run records when available, and no longer count safe `RuntimeExecutionError` failures as unhandled harness exceptions.
- Read-only file search now has a safe token fallback for inflected fact queries, so queries like `Project Atlas launch date` can match `Project Atlas launched on 2025-02-14`.
- Standard-tier tool budget increased from 2 to 8 tool calls to allow bounded retrieval and repair work without prematurely failing common grounding tasks.
- Read-only file search now handles completion/completed variants and generic comparison queries more reliably.
- Deterministic fallback answers now deduplicate repeated evidence and prioritize comparison measurements such as sample counts and completion times before owner/background lines.
- Live benchmark quality-gate aggregation now treats non-tool and non-grounded cases neutrally for tool-argument and evidence-claim rates, while preserving per-run diagnostics.
- Answer and critique prompts now explicitly constrain Qwen to the allowed top-level schema keys to reduce extra-field structured-output repairs.
- OperatorResearchEngine now follows the selected routing policy for advisory critique, so STANDARD runs skip critique while DEEP/MAX can still enable it.
- Ollama structured-output telemetry now records redacted first-error categories and schema paths, and run records preserve that telemetry for benchmark diagnosis.

Live Qwen benchmark checkpoints:

- Before transport fixes: 0/14 passed, all attempts failed at the first model call with `OllamaHTTPError`.
- After JSON-mode fallback and `think=false`: 4/14 passed, `unhandled_exception_rate=0.0`; remaining failures were functional quality gates.
- After search token fallback: 4/14 passed, citation coverage improved from `0.000` to `0.167`.
- After standard tool budget increase: 5/14 passed, citation coverage improved to `0.333`, grounding task pass rate improved to `0.111`, and unhandled exception rate remained `0.0`.
- After evaluator citation-label cleanup and safer retrieval term matching: 6/14 passed, final pass rate improved to `0.429`, grounding task pass rate improved to `0.222`, response correction success improved to `0.500`, and unhandled exception rate remained `0.0`.
- After tolerating and dropping null tool-call placeholders in plans: 7/14 passed, final structured success reached `1.000`, first-attempt structured success reached `1.000`, retrieval compliance improved to `0.778`, safety denial reached `1.000`, calculator/document verification cases passed, and unhandled exception rate remained `0.0`.
- After partial-subject and generic-document search fallback: 9/14 passed, final pass rate improved to `0.643`, retrieval compliance improved to `0.889`, retrieval non-empty result rate improved to `0.889`, citation coverage improved to `0.667`, claims-with-evidence rate reached `1.000`, and unhandled exception rate remained `0.0`.
- After deterministic evidence fallback for failed answer correction: 10/14 passed, final pass rate improved to `0.714`, grounding task pass rate improved to `0.667`, citation coverage improved to `0.833`, calculator/document evidence support rates improved to `0.929`, controlled failure rate dropped to `0.214`, and unhandled exception rate remained `0.0`.
- After dropping invalid optional tool proposals, expanding the standard tool budget to 6, and fixing ungrounded-task claim handling: 12/14 passed, final pass rate reached `0.857`, final structured success stayed `1.000`, response correction success reached `1.000`, safety denial stayed `1.000`, tool argument final validity reached `1.000`, and unhandled exception rate remained `0.0`.
- After expanding the standard tool budget to 8 and improving comparison evidence selection: 14/14 cases passed, final pass rate reached `1.000`, grounding task pass rate reached `1.000`, citation coverage stayed `1.000`, retrieval compliance stayed `1.000`, safety denial stayed `1.000`, final structured success stayed `1.000`, and unhandled exception rate remained `0.0`.
- After refining quality-gate aggregation and adding stricter answer/critique schema prompts: 14/14 cases still passed, claims-with-evidence reached `1.000`, tool argument final validity reached `1.000`, controlled failure rate reached `0.000`, mean latency improved to `13195.358ms`, and first-attempt structured success remained `0.571`. The benchmark process still returns a nonzero exit because first-attempt structured success is the only remaining failed quality gate.
- After making STANDARD follow `allow_critique=false` and preserving redacted structured-error telemetry: 14/14 cases still passed, claims-with-evidence stayed `1.000`, tool argument final validity stayed `1.000`, controlled failure rate reached `0.000`, mean latency improved to `10082.279ms`, and first-attempt structured success remained `0.571`. The remaining first-attempt failures are answer/answer-correction root-level schema validation failures.
- Rejected experiment: a simplified model-facing answer schema raised first-attempt structured success to `0.857`, but it reduced final pass rate to `0.643` by weakening citations and grounded claims. The runtime was restored to the richer `AnswerDraft` schema because final correctness is the higher-priority gate.
- Latest result directory: `C:\AGI_Runtime\benchmark-results\live-reliability-v1\2026-09-14_124425_f9f944e0`.

Validation:

```powershell
& .\11_Projects\OperatorResearchEngine\.venv\Scripts\python.exe -B -m unittest discover -v -s .\02_Agents\Runtime\tests -p 'test_*.py'
ruff check .\02_Agents\Runtime\src\agi_agent_runtime .\02_Agents\Runtime\tests
powershell -NoProfile -ExecutionPolicy Bypass -File .\11_Projects\OperatorResearchEngine\scripts\Test.ps1
```

Next task:

Continue Stage 2 by reducing repair dependence and improving strict quality gates:

1. Improve first-attempt structured output reliability so fewer live runs need schema repair.
2. Use the preserved `first_error` telemetry to reduce answer-schema root failures without weakening grounded claims or citation behavior.
3. Tighten comparison query planning to reduce duplicate retrieval and lower latency.
4. Re-run the one-iteration live benchmark and compare against the latest checkpoint above.
