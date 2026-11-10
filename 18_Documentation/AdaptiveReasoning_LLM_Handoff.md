# Adaptive reasoning integration: analysis and LLM handoff

Prepared 2026-09-12 from the supplied requirements and inspection of C:\AGI.

## What the user asked

The user supplied a specification for a local Adaptive Reasoning System and asked us to analyze it alongside the existing AGI folder, explain their alignment and whether merging the concepts is feasible, and explain the request and possible approach to another LLM. This turn requested analysis and a handoff. The implementation instructions inside the supplied specification were treated as material to analyze, not authorization to build the entire system during this review.

The intended product improves the effective reliability and reasoning performance of one local Qwen3 8B model through selective computation, tools, verification, explicit assumptions, retrieval, memory, skills, and evaluation. The objective is answer quality per unit of compute. FAST requests should normally cost one model call; difficult requests may use sequential critic, revision, and candidate passes. Reliability precedes autonomy.

The supplied constraints are Windows 11, i7-13700F, 16 GB RAM, RTX 3060 Ti with 8 GB VRAM, and Ollama. No simultaneous model loading, initial cloud dependency, autonomous browser, complex GUI, large vector database, or indefinite agent loops.

## Findings established during this review

- `nvidia-smi` reported RTX 3060 Ti, 8192 MiB VRAM, and 972 MiB used at the observation time. That is a transient reading, not inference peak consumption.
- `ollama --version` reported 0.31.2. `ollama list` showed qwen3:8b, ID 500a1f067a9f, size 5.2 GB.
- CPU and system RAM remain user-supplied specifications. The optional Windows CIM RAM query was denied; no claim of independent RAM verification is made.
- Across seven existing test roots, 80 tests ran: 78 passed and 2 skipped. The skipped checks were a symlink-creation test and the opt-in live Ollama smoke test. The benchmark integration test passed its ten scripted cases.
- PowerShell policy prevented launching Test.ps1 directly. Its unittest discovery commands were run through the existing project Python environment instead, without changing execution policy or installing dependencies.
- No live model inference or paired Qwen quality benchmark was performed. Passing offline tests does not establish increased model capability.
- The working tree already contained staged, unstaged, and untracked development work, including RunPod integration. Preserve it. Do not reset, overwrite, or commit someone else's changes as part of this proposal.
- Many numbered directories are ownership scaffolding. In particular, 03_Cognition and 04_Memory contain descriptive READMEs, not completed cognition and memory systems.

## Existing implementation and alignment

| Requirement | Existing implementation | Integration implication |
|---|---|---|
| Modular architecture | Shared src-layout packages; OperatorResearchEngine composes them | Extend existing ownership boundaries instead of copying a new standalone AI-Reasoner tree |
| Structured task state | 01_Core/Contracts: requests, plans, evidence, requirements, evaluations, run records | Add governor decisions, budgets, assumptions, verification results, and stop decisions here |
| Orchestration | 02_Agents/Runtime: injected runtime and bounded correction paths | Add adaptive strategy selection and an overall budget; current sequence is largely fixed |
| Ollama access | 07_Models/Adapters/Ollama: loopback transport, allowlisted models, schema validation, bounded retries, usage | Reuse; add explicit context and thinking controls and comprehensive attempt accounting |
| Deterministic tools | 06_Tools: restricted AST calculator and read-only snippet search behind a registry | Strong initial tool foundation; arbitrary Python execution is absent |
| Verification | 01_Core/Evaluation: rule checks, citation integrity, tool trace, expected text/markers | Useful infrastructure; not a general semantic truth verifier |
| Critic | Same injected model produces structured advisory critique | Reuse; add severity/issues and a bounded critic-to-revision transition |
| Retrieval | Literal search across permitted text-like files with snippet and scan limits | Extend beyond benchmark fixture roots deliberately; semantic retrieval and PDF extraction remain future work |
| Logging | Atomic redacted JSON run records; usage, timings, evidence, decisions, errors | Add route, escalation, all actual model attempts, verification status, and stopping reason |
| Evaluation | Scripted ten-case benchmark and sequential live reliability harness | Add paired raw-Qwen/current-runtime/adaptive comparisons and new reasoning categories |
| Terminal UI | CLI accepts --goal and optional JSON output | Add an interactive loop and optional route/status display if desired |
| Memory and reusable skills | No implemented general memory store or dynamic procedural skill loader found | Build later in their shared ownership areas |

## Important differences and risks

1. The current runtime normally calls the model for plan, answer, and critique. Corrections and adapter retries can add calls. A FAST label around that same pipeline would not satisfy the one-call requirement. Add a conservative deterministic pre-router; use model-based classification only when its cost is justified. Route scores are estimates, not measured truth, and unavailable uncertainty should be explicit rather than invented.
2. The critic currently runs after answer correction and is advisory. It does not initiate another answer revision. Existing bounded plan/argument/response repairs are useful but are not a FAST-to-DEEP-to-MAX escalation policy.
3. Citation IDs resolving to captured evidence do not prove the evidence entails a claim. The evaluator also uses expected substrings and fixture markers. Introduce task-specific numeric, constraint, code-test, and source-support checks, with states such as verified, contradicted, insufficient evidence, and unverifiable.
4. Inspect unsupported-claim propagation before extending evaluation: preliminary/corrected evaluation includes uncited generated claims, while the final evaluator call in runtime.py receives known_unsupported_claims. Persist a consistent evidence/claim assessment across passes. This is an observed code path to examine, not a demonstrated exploit.
5. Independent prompts to the same model can still yield correlated errors. Candidate agreement is supporting evidence, not independent verification. Use two candidates initially, at most three, generated sequentially with genuinely different approaches.
6. A timeout and working directory alone do not sandbox arbitrary Python. Keep the existing calculator for initial deterministic work. Before generated-code execution, select a Windows-compatible isolation boundary with restricted files, network, processes, and resource limits; test containment. Do not enable unrestricted exec or shell through the tool registry.
7. The adapter sets num_predict and temperature but does not explicitly send num_ctx, think, or keep_alive. Runtime role calls currently request 2048/4096/1024 output tokens; project max_tokens is not an effective overall task budget. Character limits do not enforce context fit, especially when schemas are inserted into prompts.
8. Current local configuration leaves the Ollama model empty; live benchmark defaults still say llama3:latest. A new local profile should explicitly select qwen3:8b without pulling models automatically.
9. RunPod code exists despite older architecture documents describing remote providers as deferred. Keep that work as an optional separate capability; explicitly select Ollama for this local reasoning product. A local composition root can avoid importing cloud dependencies unnecessarily.
10. Current live-benchmark baseline fields contain fixed historical reliability values. They are not a fresh Qwen-alone experimental control. Preserve their meaning and add a separate comparison.
11. General persistent memory, procedural skills, fact/assumption/inference/unknown separation, and evidence-based stopping are substantive missing features. Directory names and agent-profile YAML files do not implement them.

## Proposed integration boundary

Use one runtime authority for transitions, permissions, budgets, and stopping. Avoid wrapping the existing complete runtime in another uncontrolled loop.

```text
C:\AGI\
  01_Core\Contracts\                 shared reasoning and run contracts
  01_Core\Evaluation\                verifier interfaces and deterministic checks
  02_Agents\Runtime\                 execution, escalation, budgets, persistence
  03_Cognition\AdaptiveReasoning\    new agi_reasoning package
    src\agi_reasoning\
      governor.py
      policies.py
      stop_condition.py
      assumptions.py                 later
      candidates.py                  later
      prompts\                      role-specific packaged resources
    tests\
  03_Cognition\Skills\              later procedural skills and loader
  04_Memory\                        later SQLite and retrieval implementation
  06_Tools\                         existing tools; isolated Python later
  07_Models\Adapters\Ollama\        existing adapter with bounded controls
  10_Benchmarks\                    versioned cases and rubrics
  11_Projects\LocalReasoner\         thin local configuration and terminal UI
```

LocalReasoner is a proposed composition root. Retain OperatorResearchEngine as a regression target. The new application imports shared packages and does not duplicate them. Keep dependencies directed toward contracts: cognition proposes decisions; runtime enforces them; evaluation supplies checks; projects select concrete implementations.

Actual databases, logs, model artifacts, and generated benchmark results remain under C:\AGI_Runtime, following the repository's existing convention. The sample specification's in-repository memory.db and logs folders should therefore be adapted. Obtain filesystem authorization if the execution environment requires it for writes outside the workspace.

## Suggested execution policy

These are initial experimental ceilings, not validated optimal settings or existing functionality. Count classification, generation, repair, critic, revision, and synthesis calls against one task budget, including adapter retry attempts. Escalation increases the task ceiling; it never resets usage.

| Level | Initial model-call ceiling | Intended path |
|---|---:|---|
| FAST | 1 | Cheap host routing, direct answer, cheap applicable checks |
| STANDARD | 3 | Optional planning/retrieval/tool work, answer, bounded repair when needed |
| DEEP | 5 | Solve with tools, inspect assumptions, critic, targeted revision, verification |
| MAX | 8 | Two or three sequential candidates, evidence comparison, targeted correction/synthesis |

Also cap total tokens, wall time, tool calls, retrieval volume, and revisions. A ceiling is not a quota. The host can format an already-supported answer without another finalizer model call.

Stop successfully when the task's required checks pass and material contradictions are resolved. If evidence is missing, choose a conditional answer or request the missing premise. Stop unsuccessfully or partially on exhausted resources, unavailable required tools, or no new evidence/progress; report uncertainty. Never make every open-ended task wait for impossible deterministic proof. Do not ask the model to estimate further value indefinitely.

Governor contracts should include all ten requested dimensions: complexity, ambiguity, dependent steps, current-information need, math need, code need, retrieval need, consequence, verifiability, and uncertainty. Include selected level, skill, recommended tools, short decision summary, assessment source, and budget. Use bounded values and nullable unknowns, then validate and clamp decisions in host code. The governor cannot grant permissions. Current-information need without a current source should produce an explicit limitation, not more unsupported reasoning.

## Hardware feasibility

The local package is installed and the GPU matches the specification. A bounded single-model application is a reasonable engineering target, but full GPU residency, latency, and quality require live measurement. The 5.2 GB model file is not total runtime VRAM: context cache, execution buffers, and display use also consume memory. The official model listing identifies this tag as Q4_K_M: https://ollama.com/library/qwen3:8b.

Start experimentally around 4096 total context tokens, with room reserved for output and any thinking tokens, and short retrieved passages. Increase only after measuring fit and quality. Larger context increases memory demand; Ollama documents context allocation and checking offloading with ollama ps: https://docs.ollama.com/context-length.

Use one active generation at a time, including sequential candidate passes. Proposed server settings are OLLAMA_NUM_PARALLEL=1 and OLLAMA_MAX_LOADED_MODELS=1; these are recommendations, not settings changed during this review. Ollama documents concurrency-related context memory growth: https://docs.ollama.com/faq.

Expose a per-call thinking control, beginning with think=false for FAST and short structured control calls, and measuring where thinking helps harder generation. Do not confuse FAST/STANDARD/DEEP/MAX host policies with provider thinking values. Ollama documents Qwen3 thinking support and separate thinking/content fields: https://docs.ollama.com/capabilities/thinking. Discard raw thinking traces rather than storing them in logs; retain decisions, results, and usage. Watch output truncation and JSON repair cost.

Keep the first version on ordinary Python plus existing Pydantic, PyYAML, and jsonschema dependencies. Pydantic is validation infrastructure, not PydanticAI. SQLite can use the standard library. Existing root-level FastAPI/uvicorn dependencies do not make a web service necessary. No hardware purchase, extra model, orchestration framework, vector server, or container service is required to start the initial governor work.

## Recommended development sequence if implementation is subsequently requested

1. Preserve current work and establish reproducibility. Run the existing tests, record model digest and configuration, and add a small paired Qwen-alone evaluation before changing reasoning behavior. Keep scripted tests as regression checks. Add a raw-Qwen condition with the same source excerpts for document tasks to distinguish retrieval gains from reasoning gains.
2. Propose and implement the smallest foundation: reasoning contracts, cheap governor, one-call FAST path, shared budgets, Ollama context/thinking controls, and route/attempt logs. Test live Qwen JSON compatibility and latency. Do not present DEEP/MAX as complete until their actual strategies exist; report deferred capability explicitly.
3. Reuse calculator/search for STANDARD and add task-specific verification. Test unavailable tools, malformed output, missing evidence, and exhausted budgets. Count all actual requests, including repair calls hidden inside adapters.
4. Add bounded critic/revision, assumption handling, escalation, and stopping. Test a clean answer that stops immediately, a recoverable mistake, a critical missing premise, conflicting evidence, and an unresolvable task that terminates honestly.
5. Add sequential candidates only where measured benefit justifies the extra calls. Add SQLite memory and dynamically selected procedural skills afterward. Keep memory provenance and verification state; generated answers must not automatically become trusted facts.
6. Expand evaluations continuously, rather than delaying all benchmarking to the specification's Phase 6. Compare raw Qwen, the existing fixed runtime, and the adaptive runtime on held-out cases. Include ablations for tools, critique, and candidates; repeated runs; identical model/quantization; documented thinking/context settings; and matched token or time budgets where useful.

Report task correctness, unsupported-claim rate, ambiguity handling, instruction compliance, actual model calls, tokens, median/p95 latency, tool outcomes, escalation frequency, stopping reasons, and memory use. Separate cold model loading from warm execution. Do not claim improved intelligence based only on safety tests, expected-word matches, model self-confidence, or candidate agreement. The desired result is demonstrated task-level improvement within resource limits, not a claim that the model has become AGI.
