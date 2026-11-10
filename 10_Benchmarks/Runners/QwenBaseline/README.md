# Plain Qwen baseline

Forty fixed synthetic cases, four per requested category. One direct Ollama request per attempt. Model-native thinking is configurable and enabled in the initial condition. No tools, agent governor, retrieval, critique, hidden repair, or schema-constrained decoding. Raw thinking text is never written to results.

Use the existing project environment from `C:\AGI`; no new dependencies are required:

```powershell
& .\11_Projects\OperatorResearchEngine\.venv\Scripts\python.exe -B .\10_Benchmarks\Runners\QwenBaseline\run.py --validate-only
& .\11_Projects\OperatorResearchEngine\.venv\Scripts\python.exe -B -m unittest discover -s .\10_Benchmarks\Runners\QwenBaseline\tests
& .\11_Projects\OperatorResearchEngine\.venv\Scripts\python.exe -B .\10_Benchmarks\Runners\QwenBaseline\run.py
```

The live command uses the already-installed `qwen3:8b`, writes under `C:\AGI_Runtime\benchmark-results\qwen-baseline-v1`, and may require execution-environment permission for that output directory. It never pulls a model or falls back to a cloud provider. It refuses to begin if another model is already loaded.

Defaults: thinking on, 4096 context tokens, 2048 generated tokens, temperature 0.6, seed 42, 120 seconds per request, 3600 seconds total, one repetition. `--thinking off`, `--repetitions 3`, and other options produce separately labeled runs. A repeat uses seed + iteration index. Settings and generation counts are recorded; seeds do not guarantee bitwise repeatability across hardware/backend versions.

The baseline shares the local adapter and its validation dependencies, but bypasses the agent runtime. The adapter now accepts optional `context_tokens` and `think`; omitted settings preserve previous payload behavior. Ollama documents [chat settings and usage fields](https://docs.ollama.com/api/chat) and [model identities](https://docs.ollama.com/api/tags).

Each output directory contains a definition snapshot, manifest with model digest and code/definition hashes, incremental `attempts.jsonl`, complete results in JSON/CSV, summary JSON, and a Markdown report. Existing artifacts are never overwritten by a new run. Interrupted runs retain completed attempt records. A timeout stops collection to avoid submitting another request while the server may still be working.

Exit 0 means collection completed, even if Qwen answered incorrectly. Exit 2 indicates incomplete/invalid collection; exit 1 indicates a setup/runner error. `--validate-only` makes no model calls and writes no artifacts.

If the process has ended before reporting, use `run.py --recover-directory <interrupted-run-directory>` to generate an explicitly interrupted report from completed records without another model call. First confirm the original process is no longer running. Recovery cannot recover token counts for a request interrupted before its response was saved. Start a new full run in a new directory for a complete baseline.

Graders accept exact answers, case-insensitive single choices, finite numeric answers within explicit tolerance, strict JSON values, or any valid task ordering. They reject contradictory alternatives, duplicate JSON keys, extra required-format content, and truncated answers. They do not merely search for expected substrings. Coding tasks do not execute generated code. This is an initial diagnostic suite, not a validated general capability benchmark; use new held-out cases and repeated comparisons before claiming improvement.
