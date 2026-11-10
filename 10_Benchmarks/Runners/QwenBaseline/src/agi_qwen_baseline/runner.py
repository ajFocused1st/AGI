"""Sequential direct-Qwen baseline with auditable settings and incremental results."""

from __future__ import annotations

import argparse
import asyncio
import csv
import hashlib
import io
import json
import math
import os
import platform
import statistics
import subprocess
import sys
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from time import perf_counter
from typing import Any, Callable
from urllib.request import ProxyHandler, build_opener
from uuid import uuid4

from agi_contracts import ModelMessage, ModelRequest
from agi_ollama import OllamaAdapter, OllamaConfig
from agi_ollama.transport import UrllibHttpTransport

from .grading import grade, load_suite, strict_json

WORKSPACE = Path(__file__).resolve().parents[5]
DEFAULT_SUITE = WORKSPACE / "10_Benchmarks/Definitions/qwen_baseline_v1.json"
DEFAULT_OUTPUT = Path(r"C:\AGI_Runtime\benchmark-results\qwen-baseline-v1")
METRIC_FIELDS = (
    "done_reason", "prompt_eval_count", "eval_count", "total_duration",
    "load_duration", "prompt_eval_duration", "eval_duration",
)


@dataclass(frozen=True)
class Options:
    model: str = "qwen3:8b"
    think: bool = True
    context_tokens: int = 4096
    max_tokens: int = 2048
    temperature: float = 0.6
    seed: int = 42
    timeout_seconds: float = 120.0
    total_seconds: float = 3600.0
    repetitions: int = 1

    def validate(self) -> None:
        if not self.model.strip() or not 1 <= self.repetitions <= 5:
            raise ValueError("Model must be named; repetitions must be 1-5")
        if not 1024 <= self.context_tokens <= 8192:
            raise ValueError("Baseline context must be 1024-8192 tokens")
        if not 64 <= self.max_tokens <= self.context_tokens - 768:
            raise ValueError("Reserve at least 768 context tokens for these short input prompts")
        if not 1 <= self.timeout_seconds <= 120 or not 1 <= self.total_seconds <= 7200:
            raise ValueError("Timeout must be 1-120 seconds; total budget 1-7200 seconds")
        if not 0 <= self.temperature <= 2 or not 0 <= self.seed <= 2**31 - 1:
            raise ValueError("Temperature or seed outside supported bounds")


class MeasuredTransport(UrllibHttpTransport):
    """Count actual HTTP attempts and retain whitelisted metrics, never thinking traces."""

    def __init__(self) -> None:
        self.calls = 0
        self.last_metrics: dict[str, Any] = {}

    async def post_json(self, **kwargs: Any) -> Any:
        self.calls += 1
        self.last_metrics = {}
        response = await super().post_json(**kwargs)
        try:
            envelope = strict_json(response.body.decode("utf-8"))
            if isinstance(envelope, dict):
                self.last_metrics = {key: envelope[key] for key in METRIC_FIELDS if key in envelope}
        except (ValueError, UnicodeError):
            pass
        return response


def local_json(endpoint: str) -> dict[str, Any]:
    """Read fixed loopback metadata endpoints. Refuse redirects and ignore HTTP proxies."""
    if endpoint not in {"tags", "version", "ps"}:
        raise ValueError("Metadata endpoint is not allowlisted")
    url = f"http://127.0.0.1:11434/api/{endpoint}"
    from urllib.request import HTTPRedirectHandler

    class NoRedirect(HTTPRedirectHandler):
        def redirect_request(self, *args: Any, **kwargs: Any) -> None:
            return None

    with build_opener(ProxyHandler({}), NoRedirect()).open(url, timeout=5) as response:
        body = response.read(1_048_577)
    if len(body) > 1_048_576:
        raise ValueError("Metadata response too large")
    value = strict_json(body.decode("utf-8"))
    if not isinstance(value, dict):
        raise ValueError("Invalid metadata response")
    return value


def installed_model(model: str) -> dict[str, Any]:
    found = [item for item in local_json("tags").get("models", []) if item.get("name") == model]
    if len(found) != 1:
        raise ValueError(f"Exact model tag {model!r} is not installed; nothing was downloaded")
    item = found[0]
    return {key: item.get(key) for key in ("name", "digest", "size", "details")}


def command_output(arguments: list[str]) -> str | None:
    try:
        completed = subprocess.run(arguments, capture_output=True, text=True, timeout=10)
        return completed.stdout.strip() if completed.returncode == 0 else None
    except (OSError, subprocess.TimeoutExpired):
        return None


def hardware_snapshot() -> dict[str, Any]:
    return {
        "observed_at": datetime.now(UTC).isoformat(),
        "gpu_csv": command_output([
            "nvidia-smi", "--query-gpu=name,memory.total,memory.used", "--format=csv,noheader",
        ]),
        "note": "Point-in-time observation; not peak inference memory usage.",
    }


def source_hash() -> str:
    paths = list(Path(__file__).parent.glob("*.py"))
    paths.extend((WORKSPACE / "07_Models/Adapters/Ollama/src/agi_ollama").glob("*.py"))
    paths.extend((WORKSPACE / "01_Core/Contracts/src/agi_contracts").glob("*.py"))
    digest = hashlib.sha256()
    for path in sorted(paths):
        digest.update(path.relative_to(WORKSPACE).as_posix().encode())
        digest.update(b"\0")
        digest.update(path.read_bytes())
    return digest.hexdigest()


def atomic_json(path: Path, value: Any) -> None:
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
                         encoding="utf-8")
    os.replace(temporary, path)


def percentile(values: list[float], fraction: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    position = (len(ordered) - 1) * fraction
    lower, upper = math.floor(position), math.ceil(position)
    return round(ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower), 3)


def summarize(results: list[dict[str, Any]], planned: int) -> dict[str, Any]:
    latencies = [row["runtime_ms"] for row in results]
    categories = sorted({row["category"] for row in results})
    passed = sum(row["passed"] for row in results)
    return {
        "planned_attempts": planned,
        "attempted": len(results),
        "unattempted": planned - len(results),
        "passed": passed,
        "failed": len(results) - passed,
        "pass_rate_attempted": passed / len(results) if results else None,
        "pass_rate_planned": passed / planned,
        "model_calls": sum(row["model_calls"] for row in results),
        "tool_calls": 0,
        "truncated": sum(row["truncated"] for row in results),
        "errors": sum(row["error_type"] is not None for row in results),
        "median_latency_ms": percentile(latencies, 0.5),
        "p95_latency_ms": percentile(latencies, 0.95),
        "mean_latency_ms": round(statistics.mean(latencies), 3) if latencies else None,
        "measured_input_tokens": sum(row["metrics"].get("prompt_eval_count", 0) or 0 for row in results),
        "measured_output_tokens": sum(row["metrics"].get("eval_count", 0) or 0 for row in results),
        "attempts_missing_token_counts": sum(
            row["metrics"].get("eval_count") is None or row["metrics"].get("prompt_eval_count") is None
            for row in results
        ),
        "by_category": {
            category: {
                "attempted": sum(row["category"] == category for row in results),
                "passed": sum(row["passed"] and row["category"] == category for row in results),
            } for category in categories
        },
    }


def write_report(directory: Path, manifest: dict[str, Any], results: list[dict[str, Any]],
                 summary: dict[str, Any]) -> None:
    atomic_json(directory / "summary.json", summary)
    atomic_json(directory / "results.json", results)
    columns = ["case_id", "category", "iteration", "passed", "runtime_ms", "model_calls",
               "truncated", "error_type", "grading_detail", "answer"]
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=columns, extrasaction="ignore")
    writer.writeheader()
    writer.writerows(results)
    (directory / "results.csv").write_text(stream.getvalue(), encoding="utf-8")
    settings = manifest["options"]
    lines = [
        "# Plain Qwen baseline", "",
        f"Model: {settings['model']}; thinking: {settings['think']}; "
        f"context: {settings['context_tokens']}; output limit: {settings['max_tokens']}.", "",
        f"Run status: {manifest['status']}. "
        f"Passed {summary['passed']}/{summary['attempted']} attempted; "
        f"{summary['unattempted']} unattempted of {summary['planned_attempts']} planned.", "",
        f"Median latency: {summary['median_latency_ms']} ms; p95: {summary['p95_latency_ms']} ms. "
        f"Truncated: {summary['truncated']}; errors: {summary['errors']}.", "",
        "| Category | Passed | Attempted |", "|---|---:|---:|",
    ]
    for category, counts in summary["by_category"].items():
        lines.append(f"| {category} | {counts['passed']} | {counts['attempted']} |")
    lines.extend([
        "", "Each attempt has one fresh user message and no agent, tools, retrieval, "
        "schema-constrained decoding, critic, or repair. Native model thinking follows the "
        "recorded setting. Thinking text is discarded. Provider token counts are retained; "
        "missing counts are not estimated as zero-cost execution.", "",
        "Strict requested output format is part of task correctness. A format failure does "
        "not by itself establish a reasoning failure. These are 40 synthetic diagnostic "
        "cases, not a validated general-intelligence or professional-work benchmark. "
        "Coding covers tracing and diagnosis, not code generation. Four cases per category "
        "and one repetition offer limited statistical evidence.", "",
        "Latency includes model loading where applicable; per-attempt load_duration is "
        "recorded in nanoseconds. No model-unload operation is forced. Hardware snapshots "
        "are observations, not memory peaks. No improvement claim is possible until "
        "comparison conditions are run with documented equivalent settings.", "",
        "See results.json for prompts, grading rules, answers, errors, timings, and usage; "
        "manifest.json for model digest, source/definition hashes, and environment.",
    ])
    (directory / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


async def run_cases(*, suite: dict[str, Any], options: Options, adapter: Any, transport: Any,
                    directory: Path, progress: Callable[[str], None] = print) -> list[dict[str, Any]]:
    """Bounded execution shared by live CLI and network-free tests."""
    results: list[dict[str, Any]] = []
    started = perf_counter()
    planned = len(suite["cases"]) * options.repetitions
    for iteration in range(options.repetitions):
        for case in suite["cases"]:
            # Do not begin work without enough remaining time for its configured timeout.
            if options.total_seconds - (perf_counter() - started) < options.timeout_seconds:
                return results
            request = ModelRequest(
                model=options.model,
                messages=[ModelMessage(role="user", content=case["prompt"])],
                temperature=options.temperature, seed=options.seed + iteration,
                max_tokens=options.max_tokens,
            )
            before_calls = transport.calls
            transport.last_metrics = {}
            attempt_start = perf_counter()
            answer, error_type, error_message = "", None, None
            try:
                response = await adapter.generate(request)
                answer = response.content
                if "<think>" in answer or "</think>" in answer:
                    # Never persist legacy inline reasoning, including on error paths.
                    answer = ""
                    error_type = "InlineThinkingProtocolError"
                    error_message = "Inline thinking discarded; final answer was not safely separated"
            except Exception as exc:
                error_type = type(exc).__name__
                # Exception messages can contain raw provider output; do not persist them.
                error_message = "Model request failed; no retry was made"
            runtime_ms = round((perf_counter() - attempt_start) * 1000, 3)
            metrics = dict(transport.last_metrics)
            truncated = metrics.get("done_reason") == "length"
            passed, detail = grade(answer, case["grading"])
            if truncated or error_type:
                passed = False
                detail = "Output budget exhausted" if truncated else "Model request failed"
            row = {
                "case_id": case["id"], "category": case["category"], "iteration": iteration + 1,
                "seed": options.seed + iteration, "question": case["prompt"],
                "grading_rule": case["grading"], "answer": answer, "passed": passed,
                "grading_detail": detail, "runtime_ms": runtime_ms,
                "model_calls": transport.calls - before_calls, "metrics": metrics,
                "truncated": truncated, "error_type": error_type, "error_message": error_message,
            }
            results.append(row)
            with (directory / "attempts.jsonl").open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")
                handle.flush()
            progress(f"[{len(results)}/{planned}] {case['id']}: "
                     f"{'PASS' if passed else 'FAIL'} | {runtime_ms / 1000:.1f}s | "
                     f"tokens={metrics.get('eval_count', '?')} | {detail}")
            if error_type in {"OllamaTimeoutError", "OllamaConnectionError", "OllamaHTTPError",
                              "OllamaModelNotFoundError"}:
                # The server may still be working after a timeout; never queue more calls.
                return results
    return results


async def run_live(options: Options, suite_path: Path, output_root: Path) -> tuple[int, Path]:
    options.validate()
    suite = load_suite(suite_path)
    output_root = output_root.resolve()
    if output_root.is_relative_to(WORKSPACE.resolve()):
        raise ValueError("Generated results must remain outside C:\\AGI")
    model_info = installed_model(options.model)
    loaded_before = local_json("ps")
    other_models = [item.get("name") for item in loaded_before.get("models", [])
                    if item.get("name") != options.model]
    if other_models:
        raise ValueError("Another model is loaded; baseline will not load a second model")
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    directory = output_root / f"{stamp}_{'thinking' if options.think else 'direct'}_{uuid4().hex[:8]}"
    directory.mkdir(parents=True, exist_ok=False)
    manifest = {
        "benchmark_id": suite["benchmark_id"], "definition_version": suite["version"],
        "started_at": datetime.now(UTC).isoformat(), "status": "running",
        "options": asdict(options), "model_info": model_info,
        "definition_sha256": hashlib.sha256(suite_path.read_bytes()).hexdigest(),
        "source_sha256": source_hash(), "python": sys.version, "platform": platform.platform(),
        "ollama_version": local_json("version"), "loaded_before": loaded_before,
        "hardware_before": hardware_snapshot(),
        "git_commit": command_output(["git", "-c", f"safe.directory={WORKSPACE.as_posix()}",
                                      "-C", str(WORKSPACE), "rev-parse", "HEAD"]),
        "working_tree_status": command_output([
            "git", "-c", f"safe.directory={WORKSPACE.as_posix()}", "-C", str(WORKSPACE),
            "status", "--short",
        ]),
        "transport_retries": 0, "structured_output_retries": 0,
        "tools_enabled": False, "schema_constrained_decoding": False,
        "thinking_text_stored": False,
    }
    atomic_json(directory / "manifest.json", manifest)
    atomic_json(directory / "definition.json", suite)
    transport = MeasuredTransport()
    adapter = OllamaAdapter(
        OllamaConfig(
            allowed_models=(options.model,), timeout_seconds=options.timeout_seconds,
            max_retries=0, structured_output_retries=0, context_tokens=options.context_tokens,
            think=options.think, max_tokens_per_request=options.max_tokens,
        ), transport=transport,
    )
    print(f"Baseline output: {directory}", flush=True)
    results = await run_cases(suite=suite, options=options, adapter=adapter, transport=transport,
                              directory=directory, progress=lambda value: print(value, flush=True))
    summary = summarize(results, len(suite["cases"]) * options.repetitions)
    manifest["completed_at"] = datetime.now(UTC).isoformat()
    manifest["status"] = "completed" if not summary["unattempted"] else "incomplete"
    manifest["hardware_after"] = hardware_snapshot()
    try:
        manifest["loaded_after"] = local_json("ps")
        manifest["model_digest_unchanged"] = installed_model(options.model)["digest"] == model_info["digest"]
    except Exception:
        manifest["model_digest_unchanged"] = None
    if manifest["model_digest_unchanged"] is False:
        manifest["status"] = "invalid_model_changed"
    atomic_json(directory / "manifest.json", manifest)
    write_report(directory, manifest, results, summary)
    print(f"Result: {summary['passed']}/{summary['attempted']} passed; "
          f"{summary['unattempted']} unattempted. Report: {directory / 'report.md'}", flush=True)
    # Wrong answers are baseline data, not runner failure. Incomplete collection exits nonzero.
    return (0 if manifest["status"] == "completed" else 2), directory


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--suite", type=Path, default=DEFAULT_SUITE)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--validate-only", action="store_true")
    parser.add_argument("--recover-directory", type=Path,
                        help="Build an incomplete report from saved attempts; makes no model calls")
    parser.add_argument("--model", default="qwen3:8b")
    parser.add_argument("--thinking", choices=("on", "off"), default="on")
    parser.add_argument("--context-tokens", type=int, default=4096)
    parser.add_argument("--max-tokens", type=int, default=2048)
    parser.add_argument("--temperature", type=float, default=0.6)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--timeout-seconds", type=float, default=120.0)
    parser.add_argument("--total-seconds", type=float, default=3600.0)
    parser.add_argument("--repetitions", type=int, default=1)
    args = parser.parse_args(argv)
    options = Options(model=args.model, think=args.thinking == "on", context_tokens=args.context_tokens,
                      max_tokens=args.max_tokens, temperature=args.temperature, seed=args.seed,
                      timeout_seconds=args.timeout_seconds, total_seconds=args.total_seconds,
                      repetitions=args.repetitions)
    try:
        if args.recover_directory is not None:
            directory = args.recover_directory.resolve()
            if directory.is_relative_to(WORKSPACE.resolve()):
                raise ValueError("Recovery artifacts must remain outside the source workspace")
            manifest = strict_json((directory / "manifest.json").read_text(encoding="utf-8"))
            if manifest["status"] != "running":
                raise ValueError("Only an interrupted running record can be recovered")
            suite = load_suite(directory / "definition.json")
            results = []
            attempts_path = directory / "attempts.jsonl"
            if attempts_path.exists():
                for line in attempts_path.read_text(encoding="utf-8").splitlines():
                    results.append(strict_json(line))
            manifest["status"] = "interrupted"
            manifest["recovered_at"] = datetime.now(UTC).isoformat()
            manifest["recovery_note"] = (
                "Only completed persisted attempts are counted. A request interrupted before "
                "its result was saved may have consumed unmeasured compute. This run is not "
                "a complete baseline. Recovery did not call the model or change original results."
            )
            summary = summarize(results, len(suite["cases"]) * manifest["options"]["repetitions"])
            atomic_json(directory / "manifest.json", manifest)
            write_report(directory, manifest, results, summary)
            print(f"Recovered {len(results)} completed attempts: {directory / 'report.md'}")
            return 0
        options.validate()
        suite = load_suite(args.suite)
        if args.validate_only:
            print(f"Validated {len(suite['cases'])} cases across 10 categories; no model calls.")
            return 0
        code, _ = asyncio.run(run_live(options, args.suite, args.output_root))
        return code
    except KeyboardInterrupt:
        print("Interrupted. Completed attempts remain in attempts.jsonl.", file=sys.stderr)
        return 130
    except Exception as exc:
        print(f"Baseline could not complete: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
