"""Score the engine against a dataset's answer key (10_Benchmarks/Datasets/<name>/answer_key.jsonl).

The answer key is the referee: it says whether each answer is right. The engine's own verdict is
then compared with it. A wrong answer the engine marked "passed" is a false pass, the dangerous
kind of disagreement because a user would trust it. A right answer it did not pass is a safe miss.

Usage (from 11_Projects/OperatorResearchEngine):
  python -m operator_research_engine.answer_key check <dataset>
  python -m operator_research_engine.answer_key run <dataset> [--config configs/real.yaml] [--runs 3]
"""

from __future__ import annotations

import argparse
import asyncio
import json
import re
import sys
from collections.abc import Iterable
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from agi_agent_runtime.runtime import FALLBACK_NOTE
from agi_evaluation import contains_uncertainty_statement

from operator_research_engine.settings import ProjectSettings, load_settings

WORKSPACE_ROOT = Path(__file__).resolve().parents[4]
DATASETS_ROOT = WORKSPACE_ROOT / "10_Benchmarks" / "Datasets"
PROFILES_ROOT = WORKSPACE_ROOT / "02_Agents" / "Profiles"

_NOT_STATED = re.compile(
    r"\b(not (published|disclosed|available|stated|mentioned|recorded|specified|provided)|withheld|"
    r"(none|no part) of the (text|evidence|documents?|excerpts?|passages?) "
    r"(mentions|states|identifies|names|specifies))\b"
)


def normalize(text: str) -> str:
    text = text.casefold().replace("’", "'").replace("“", '"').replace("”", '"')
    text = re.sub(r"(?<=\d),(?=\d{3}\b)", "", text)  # 14,235 -> 14235
    return " ".join(text.split())


def load_key(path: Path) -> list[dict[str, Any]]:
    items = []
    for number, line in enumerate(path.read_text(encoding="utf-8-sig").splitlines(), 1):
        if line.strip():
            try:
                items.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"answer key line {number} is not valid JSON: {exc}") from exc
    return items


def is_correct(item: dict[str, Any], answer: str) -> bool:
    """Whether an answer matches the key. Passages shown in place of an answer never count."""
    if answer.startswith(FALLBACK_NOTE):
        return False
    normalized = normalize(answer)
    accepted = [a for a in [item.get("answer", "")] + list(item.get("accept") or []) if a]
    if item.get("not_in_documents"):
        return (
            contains_uncertainty_statement(answer)
            or any(normalize(a) in normalized for a in item.get("accept") or [] if a)
            or bool(_NOT_STATED.search(normalized))
        )
    return any(normalize(a) in normalized for a in accepted)


def rejected_mentions(item: dict[str, Any], answer: str) -> list[str]:
    normalized = normalize(answer)
    return [r for r in item.get("reject") or [] if r and normalize(r) in normalized]


@dataclass(frozen=True)
class Dataset:
    name: str
    root: Path
    key: list[dict[str, Any]]

    @property
    def documents(self) -> Path:
        return self.root / "documents"

    @property
    def profile(self) -> Path:
        return PROFILES_ROOT / f"researcher_{self.name.removesuffix('_v1')}.yaml"


def load_dataset(name: str) -> Dataset:
    if not re.fullmatch(r"[A-Za-z0-9_]+", name):
        raise ValueError("dataset must be a bare dataset folder name")
    root = DATASETS_ROOT / name
    return Dataset(name=name, root=root, key=load_key(root / "answer_key.jsonl"))


def check(dataset: Dataset) -> int:
    """Check the key itself: quotes appear verbatim and calculations are right."""
    documents = {
        path.name: normalize(path.read_text(encoding="utf-8"))
        for path in sorted(dataset.documents.glob("*.txt"))
    }
    problems = 0
    counts: dict[str, int] = {}
    for item in dataset.key:
        category = item.get("category", "?")
        counts[category] = counts.get(category, 0) + 1
        for source in item.get("sources") or []:
            text = documents.get(source.get("file"))
            if text is None:
                print(f"Q{item['id']}: unknown file {source.get('file')!r}")
                problems += 1
            elif normalize(source.get("quote", "")) not in text:
                print(f"Q{item['id']}: quote not found verbatim in {source['file']}")
                problems += 1
        calculation = item.get("calculation")
        if calculation:
            left, _, right = calculation.partition("=")
            if not re.fullmatch(r"[\d\s.,+\-*/()]+", left):
                print(f"Q{item['id']}: calculation has more than numbers and + - * /: {calculation!r}")
                problems += 1
                continue
            value = eval(left.replace(",", ""), {"__builtins__": {}})  # noqa: S307 - digits only
            if abs(float(value) - float(right.replace(",", "").strip())) > 1e-9:
                print(f"Q{item['id']}: calculation wrong: {calculation}")
                problems += 1
    print("categories:", counts, "| total:", sum(counts.values()))
    print("problems found:", problems)
    return problems


def score(item: dict[str, Any], answer: str, verdict: str, evidence_text: str | None) -> dict[str, Any]:
    correct = is_correct(item, answer)
    quotes = [s["quote"] for s in item.get("sources") or []]
    retrieved = None
    if evidence_text is not None and quotes:
        retrieved = all(normalize(q) in normalize(evidence_text) for q in quotes)
    return {
        "id": item["id"],
        "category": item.get("category"),
        "correct": correct,
        "app_verdict": verdict,
        "false_pass": verdict == "passed" and not correct,
        "safe_miss": verdict != "passed" and correct,
        "fallback": answer.startswith(FALLBACK_NOTE),
        "mentions_rejected": rejected_mentions(item, answer),
        "evidence_had_quotes": retrieved,
        "question": item["question"],
        "expected": item.get("answer"),
        "answer": answer,
    }


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    total = len(rows)
    by_category: dict[str, dict[str, int]] = {}
    for row in rows:
        entry = by_category.setdefault(row["category"], {"total": 0, "right": 0, "agree": 0})
        entry["total"] += 1
        entry["right"] += row["correct"]
        entry["agree"] += not (row["false_pass"] or row["safe_miss"])
    return {
        "total": total,
        "right": sum(r["correct"] for r in rows),
        "agreement": sum(not (r["false_pass"] or r["safe_miss"]) for r in rows),
        "false_passes": [r["id"] for r in rows if r["false_pass"]],
        "safe_misses": [r["id"] for r in rows if r["safe_miss"]],
        "fallbacks": [r["id"] for r in rows if r["fallback"]],
        "crashes": [r["id"] for r in rows if r["app_verdict"] == "crashed"],
        "by_category": by_category,
    }


def across_runs(runs: list[list[dict[str, Any]]]) -> dict[str, Any]:
    ids = [row["id"] for row in runs[0]]
    right_counts = {i: sum(row["correct"] for run in runs for row in run if row["id"] == i) for i in ids}
    false_pass_counts = {
        i: sum(row["false_pass"] for run in runs for row in run if row["id"] == i) for i in ids
    }
    totals = [summarize(run) for run in runs]
    return {
        "runs": len(runs),
        "right_per_run": [t["right"] for t in totals],
        "agreement_per_run": [t["agreement"] for t in totals],
        "false_passes_per_run": [len(t["false_passes"]) for t in totals],
        "always_right": [i for i, n in right_counts.items() if n == len(runs)],
        "sometimes_right": [i for i, n in right_counts.items() if 0 < n < len(runs)],
        "never_right": [i for i, n in right_counts.items() if n == 0],
        "ever_false_pass": [i for i, n in false_pass_counts.items() if n],
    }


def settings_for(dataset: Dataset, config: Path) -> ProjectSettings:
    settings = load_settings(config)
    return settings.model_copy(
        update={
            "profile_path": dataset.profile.resolve(),
            "document_roots": (dataset.documents.resolve(),),
        }
    )


async def run_once(dataset: Dataset, settings: ProjectSettings, output: Path) -> list[dict[str, Any]]:
    from operator_research_engine.dependencies import build_engine

    engine = build_engine(settings)
    rows = []
    with output.open("w", encoding="utf-8") as handle:
        for item in dataset.key:
            evidence_text = None
            run_id = ""
            try:
                response = await engine.research(item["question"])
                answer, verdict, run_id = (
                    response.answer,
                    response.evaluation.verdict.value,
                    str(response.run_id),
                )
                log = settings.runtime_root / "logs" / run_id / "run.json"
                if log.exists():
                    record = json.loads(log.read_text(encoding="utf-8"))
                    evidence_text = " ".join(
                        e.get("excerpt") or "" for e in record["evidence_bundle"]["items"]
                    )
            except Exception as exc:  # noqa: BLE001 - a crash is a result too
                answer, verdict = f"{type(exc).__name__}: {exc}", "crashed"
            row = score(item, answer, verdict, evidence_text) | {"run_id": run_id}
            rows.append(row)
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
            handle.flush()
            flag = " FALSE PASS" if row["false_pass"] else ""
            print(
                f"Q{row['id']:>2} {row['category']:<16} {'RIGHT' if row['correct'] else 'wrong':5} "
                f"verdict={verdict:<8} evidence={row['evidence_had_quotes']}{flag}",
                flush=True,
            )
    return rows


def _print_summary(summary: dict[str, Any]) -> None:
    print("\ncategory          right   app agrees")
    for category, entry in sorted(summary["by_category"].items()):
        print(
            f"{category:<16} {entry['right']:>3}/{entry['total']:<3} {entry['agree']:>5}/{entry['total']}"
        )
    print(
        f"TOTAL right {summary['right']}/{summary['total']}; verdict agrees "
        f"{summary['agreement']}/{summary['total']}; false passes {summary['false_passes']}; "
        f"safe misses {summary['safe_misses']}; fallbacks {summary['fallbacks']}; "
        f"crashes {summary['crashes']}"
    )


async def run(dataset: Dataset, config: Path, runs: int) -> Path:
    settings = settings_for(dataset, config)
    stamp = datetime.now(UTC).strftime("%Y-%m-%d_%H%M%S")
    directory = settings.runtime_root / "answer_keys" / dataset.name / stamp
    directory.mkdir(parents=True, exist_ok=False)
    all_runs = []
    for number in range(1, runs + 1):
        print(f"\n== run {number}/{runs} ==", flush=True)
        rows = await run_once(dataset, settings, directory / f"run{number}.jsonl")
        _print_summary(summarize(rows))
        all_runs.append(rows)
    report = {
        "dataset": dataset.name,
        "config": str(config),
        "model": settings.ollama.model,
        "per_run": [summarize(rows) for rows in all_runs],
        "across_runs": across_runs(all_runs),
    }
    (directory / "summary.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    if runs > 1:
        print("\nacross runs:", json.dumps(report["across_runs"]))
    print(f"results: {directory}")
    return directory


def main(argv: Iterable[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    commands = parser.add_subparsers(dest="command", required=True)
    check_parser = commands.add_parser("check", help="verify the key against its documents")
    check_parser.add_argument("dataset")
    run_parser = commands.add_parser("run", help="ask every question and score the answers")
    run_parser.add_argument("dataset")
    run_parser.add_argument("--config", type=Path, default=Path("configs/real.yaml"))
    run_parser.add_argument("--runs", type=int, default=1)
    args = parser.parse_args(list(argv) if argv is not None else None)
    dataset = load_dataset(args.dataset)
    if args.command == "check":
        return 1 if check(dataset) else 0
    if args.runs < 1:
        parser.error("--runs must be at least 1")
    asyncio.run(run(dataset, args.config, args.runs))
    return 0


if __name__ == "__main__":
    sys.exit(main())
