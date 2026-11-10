"""Offline, model-free retrieval scoring against benchmark expected_facts (zero tokens).

Usage:  python -m operator_research_engine.retrieval_eval [definition ...]
"""

from __future__ import annotations

import argparse
import asyncio
import json
import re
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from agi_tools import ReadOnlyFileSearch, ReadOnlyFileSearchConfig, SearchRoot
from agi_tools.file_search import ReadOnlyFileSearchInput

WORKSPACE_ROOT = Path(__file__).resolve().parents[4]
DEFAULT_DEFINITIONS = ("live_reliability_v1", "live_reliability_holdout_v1")
QUERY_STYLES = ("goal", "subject")


def _queries(case: dict[str, Any]) -> dict[str, str]:
    subjects = case.get("requirements", {}).get("task_subjects", [])
    return {"goal": case["goal"][:200], "subject": " ".join(subjects)[:200] or case["goal"][:200]}


async def score_definition(name: str) -> dict[str, Any]:
    definition = json.loads(
        (WORKSPACE_ROOT / "10_Benchmarks" / "Definitions" / f"{name}.json").read_text(encoding="utf-8")
    )
    documents = WORKSPACE_ROOT / "10_Benchmarks" / "Datasets" / definition["dataset_version"] / "documents"
    search = ReadOnlyFileSearch(
        ReadOnlyFileSearchConfig(roots=(SearchRoot(label="documents1", path=documents),))
    )
    totals = {style: {"expected": 0, "found": 0, "returned": 0, "relevant": 0, "rr": 0.0} for style in QUERY_STYLES}
    cases = [case for case in definition["cases"] if case.get("expected_facts")]
    for case in cases:
        expected = {f"[{fact}]" for fact in case["expected_facts"]}
        for style, query in _queries(case).items():
            output = await search.run(ReadOnlyFileSearchInput(query=query, max_matches=20))
            snippets = [match.snippet for match in output.matches]
            hits = [any(marker in snippet for marker in expected) for snippet in snippets]
            found = {marker for marker in expected if any(marker in s for s in snippets)}
            first = next((index for index, hit in enumerate(hits, start=1) if hit), None)
            bucket = totals[style]
            bucket["expected"] += len(expected)
            bucket["found"] += len(found)
            bucket["returned"] += len(snippets)
            bucket["relevant"] += sum(hits)
            bucket["rr"] += 1.0 / first if first else 0.0
    return {
        "definition": name,
        "cases": len(cases),
        **{
            style: {
                "recall": round(b["found"] / b["expected"], 3) if b["expected"] else None,
                "precision": round(b["relevant"] / b["returned"], 3) if b["returned"] else 0.0,
                "mrr": round(b["rr"] / len(cases), 3) if cases else None,
                "avg_returned": round(b["returned"] / len(cases), 1) if cases else 0.0,
            }
            for style, b in totals.items()
        },
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("definitions", nargs="*", default=list(DEFAULT_DEFINITIONS))
    arguments = parser.parse_args(argv)
    for name in arguments.definitions:
        if not re.fullmatch(r"[A-Za-z0-9_]+", name):
            raise ValueError(f"invalid definition name: {name}")
        print(json.dumps(asyncio.run(score_definition(name))))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
