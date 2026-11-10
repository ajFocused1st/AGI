"""Command-line entry point for a live, explicitly configured Ollama run."""

from __future__ import annotations

import argparse
import asyncio
from collections.abc import Sequence

from .dependencies import build_engine
from .settings import load_settings


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Research approved local documents safely.")
    parser.add_argument("--goal", required=True, help="Research question or calculation goal")
    parser.add_argument("--config", help="Path to a project YAML configuration")
    parser.add_argument(
        "--tier",
        choices=("fast", "standard", "deep", "max"),
        help="Reasoning budget tier for this run",
    )
    parser.add_argument("--json", action="store_true", help="Print the full response as JSON")
    return parser


async def _run(arguments: argparse.Namespace) -> int:
    settings = load_settings(arguments.config)
    if arguments.tier:
        settings = settings.model_copy(update={"reasoning_tier": arguments.tier})
    response = await build_engine(settings).research(arguments.goal)
    if arguments.json:
        print(response.model_dump_json(indent=2))
    else:
        print(response.answer)
        print(f"\nrun_id: {response.run_id}")
        if response.evaluation is not None:
            print(f"evaluation: {response.evaluation.verdict.value}")
    return 0 if response.status.value == "completed" else 1


def main(argv: Sequence[str] | None = None) -> int:
    arguments = build_parser().parse_args(argv)
    return asyncio.run(_run(arguments))


if __name__ == "__main__":
    raise SystemExit(main())
