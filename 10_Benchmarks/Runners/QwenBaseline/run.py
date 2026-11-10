"""Run with the existing OperatorResearchEngine Python environment; no installation needed."""

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from agi_qwen_baseline.runner import main  # noqa: E402

if __name__ == "__main__":
    raise SystemExit(main())
