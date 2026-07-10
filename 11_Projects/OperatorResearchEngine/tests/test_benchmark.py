from __future__ import annotations

import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from operator_research_engine.benchmark import run_benchmark


class BenchmarkIntegrationTests(unittest.IsolatedAsyncioTestCase):
    async def test_all_ten_offline_cases_pass(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            with patch.dict(os.environ, {"AGI_RUNTIME_ROOT": temporary}):
                exit_code, result_path = await run_benchmark()
            summary = json.loads(Path(result_path).read_text(encoding="utf-8"))
            self.assertEqual(exit_code, 0)
            self.assertEqual(summary["case_count"], 10)
            self.assertEqual(summary["passed"], 10)
            self.assertEqual(summary["required_safety_failures"], [])


if __name__ == "__main__":
    unittest.main()
