"""Golden run-record snapshots: any behavior change in the governed loop must be deliberate.

Regenerate after an intended change with:  $env:AGI_UPDATE_GOLDEN = "1"; scripts\\Test.ps1
then review the snapshot diff in Git before committing.
"""

from __future__ import annotations

import json
import os
import unittest

from golden_scenarios import GOLDEN_PATH, diff_paths, run_all


class GoldenSnapshotTests(unittest.IsolatedAsyncioTestCase):
    async def test_run_records_match_golden_snapshot(self) -> None:
        actual = await run_all()
        if os.environ.get("AGI_UPDATE_GOLDEN") == "1":
            GOLDEN_PATH.parent.mkdir(parents=True, exist_ok=True)
            GOLDEN_PATH.write_text(
                json.dumps(actual, indent=1, sort_keys=True, ensure_ascii=False) + "\n",
                encoding="utf-8",
            )
            self.skipTest("golden snapshot regenerated")
        self.assertTrue(GOLDEN_PATH.is_file(), "missing snapshot; run with AGI_UPDATE_GOLDEN=1")
        expected = json.loads(GOLDEN_PATH.read_text(encoding="utf-8"))
        differences = diff_paths(expected, actual)
        self.assertEqual(differences, [], "run records drifted:\n" + "\n".join(differences))

    async def test_answer_source_labels_who_wrote_the_answer(self) -> None:
        sources = {case_id: record["answer_source"] for case_id, record in (await run_all()).items()}
        self.assertEqual(sources.pop("06-compare-atlas-beacon"), "host_fallback")
        self.assertEqual(sources.pop("07-missing-information"), "model_correction")
        self.assertEqual(set(sources.values()), {"model"})

    async def test_snapshot_is_deterministic(self) -> None:
        first, second = await run_all(), await run_all()
        self.assertEqual(diff_paths(first, second), [])


if __name__ == "__main__":
    unittest.main()
