from __future__ import annotations

import unittest
from contextlib import redirect_stderr
from io import StringIO

from operator_research_engine.app import build_parser


class AppParserTests(unittest.TestCase):
    def test_tier_argument_is_public_and_restricted(self) -> None:
        parsed = build_parser().parse_args(["--goal", "test", "--tier", "deep"])

        self.assertEqual(parsed.tier, "deep")

        with redirect_stderr(StringIO()), self.assertRaises(SystemExit):
            build_parser().parse_args(["--goal", "test", "--tier", "expensive"])


if __name__ == "__main__":
    unittest.main()
