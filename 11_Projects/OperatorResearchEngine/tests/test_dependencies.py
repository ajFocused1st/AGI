from __future__ import annotations

import unittest
from unittest.mock import patch

from agi_agent_runtime import ReasoningTier
from operator_research_engine.dependencies import build_engine
from operator_research_engine.settings import load_settings


class DependencyCompositionTests(unittest.TestCase):
    def test_runtime_critique_follows_reasoning_policy(self) -> None:
        with patch.dict("os.environ", {"OLLAMA_MODEL": "qwen3:8b"}, clear=False):
            base = load_settings().model_copy(update={"reasoning_tier": ReasoningTier.STANDARD})
            deep = load_settings().model_copy(update={"reasoning_tier": ReasoningTier.DEEP})

        with patch("operator_research_engine.dependencies.OllamaAdapter", return_value=object()):
            standard_engine = build_engine(base)
            deep_engine = build_engine(deep)

        self.assertFalse(standard_engine._runtime._enable_model_critique)
        self.assertTrue(deep_engine._runtime._enable_model_critique)
        self.assertIn("read_only_file_reader", standard_engine._runtime._tools.tools)


if __name__ == "__main__":
    unittest.main()
