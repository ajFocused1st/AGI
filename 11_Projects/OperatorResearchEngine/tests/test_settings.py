from __future__ import annotations

import unittest
from unittest.mock import patch
from pathlib import Path

from agi_agent_runtime import ReasoningTier
from operator_research_engine.settings import ProjectSettings, load_settings


class ProjectSettingsTests(unittest.TestCase):
    def test_default_configuration_preserves_runtime_separation(self) -> None:
        settings = load_settings()
        self.assertEqual(settings.workspace_root, Path(r"C:\AGI"))
        self.assertEqual(settings.runtime_root, Path(r"C:\AGI_Runtime"))
        self.assertEqual(settings.reasoning_tier, ReasoningTier.STANDARD)

    def test_reasoning_tier_environment_override_is_validated(self) -> None:
        with patch.dict("os.environ", {"AGI_REASONING_TIER": "deep"}, clear=False):
            settings = load_settings()

        self.assertEqual(settings.reasoning_tier, ReasoningTier.DEEP)

    def test_runtime_inside_workspace_is_rejected(self) -> None:
        base = load_settings().model_dump()
        base["runtime_root"] = Path(r"C:\AGI\runtime")
        with self.assertRaises(ValueError):
            ProjectSettings.model_validate(base)

    def test_invalid_reasoning_tier_is_rejected(self) -> None:
        base = load_settings().model_dump()
        base["reasoning_tier"] = "expensive"
        with self.assertRaises(ValueError):
            ProjectSettings.model_validate(base)


if __name__ == "__main__":
    unittest.main()
