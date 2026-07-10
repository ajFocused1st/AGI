from __future__ import annotations

import unittest
from pathlib import Path

from operator_research_engine.settings import ProjectSettings, load_settings


class ProjectSettingsTests(unittest.TestCase):
    def test_default_configuration_preserves_runtime_separation(self) -> None:
        settings = load_settings()
        self.assertEqual(settings.workspace_root, Path(r"C:\AGI"))
        self.assertEqual(settings.runtime_root, Path(r"C:\AGI_Runtime"))

    def test_runtime_inside_workspace_is_rejected(self) -> None:
        base = load_settings().model_dump()
        base["runtime_root"] = Path(r"C:\AGI\runtime")
        with self.assertRaises(ValueError):
            ProjectSettings.model_validate(base)


if __name__ == "__main__":
    unittest.main()
