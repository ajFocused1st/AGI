"""The sole composition root for concrete adapters, tools, and policies."""

from __future__ import annotations

from pathlib import Path

import yaml
from agi_agent_runtime import AgentRuntime, AtomicJsonRunStore
from agi_contracts import AgentProfile
from agi_evaluation import RuleBasedEvaluator
from agi_ollama import OllamaAdapter, OllamaConfig
from agi_tools import (
    Calculator,
    ReadOnlyFileSearch,
    ReadOnlyFileSearchConfig,
    SearchRoot,
    ToolRegistry,
)

from .runtime import OperatorResearchEngine
from .settings import ProjectSettings


def build_engine(settings: ProjectSettings) -> OperatorResearchEngine:
    """Bind the one-model, two-tool first slice without hidden fallbacks."""

    model_name = settings.ollama.model.strip()
    if not model_name:
        raise ValueError(
            "OLLAMA_MODEL must name an already-installed model; models are never pulled automatically"
        )
    profile = _load_profile(settings.profile_path).model_copy(update={"model": model_name})
    configured_roots = tuple(str(path) for path in settings.document_roots)
    profile_roots = tuple(str(Path(path).resolve()) for path in profile.allowed_document_roots)
    if set(configured_roots) != set(profile_roots):
        raise ValueError("profile document roots must exactly match trusted project settings")

    registry = ToolRegistry()
    registry.register(Calculator())
    registry.register(
        ReadOnlyFileSearch(
            ReadOnlyFileSearchConfig(
                roots=tuple(
                    SearchRoot(label=f"documents{index}", path=path)
                    for index, path in enumerate(settings.document_roots, start=1)
                )
            )
        )
    )
    adapter = OllamaAdapter(
        OllamaConfig(
            base_url=settings.ollama.base_url,
            allowed_models=(model_name,),
            timeout_seconds=settings.ollama.timeout_seconds,
            max_retries=settings.ollama.max_retries,
            max_tokens_per_request=max(settings.ollama.max_tokens, 4096),
        )
    )
    runtime = AgentRuntime(
        model_adapter=adapter,
        tool_executor=registry,
        evaluator=RuleBasedEvaluator(),
        run_store=AtomicJsonRunStore(
            settings.runtime_root / "logs",
            forbidden_root=settings.workspace_root,
        ),
        enable_model_critique=True,
    )
    return OperatorResearchEngine(runtime=runtime, profile=profile)


def _load_profile(path: Path) -> AgentProfile:
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError("agent profile must be a mapping")
    return AgentProfile.model_validate(raw)
