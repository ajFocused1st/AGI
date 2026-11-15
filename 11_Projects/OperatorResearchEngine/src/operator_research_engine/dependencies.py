"""The sole composition root for concrete adapters, tools, and policies."""

from __future__ import annotations

from pathlib import Path

import yaml
from agi_agent_runtime import (
    AgentRuntime,
    AtomicJsonRunStore,
    run_host_precheck,
    get_routing_policy,
)
from agi_contracts import AgentProfile
from agi_evaluation import RuleBasedEvaluator
from agi_ollama import OllamaAdapter, OllamaConfig
from agi_runpod import RunPodAdapter, RunPodConfig
from agi_tools import (
    Calculator,
    ReadOnlyFileReader,
    ReadOnlyFileReaderConfig,
    ReadOnlyFileSearch,
    ReadOnlyFileSearchConfig,
    SearchRoot,
    ToolRegistry,
)

from .runtime import OperatorResearchEngine
from .settings import ProjectSettings


def build_engine(settings: ProjectSettings) -> OperatorResearchEngine:
    """Bind the one-model, two-tool first slice without hidden fallbacks."""

    routing_policy = get_routing_policy(settings.reasoning_tier)
    if settings.model_provider == "ollama":
        model_name = settings.ollama.model.strip()
        if not model_name:
            raise ValueError(
                "OLLAMA_MODEL must name an already-installed model; models are never pulled automatically"
            )
        adapter = OllamaAdapter(
            OllamaConfig(
                base_url=settings.ollama.base_url,
                allowed_models=(model_name,),
                timeout_seconds=settings.ollama.timeout_seconds,
                max_retries=settings.ollama.max_retries,
                max_tokens_per_request=max(settings.ollama.max_tokens, 4096),
                think=routing_policy.native_thinking,
                context_tokens=settings.ollama.context_tokens,
            )
        )
    elif settings.model_provider == "runpod":
        model_name = settings.runpod.model.strip()
        if settings.runpod.api_key is None:
            raise ValueError("RUNPOD_API_KEY is required when MODEL_PROVIDER=runpod")
        adapter = RunPodAdapter(
            RunPodConfig(
                api_key=settings.runpod.api_key,
                endpoint_id=settings.runpod.endpoint_id,
                base_url=settings.runpod.base_url,
                model=model_name,
                timeout_seconds=settings.runpod.timeout_seconds,
                max_retries=settings.runpod.max_retries,
                max_tokens_per_request=max(settings.runpod.max_tokens, 4096),
            )
        )
    else:
        raise ValueError(f"Unsupported model provider: {settings.model_provider}")
    profile = _load_profile(settings.profile_path).model_copy(update={"model": model_name})
    configured_roots = tuple(str(path) for path in settings.document_roots)
    profile_roots = tuple(str(Path(path).resolve()) for path in profile.allowed_document_roots)
    if set(configured_roots) != set(profile_roots):
        raise ValueError("profile document roots must exactly match trusted project settings")

    registry = ToolRegistry()
    registry.register(Calculator())
    document_roots = tuple(
        SearchRoot(label=f"documents{index}", path=path)
        for index, path in enumerate(settings.document_roots, start=1)
    )
    registry.register(
        ReadOnlyFileSearch(
            ReadOnlyFileSearchConfig(
                roots=document_roots
            )
        )
    )
    registry.register(
        ReadOnlyFileReader(
            ReadOnlyFileReaderConfig(
                roots=document_roots
            )
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
        enable_model_critique=routing_policy.allow_critique,
        routing_policy=settings.reasoning_tier,
    )
    run_host_precheck(
        policy=routing_policy,
        profile=profile,
        tools=registry.tools,
    )
    return OperatorResearchEngine(runtime=runtime, profile=profile)


def _load_profile(path: Path) -> AgentProfile:
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError("agent profile must be a mapping")
    return AgentProfile.model_validate(raw)
