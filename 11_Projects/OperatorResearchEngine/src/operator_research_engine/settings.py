"""Validated project settings with explicit environment overrides."""

from __future__ import annotations

import os
from pathlib import Path

from agi_agent_runtime import ReasoningTier
import yaml
from pydantic import BaseModel, ConfigDict, Field, SecretStr, field_validator, model_validator


class OllamaSettings(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    base_url: str = "http://127.0.0.1:11434"
    model: str = ""
    timeout_seconds: float = Field(default=30, ge=0.1, le=120)
    max_retries: int = Field(default=1, ge=0, le=3)
    temperature: float = Field(default=0, ge=0, le=2)
    max_tokens: int = Field(default=2048, ge=1, le=8192)
    # Sent as Ollama's num_ctx. Without it Ollama uses its own default (often 4,096 tokens)
    # and silently cuts off the start of any longer prompt, evidence included.
    context_tokens: int | None = Field(default=8192, ge=512, le=131_072)


class RunPodSettings(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    api_key: SecretStr | None = None
    endpoint_id: str = ""
    base_url: str | None = None
    model: str = "qwen3-32b-awq"
    timeout_seconds: float = Field(default=300, ge=1, le=600)
    max_retries: int = Field(default=1, ge=0, le=3)
    temperature: float = Field(default=0.1, ge=0, le=2)
    max_tokens: int = Field(default=1200, ge=1, le=8192)


class ProjectSettings(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, protected_namespaces=())

    environment: str = "development"
    workspace_root: Path
    runtime_root: Path
    profile_path: Path
    document_roots: tuple[Path, ...] = Field(min_length=1, max_length=16)
    model_provider: str = "ollama"
    reasoning_tier: ReasoningTier = ReasoningTier.STANDARD
    ollama: OllamaSettings
    runpod: RunPodSettings = Field(default_factory=RunPodSettings)

    @field_validator("model_provider")
    @classmethod
    def validate_provider(cls, value: str) -> str:
        normalized = value.strip().lower()
        if normalized not in {"ollama", "runpod"}:
            raise ValueError("MODEL_PROVIDER must be either 'ollama' or 'runpod'")
        return normalized

    @field_validator("reasoning_tier", mode="before")
    @classmethod
    def normalize_reasoning_tier(cls, value: object) -> str:
        return str(value).strip().lower()

    @field_validator("workspace_root", "runtime_root", "profile_path", mode="before")
    @classmethod
    def normalize_path(cls, value: object) -> Path:
        return Path(str(value)).expanduser().resolve()

    @field_validator("document_roots", mode="before")
    @classmethod
    def normalize_roots(cls, value: object) -> tuple[Path, ...]:
        if isinstance(value, (str, bytes)):
            raise ValueError("document_roots must be a list")
        return tuple(Path(str(item)).expanduser().resolve() for item in value)

    @model_validator(mode="after")
    def validate_storage_and_inputs(self) -> "ProjectSettings":
        if _is_within(self.runtime_root, self.workspace_root):
            raise ValueError("runtime_root must remain outside the source workspace")
        if not self.profile_path.is_file():
            raise ValueError("profile_path must identify an existing profile")
        for root in self.document_roots:
            if not root.is_dir():
                raise ValueError(f"document root is unavailable: {root}")
            if not _is_within(root, self.workspace_root):
                raise ValueError("v1 document roots must be versioned workspace fixtures")
        return self


def load_settings(config_path: str | Path | None = None) -> ProjectSettings:
    project_root = Path(__file__).resolve().parents[2]
    workspace_root = project_root.parents[1]
    path = (
        Path(config_path).expanduser().resolve()
        if config_path is not None
        else project_root / "configs" / "development.yaml"
    )
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError("project configuration must be a mapping")
    raw["workspace_root"] = workspace_root

    ollama = dict(raw.get("ollama") or {})
    if "OLLAMA_BASE_URL" in os.environ:
        ollama["base_url"] = os.environ["OLLAMA_BASE_URL"]
    if "OLLAMA_MODEL" in os.environ:
        ollama["model"] = os.environ["OLLAMA_MODEL"]
    raw["ollama"] = ollama
    if "MODEL_PROVIDER" in os.environ:
        raw["model_provider"] = os.environ["MODEL_PROVIDER"]
    if "AGI_REASONING_TIER" in os.environ:
        raw["reasoning_tier"] = os.environ["AGI_REASONING_TIER"]
    runpod = dict(raw.get("runpod") or {})
    environment_fields = {
        "RUNPOD_API_KEY": "api_key",
        "RUNPOD_ENDPOINT_ID": "endpoint_id",
        "RUNPOD_BASE_URL": "base_url",
        "RUNPOD_MODEL": "model",
        "RUNPOD_TIMEOUT_SECONDS": "timeout_seconds",
        "RUNPOD_MAX_RETRIES": "max_retries",
        "RUNPOD_TEMPERATURE": "temperature",
        "RUNPOD_MAX_TOKENS": "max_tokens",
    }
    for environment_name, field_name in environment_fields.items():
        if environment_name in os.environ:
            runpod[field_name] = os.environ[environment_name]
    raw["runpod"] = runpod
    if "AGI_RUNTIME_ROOT" in os.environ:
        raw["runtime_root"] = os.environ["AGI_RUNTIME_ROOT"]
    if "AGI_DOCUMENT_ROOT" in os.environ:
        raw["document_roots"] = [os.environ["AGI_DOCUMENT_ROOT"]]
    return ProjectSettings.model_validate(raw)


def _is_within(candidate: Path, root: Path) -> bool:
    candidate_text = os.path.normcase(os.path.abspath(candidate))
    root_text = os.path.normcase(os.path.abspath(root))
    try:
        return os.path.commonpath((candidate_text, root_text)) == root_text
    except ValueError:
        return False
