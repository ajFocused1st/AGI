"""Configuration with deliberately narrow network and resource bounds."""

from __future__ import annotations

from ipaddress import ip_address
from urllib.parse import urlsplit

from pydantic import BaseModel, ConfigDict, Field, field_validator


class OllamaConfig(BaseModel):
    """Runtime limits for an Ollama adapter instance.

    The endpoint is intentionally not configurable to a remote host. Supporting
    a remote model service belongs in a separate adapter with its own auth and
    threat model.
    """

    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    base_url: str = "http://127.0.0.1:11434"
    allowed_models: tuple[str, ...] = ()
    timeout_seconds: float = Field(default=30.0, ge=0.01, le=120.0)
    max_retries: int = Field(default=1, ge=0, le=3)
    structured_output_retries: int = Field(default=1, ge=0, le=2)
    retry_backoff_seconds: float = Field(default=0.25, ge=0.0, le=5.0)
    max_tokens_per_request: int = Field(default=4096, ge=1, le=8192)
    max_prompt_characters: int = Field(default=200_000, ge=1, le=1_000_000)
    max_response_bytes: int = Field(
        default=4 * 1024 * 1024,
        ge=1024,
        le=16 * 1024 * 1024,
    )

    @field_validator("allowed_models")
    @classmethod
    def validate_allowed_models(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        if len(value) > 16:
            raise ValueError("At most 16 Ollama models may be allowed")
        normalized: list[str] = []
        for model in value:
            stripped = model.strip()
            if not stripped or len(stripped) > 256:
                raise ValueError("Allowed Ollama model names must contain 1-256 characters")
            normalized.append(stripped)
        if len(set(normalized)) != len(normalized):
            raise ValueError("Allowed Ollama model names must be unique")
        return tuple(normalized)

    @field_validator("base_url")
    @classmethod
    def require_literal_loopback_url(cls, value: str) -> str:
        parsed = urlsplit(value)
        if parsed.scheme != "http":
            raise ValueError("Ollama base_url must use http")
        if not parsed.hostname:
            raise ValueError("Ollama base_url must include a host")
        if parsed.username or parsed.password:
            raise ValueError("Ollama base_url must not contain credentials")
        if parsed.query or parsed.fragment or parsed.path not in ("", "/"):
            raise ValueError("Ollama base_url must not contain a path, query, or fragment")

        hostname = parsed.hostname.casefold()
        is_loopback = hostname == "localhost"
        if not is_loopback:
            try:
                is_loopback = ip_address(hostname).is_loopback
            except ValueError:
                is_loopback = False
        if not is_loopback:
            raise ValueError("Ollama base_url must use a literal loopback host")

        try:
            port = parsed.port
        except ValueError as exc:
            raise ValueError("Ollama base_url contains an invalid port") from exc
        if port is not None and not 1 <= port <= 65_535:
            raise ValueError("Ollama base_url contains an invalid port")
        return value.rstrip("/")
