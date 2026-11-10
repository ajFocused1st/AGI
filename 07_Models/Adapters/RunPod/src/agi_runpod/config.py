"""Strict RunPod configuration with secrets excluded from representations."""

from __future__ import annotations

import re
from urllib.parse import urlsplit

from pydantic import BaseModel, ConfigDict, Field, SecretStr, field_validator, model_validator


class RunPodConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    api_key: SecretStr
    endpoint_id: str
    model: str = "qwen3-32b-awq"
    base_url: str | None = None
    timeout_seconds: float = Field(default=300.0, ge=1.0, le=600.0)
    max_retries: int = Field(default=1, ge=0, le=3)
    structured_output_retries: int = Field(default=1, ge=0, le=1)
    retry_backoff_seconds: float = Field(default=0.5, ge=0.0, le=5.0)
    max_tokens_per_request: int = Field(default=1200, ge=1, le=8192)
    max_prompt_characters: int = Field(default=200_000, ge=1, le=1_000_000)
    max_response_bytes: int = Field(default=4 * 1024 * 1024, ge=1024, le=16 * 1024 * 1024)

    @field_validator("api_key")
    @classmethod
    def require_key(cls, value: SecretStr) -> SecretStr:
        if not value.get_secret_value().strip():
            raise ValueError("RunPod API key must not be blank")
        return value

    @field_validator("endpoint_id")
    @classmethod
    def require_endpoint_id(cls, value: str) -> str:
        stripped = value.strip()
        if not re.fullmatch(r"[A-Za-z0-9_-]{1,128}", stripped):
            raise ValueError("RunPod endpoint ID contains invalid characters")
        return stripped

    @field_validator("model")
    @classmethod
    def require_model(cls, value: str) -> str:
        stripped = value.strip()
        if not stripped or len(stripped) > 256 or any(character.isspace() for character in stripped):
            raise ValueError("RunPod model must contain 1-256 non-whitespace characters")
        return stripped

    @model_validator(mode="after")
    def validate_url(self) -> "RunPodConfig":
        candidate = self.base_url or (
            f"https://api.runpod.ai/v2/{self.endpoint_id}/openai/v1"
        )
        parsed = urlsplit(candidate)
        if parsed.scheme != "https" or parsed.hostname != "api.runpod.ai":
            raise ValueError("RunPod base_url must use https://api.runpod.ai")
        if parsed.username or parsed.password or parsed.query or parsed.fragment:
            raise ValueError("RunPod base_url must not contain credentials, query, or fragment")
        expected_path = f"/v2/{self.endpoint_id}/openai/v1"
        if parsed.path.rstrip("/") != expected_path:
            raise ValueError("RunPod base_url path must match the configured endpoint ID")
        object.__setattr__(self, "base_url", candidate.rstrip("/"))
        return self
