"""Typed failures emitted by the Ollama adapter."""

from __future__ import annotations


class OllamaAdapterError(Exception):
    """Base class for all adapter failures."""


class OllamaConfigurationError(OllamaAdapterError):
    """The adapter was configured with unsafe or invalid limits."""


class OllamaRequestError(OllamaAdapterError):
    """A shared model request cannot safely be sent to Ollama."""


class OllamaTransportError(OllamaAdapterError):
    """The local Ollama endpoint could not be reached reliably."""


class OllamaTimeoutError(OllamaTransportError):
    """A local Ollama request exceeded its finite deadline."""


class OllamaConnectionError(OllamaTransportError):
    """A connection to the local Ollama endpoint failed."""


class OllamaResponseTooLargeError(OllamaTransportError):
    """Ollama returned more bytes than the configured response ceiling."""


class OllamaHTTPError(OllamaAdapterError):
    """Ollama returned a non-success HTTP status."""

    def __init__(self, status_code: int, detail: str) -> None:
        self.status_code = status_code
        self.detail = detail[:1000]
        super().__init__(f"Ollama HTTP {status_code}: {self.detail}")


class OllamaModelNotFoundError(OllamaHTTPError):
    """The requested local model is absent; no automatic pull is attempted."""


class OllamaProtocolError(OllamaAdapterError):
    """Ollama returned a malformed or incomplete provider envelope."""


class OllamaResponseValidationError(OllamaAdapterError):
    """Generated structured output failed JSON or schema validation."""
