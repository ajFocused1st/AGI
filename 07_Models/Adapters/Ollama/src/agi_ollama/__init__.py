"""Loopback-only Ollama adapter for the AGI model boundary."""

from .adapter import BaseModelAdapter, ModelAdapter, OllamaAdapter
from .config import OllamaConfig
from .errors import (
    OllamaAdapterError,
    OllamaConfigurationError,
    OllamaConnectionError,
    OllamaHTTPError,
    OllamaModelNotFoundError,
    OllamaProtocolError,
    OllamaRequestError,
    OllamaResponseTooLargeError,
    OllamaResponseValidationError,
    OllamaTimeoutError,
    OllamaTransportError,
)
from .scripted import ScriptedModelAdapter, ScriptedReply
from .transport import AsyncHttpTransport, HttpResponse, UrllibHttpTransport

__all__ = [
    "AsyncHttpTransport",
    "BaseModelAdapter",
    "HttpResponse",
    "ModelAdapter",
    "OllamaAdapter",
    "OllamaAdapterError",
    "OllamaConfig",
    "OllamaConfigurationError",
    "OllamaConnectionError",
    "OllamaHTTPError",
    "OllamaModelNotFoundError",
    "OllamaProtocolError",
    "OllamaRequestError",
    "OllamaResponseTooLargeError",
    "OllamaResponseValidationError",
    "OllamaTimeoutError",
    "OllamaTransportError",
    "ScriptedModelAdapter",
    "ScriptedReply",
    "UrllibHttpTransport",
]
