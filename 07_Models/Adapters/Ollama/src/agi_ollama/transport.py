"""Small asynchronous HTTP transport backed only by the Python standard library."""

from __future__ import annotations

import asyncio
import json
import socket
from dataclasses import dataclass
from typing import Any, Mapping, Protocol
from urllib import error, request

from .errors import (
    OllamaConnectionError,
    OllamaResponseTooLargeError,
    OllamaTimeoutError,
)


@dataclass(frozen=True, slots=True)
class HttpResponse:
    """Transport-neutral HTTP response used by the adapter and test fakes."""

    status_code: int
    body: bytes
    headers: Mapping[str, str]


class AsyncHttpTransport(Protocol):
    """Minimal injected transport surface; intentionally excludes general HTTP."""

    async def post_json(
        self,
        *,
        url: str,
        payload: Mapping[str, Any],
        timeout_seconds: float,
        max_response_bytes: int,
    ) -> HttpResponse:
        """POST JSON and return a size-bounded response."""


class UrllibHttpTransport:
    """Async wrapper over urllib; requests run in a worker thread."""

    async def post_json(
        self,
        *,
        url: str,
        payload: Mapping[str, Any],
        timeout_seconds: float,
        max_response_bytes: int,
    ) -> HttpResponse:
        return await asyncio.to_thread(
            self._post_json_sync,
            url,
            payload,
            timeout_seconds,
            max_response_bytes,
        )

    @staticmethod
    def _post_json_sync(
        url: str,
        payload: Mapping[str, Any],
        timeout_seconds: float,
        max_response_bytes: int,
    ) -> HttpResponse:
        try:
            encoded = json.dumps(
                payload,
                ensure_ascii=False,
                allow_nan=False,
                separators=(",", ":"),
            ).encode("utf-8")
        except (TypeError, ValueError) as exc:
            raise OllamaConnectionError(f"Could not encode Ollama request: {exc}") from exc

        http_request = request.Request(
            url,
            data=encoded,
            headers={
                "Accept": "application/json",
                "Content-Type": "application/json; charset=utf-8",
                "User-Agent": "agi-ollama/0.1",
            },
            method="POST",
        )
        try:
            with request.urlopen(http_request, timeout=timeout_seconds) as response:
                body = _read_bounded(response, max_response_bytes)
                return HttpResponse(
                    status_code=int(response.status),
                    body=body,
                    headers=dict(response.headers.items()),
                )
        except error.HTTPError as exc:
            body = _read_bounded(exc, max_response_bytes)
            return HttpResponse(
                status_code=int(exc.code),
                body=body,
                headers=dict(exc.headers.items()) if exc.headers else {},
            )
        except (socket.timeout, TimeoutError) as exc:
            raise OllamaTimeoutError("Timed out waiting for local Ollama") from exc
        except error.URLError as exc:
            if isinstance(exc.reason, (socket.timeout, TimeoutError)):
                raise OllamaTimeoutError("Timed out waiting for local Ollama") from exc
            raise OllamaConnectionError(
                f"Could not connect to local Ollama: {exc.reason}"
            ) from exc
        except OSError as exc:
            raise OllamaConnectionError(f"Local Ollama transport failed: {exc}") from exc


def _read_bounded(stream: Any, max_response_bytes: int) -> bytes:
    body = stream.read(max_response_bytes + 1)
    if len(body) > max_response_bytes:
        raise OllamaResponseTooLargeError(
            f"Ollama response exceeded {max_response_bytes} bytes"
        )
    return body
