"""Minimal authenticated HTTP transport with bounded response reads."""

from __future__ import annotations

import asyncio
import json
import socket
from dataclasses import dataclass
from typing import Any, Mapping, Protocol
from urllib import error, request

from .errors import RunPodConnectionError, RunPodResponseTooLargeError, RunPodTimeoutError


@dataclass(frozen=True, slots=True)
class HttpResponse:
    status_code: int
    body: bytes
    headers: Mapping[str, str]


class AsyncHttpTransport(Protocol):
    async def post_json(
        self, *, url: str, payload: Mapping[str, Any], authorization: str,
        timeout_seconds: float, max_response_bytes: int,
    ) -> HttpResponse: ...


class UrllibHttpTransport:
    async def post_json(
        self, *, url: str, payload: Mapping[str, Any], authorization: str,
        timeout_seconds: float, max_response_bytes: int,
    ) -> HttpResponse:
        return await asyncio.to_thread(
            self._post_json_sync, url, payload, authorization,
            timeout_seconds, max_response_bytes,
        )

    @staticmethod
    def _post_json_sync(
        url: str, payload: Mapping[str, Any], authorization: str,
        timeout_seconds: float, max_response_bytes: int,
    ) -> HttpResponse:
        try:
            encoded = json.dumps(payload, ensure_ascii=False, allow_nan=False).encode("utf-8")
        except (TypeError, ValueError) as exc:
            raise RunPodConnectionError("Could not encode RunPod request") from exc
        http_request = request.Request(
            url, data=encoded,
            headers={
                "Accept": "application/json", "Content-Type": "application/json",
                "Authorization": authorization, "User-Agent": "agi-runpod/0.1",
            }, method="POST",
        )
        try:
            with request.urlopen(http_request, timeout=timeout_seconds) as response:
                return HttpResponse(int(response.status), _read_bounded(response, max_response_bytes), dict(response.headers.items()))
        except error.HTTPError as exc:
            return HttpResponse(int(exc.code), _read_bounded(exc, max_response_bytes), dict(exc.headers.items()) if exc.headers else {})
        except (socket.timeout, TimeoutError) as exc:
            raise RunPodTimeoutError("Timed out waiting for RunPod") from exc
        except error.URLError as exc:
            if isinstance(exc.reason, (socket.timeout, TimeoutError)):
                raise RunPodTimeoutError("Timed out waiting for RunPod") from exc
            raise RunPodConnectionError("Could not connect to RunPod") from exc
        except OSError as exc:
            raise RunPodConnectionError("RunPod transport failed") from exc


def _read_bounded(stream: Any, limit: int) -> bytes:
    body = stream.read(limit + 1)
    if len(body) > limit:
        raise RunPodResponseTooLargeError(f"RunPod response exceeded {limit} bytes")
    return body
