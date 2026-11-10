"""Typed, secret-safe RunPod adapter failures."""


class RunPodAdapterError(Exception):
    """Base class for RunPod failures."""


class RunPodRequestError(RunPodAdapterError):
    """A shared request cannot safely be sent."""


class RunPodTransportError(RunPodAdapterError):
    """The endpoint could not be reached reliably."""


class RunPodTimeoutError(RunPodTransportError):
    """A request exceeded its finite deadline."""


class RunPodConnectionError(RunPodTransportError):
    """A connection failed."""


class RunPodResponseTooLargeError(RunPodTransportError):
    """A response exceeded its configured ceiling."""


class RunPodHTTPError(RunPodAdapterError):
    """RunPod returned a non-success HTTP status."""

    def __init__(self, status_code: int, category: str, detail: str) -> None:
        self.status_code = status_code
        self.category = category
        self.detail = detail[:500]
        super().__init__(f"RunPod HTTP {status_code} ({category}): {self.detail}")


class RunPodProtocolError(RunPodAdapterError):
    """RunPod returned an invalid provider envelope."""


class RunPodResponseValidationError(RunPodAdapterError):
    """Generated structured output failed validation."""
