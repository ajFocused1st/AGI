"""Provider telemetry helpers containing no prompts, output, or credentials."""


def structured_output_telemetry(attempt: int, *, valid: bool) -> dict[str, object]:
    return {
        "first_attempt_valid": attempt == 0 and valid,
        "repair_attempted": attempt > 0,
        "repair_succeeded": attempt > 0 and valid,
        "final_valid": valid,
        "attempt_count": attempt + 1,
    }


def http_status_category(status_code: int) -> str:
    if status_code in (401, 403):
        return "authentication"
    if status_code == 404:
        return "endpoint_or_model_not_found"
    if status_code == 429:
        return "rate_limited"
    if 400 <= status_code < 500:
        return "request_rejected"
    if status_code >= 500:
        return "provider_unavailable"
    return "success"
