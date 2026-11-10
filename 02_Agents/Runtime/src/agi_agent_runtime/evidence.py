"""Host-owned construction of bounded evidence bundles from approved tool results."""

from __future__ import annotations

import hashlib
import json

from agi_contracts import EvidenceBundle, EvidenceItem, ToolRequest, ToolResult


def build_evidence_bundle(
    requests: list[ToolRequest], results: list[ToolResult]
) -> EvidenceBundle:
    request_by_id = {request.request_id: request for request in requests}
    items: list[EvidenceItem] = []
    for result in results:
        if not result.success:
            continue
        if result.tool_name == "calculator":
            item = _calculator_evidence_item(result)
            if item is not None:
                items.append(item)
            continue
        if result.tool_name != "read_only_file_search":
            continue
        request = request_by_id.get(result.request_id)
        query = request.arguments.get("query") if request else None
        if not isinstance(query, str) or not query.strip():
            continue
        matches = result.output.get("matches")
        if not isinstance(matches, list):
            continue
        for match in matches:
            if not isinstance(match, dict):
                continue
            source_hash = match.get("content_sha256")
            if not isinstance(source_hash, str) or len(source_hash) != 64:
                continue
            excerpt = _sanitize_excerpt(str(match.get("snippet", "")))
            if not excerpt:
                continue
            items.append(
                EvidenceItem(
                    source_id=str(match["source_id"]),
                    source_hash=source_hash,
                    excerpt=excerpt,
                    retrieval_query=query,
                    retrieval_tool=result.tool_name,
                    metadata={
                        "relative_path": str(match.get("relative_path", "")),
                        "line_start": int(match.get("line_start", 1)),
                        "line_end": int(match.get("line_end", 1)),
                    },
                )
            )
    digest_payload = [
        {
            "source_id": item.source_id,
            "source_hash": item.source_hash,
            "excerpt": item.excerpt,
            "retrieval_query": item.retrieval_query,
            "retrieval_tool": item.retrieval_tool,
            "metadata": item.metadata,
        }
        for item in items
    ]
    encoded = json.dumps(
        digest_payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode("utf-8")
    return EvidenceBundle(
        items=items,
        source_count=len({item.source_id for item in items}),
        evidence_count=len(items),
        bundle_hash=hashlib.sha256(encoded).hexdigest(),
    )


def _sanitize_excerpt(value: str) -> str:
    return " ".join(value.replace("\x00", "").split())[:4096]


def _calculator_evidence_item(result: ToolResult) -> EvidenceItem | None:
    expression = result.output.get("expression")
    value = result.output.get("value")
    if not isinstance(expression, str) or not expression.strip():
        return None
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    payload = {
        "tool": result.tool_name,
        "expression": expression,
        "value": value,
    }
    encoded = json.dumps(
        payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode("utf-8")
    source_hash = hashlib.sha256(encoded).hexdigest()
    return EvidenceItem(
        source_id=f"tool:{result.tool_name}",
        source_hash=source_hash,
        source_type="tool_result",
        excerpt=f"{expression} = {value}",
        retrieval_query=expression,
        retrieval_tool=result.tool_name,
        metadata={
            "expression": expression,
            "value": value,
        },
    )
