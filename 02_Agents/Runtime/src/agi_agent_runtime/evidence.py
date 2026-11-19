"""Host-owned construction of bounded evidence bundles from approved tool results."""

from __future__ import annotations

import hashlib
import json
import uuid

from agi_contracts import EvidenceBundle, EvidenceItem, ToolRequest, ToolResult


def build_evidence_bundle(
    requests: list[ToolRequest], results: list[ToolResult]
) -> EvidenceBundle:
    request_by_id = {request.request_id: request for request in requests}
    items: list[EvidenceItem] = []
    # Two searches often return the same passage; the model should read it once.
    seen_passages: set[tuple[str, int, int, str]] = set()
    for result in results:
        if not result.success:
            continue
        if result.tool_name == "calculator":
            item = _calculator_evidence_item(result)
            if item is not None:
                items.append(item)
            continue
        if result.tool_name == "read_only_file_reader":
            items.extend(_reader_evidence_items(result))
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
            line_start = int(match.get("line_start", 1))
            line_end = int(match.get("line_end", 1))
            passage = (str(match["source_id"]), line_start, line_end, excerpt)
            if passage in seen_passages:
                continue
            seen_passages.add(passage)
            title = match.get("document_title")
            items.append(
                EvidenceItem(
                    source_id=str(match["source_id"]),
                    source_hash=source_hash,
                    excerpt=excerpt,
                    retrieval_query=query,
                    retrieval_tool=result.tool_name,
                    metadata={
                        "relative_path": str(match.get("relative_path", "")),
                        "line_start": line_start,
                        "line_end": line_end,
                        **(
                            {"document_title": _sanitize_excerpt(title)[:200]}
                            if isinstance(title, str) and title.strip()
                            else {}
                        ),
                    },
                )
            )
    items = _with_stable_ids(items)
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


_EVIDENCE_NAMESPACE = uuid.UUID("6f2c1e9a-4b7d-5a3e-9c1f-2d8e7b4a6c05")


def _with_stable_ids(items: list[EvidenceItem]) -> list[EvidenceItem]:
    """Give each item an id derived from its content, and drop exact repeats.

    The bundle is rebuilt after every dispatch. With random ids, a search made after
    the answer was written (a follow-up or bridge search) gave every item a new id and
    left the answer's citations pointing at nothing. The same passage now keeps the
    same id for the whole run.
    """

    stable: list[EvidenceItem] = []
    seen: set[uuid.UUID] = set()
    for item in items:
        key = json.dumps(
            [item.source_id, item.source_hash, item.excerpt, item.metadata.get("line_start"), item.metadata.get("line_end")],
            ensure_ascii=False,
            default=str,
        )
        evidence_id = uuid.uuid5(_EVIDENCE_NAMESPACE, key)
        if evidence_id in seen:
            continue  # the same passage or calculation twice: read it once
        seen.add(evidence_id)
        stable.append(item.model_copy(update={"evidence_id": evidence_id}))
    return stable


def _sanitize_excerpt(value: str) -> str:
    return " ".join(value.replace("\x00", "").split())[:4096]


_MAX_READER_LINES = 64


def _reader_evidence_items(result: ToolResult) -> list[EvidenceItem]:
    """One citable item per non-empty line, so citations stay line-precise."""

    output = result.output
    source_hash = output.get("content_sha256")
    content = output.get("content")
    if not isinstance(source_hash, str) or len(source_hash) != 64 or not isinstance(content, str):
        return []
    try:
        first_line = int(output.get("line_start", 1))
    except (TypeError, ValueError):
        return []
    source_id = str(output.get("source_id", ""))
    relative_path = str(output.get("relative_path", ""))
    query = f"{relative_path}:{first_line}"
    items: list[EvidenceItem] = []
    title: str | None = None
    for offset, line in enumerate(content.splitlines()):
        excerpt = _sanitize_excerpt(line)
        if not excerpt:
            continue
        if excerpt.startswith("#"):
            title = excerpt.lstrip("#").strip()[:200] or None
        line_number = first_line + offset
        items.append(
            EvidenceItem(
                source_id=source_id,
                source_hash=source_hash,
                excerpt=excerpt,
                retrieval_query=query,
                retrieval_tool=result.tool_name,
                metadata={
                    "relative_path": relative_path,
                    "line_start": line_number,
                    "line_end": line_number,
                    **({"document_title": title} if title else {}),
                },
            )
        )
        if len(items) >= _MAX_READER_LINES:
            break
    return items


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
