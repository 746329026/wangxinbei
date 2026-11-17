"""M7 agent protocol: strict JSON between the model and the runtime.

The model may only emit one of two shapes per step::

    {"tool": "<name>", "args": {...}, "why": "<optional, <=200 chars>"}
    {"final": {...}, "evidence": [<step indices, optional>]}

No free-form prose is accepted. ``why``/``evidence`` are audit metadata only;
they never change the decision logic.
"""

from __future__ import annotations

import json
import re
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, ValidationError

__all__ = [
    "ProtocolError",
    "ToolCall",
    "extract_json_object",
    "parse_action",
    "strict_json_load",
]


class ProtocolError(ValueError):
    """Raised when the model output is not a valid M7 action."""


class ToolCall(BaseModel):
    model_config = ConfigDict(extra="forbid")

    tool: str
    args: dict[str, Any] = Field(default_factory=dict)
    why: str | None = Field(default=None, max_length=200)


def _repair_json_text(text: str) -> str:
    """Best-effort repair of the common Qwen3 JSON mistakes.

    - unclosed string at the end of a truncated output;
    - unbalanced ``{``/``[`` (missing closing braces);
    - trailing commas before a closing bracket.
    """

    stack: list[str] = []
    in_string = False
    escaped = False
    for char in text:
        if in_string:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == '"':
                in_string = False
            continue
        if char == '"':
            in_string = True
        elif char in "{[":
            stack.append(char)
        elif char in "}]":
            if stack:
                stack.pop()
    repaired = text
    if in_string:
        repaired += '"'
    repaired = re.sub(r",\s*([}\]])", r"\1", repaired)
    for opener in reversed(stack):
        repaired += "}" if opener == "{" else "]"
    return repaired


def extract_json_object(text: str) -> dict[str, Any]:
    """Extract one JSON object from a model response.

    Handles the known Qwen3 failure modes: an accidental ``<think>`` block,
    markdown fences, trailing prose around the object, and missing closing
    braces. Repairs are recorded nowhere on purpose; the raw output is logged by
    the runtime before parsing.
    """

    cleaned = (text or "").strip()
    if "<think>" in cleaned:
        if "</think>" in cleaned:
            cleaned = cleaned.rsplit("</think>", 1)[1]
        else:
            cleaned = cleaned.split("<think>", 1)[0]
        cleaned = cleaned.strip()

    fenced = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", cleaned, re.S | re.I)
    candidate = fenced.group(1) if fenced else cleaned
    start = candidate.find("{")
    if start < 0:
        raise ProtocolError(f"output does not contain a JSON object: {text!r}")
    body = candidate[start:]
    decoder = json.JSONDecoder()
    try:
        payload, _ = decoder.raw_decode(body)
    except json.JSONDecodeError as first_error:
        repaired = _repair_json_text(body)
        try:
            payload, _ = decoder.raw_decode(repaired)
        except json.JSONDecodeError as exc:
            raise ProtocolError(
                f"invalid JSON ({exc}); repair also failed ({first_error}): {text!r}"
            ) from exc
    if not isinstance(payload, dict):
        raise ProtocolError(f"JSON payload must be an object: {payload!r}")
    return payload


def strict_json_load(text: str) -> dict[str, Any]:
    """Parse the raw model output as-is; no extraction, no repair.

    This is the only path used in strict mode. It guarantees that the model
    itself emitted one syntactically valid JSON object (surrounding whitespace
    only), rather than the framework reconstructing JSON from prose.
    """

    stripped = (text or "").strip()
    try:
        payload = json.loads(stripped)
    except json.JSONDecodeError as exc:
        raise ProtocolError(f"strict JSON parse failed: {exc}; raw={text!r}") from exc
    if not isinstance(payload, dict):
        raise ProtocolError(f"JSON payload must be an object: {payload!r}")
    return payload


def _validate_action(payload: dict[str, Any]) -> tuple[str, dict[str, Any]]:
    has_tool = "tool" in payload
    has_final = "final" in payload
    if has_tool == has_final:
        raise ProtocolError("exactly one of 'tool' or 'final' must be present")

    if has_final:
        extra = set(payload) - {"final", "evidence"}
        if extra:
            raise ProtocolError(f"unexpected keys in final action: {sorted(extra)}")
        final = payload["final"]
        if not isinstance(final, dict):
            raise ProtocolError("'final' must be a JSON object")
        evidence = payload.get("evidence", [])
        if evidence is None:
            evidence = []
        if not isinstance(evidence, list) or not all(isinstance(i, int) for i in evidence):
            raise ProtocolError("'evidence' must be a list of step indices")
        return "final", {"answer": final, "evidence": evidence[:10]}

    extra = set(payload) - {"tool", "args", "why"}
    if extra:
        raise ProtocolError(f"unexpected keys in tool action: {sorted(extra)}")
    try:
        call = ToolCall(
            tool=payload["tool"],
            args=payload.get("args") or {},
            why=payload.get("why"),
        )
    except ValidationError as exc:
        raise ProtocolError(f"invalid tool action: {exc}") from exc
    return "call", call.model_dump()


def parse_action(text: str, *, strict: bool = False) -> tuple[str, dict[str, Any]]:
    """Return ``("call", {...})`` or ``("final", {...})``.

    ``strict=True`` requires the raw output to be valid JSON by itself.
    ``strict=False`` keeps the historical lenient extraction/repair path (kept
    for backward compatibility and for measuring how often repair is needed).
    """

    payload = strict_json_load(text) if strict else extract_json_object(text)
    return _validate_action(payload)
