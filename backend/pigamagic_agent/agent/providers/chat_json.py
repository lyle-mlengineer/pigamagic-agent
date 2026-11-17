"""Shared plumbing for "a chat completion that must return one JSON object".

Two real providers need exactly this: the Groq planner (`llm_groq.py`) and the
NVIDIA vision model (`vision_nvidia.py`). Both speak the OpenAI chat-completions
shape and both are useless to this pipeline unless the reply is structured, so
the parsing, the failure typing and the schema check live here once instead of
being re-invented per vendor.

The rule this module exists to enforce: **a model's chat text is never trusted
blindly**. It is parsed, type-checked against the caller's schema, and rejected
with a retryable `ProviderError` when it is not an object at all. The pipeline
above then falls back (a procedural plan, or an unanalysed reference) instead of
crashing on a stray sentence of prose.
"""

from __future__ import annotations

import json
import re
from typing import Any, Mapping

from pigamagic_agent.agent.providers.base import ProviderError

#: Room for a shot list plus its reasoning. Callers may override per request.
DEFAULT_MAX_TOKENS = 4096

#: `max_tokens` ceiling that most hosted chat endpoints accept.
MAX_TOKENS_CEILING = 8192

_FENCE = re.compile(r"^\s*```(?:json|JSON)?\s*|\s*```\s*$")

_JSON_TYPES: dict[str, tuple[type, ...]] = {
    "object": (dict,),
    "array": (list,),
    "string": (str,),
    "number": (int, float),
    "integer": (int,),
    "boolean": (bool,),
    "null": (type(None),),
}

#: Pin a plausible name so a vendor that requires one never rejects the request.
SCHEMA_NAME = "plusone_result"


def clamp_tokens(max_tokens: int | None, *, default: int = DEFAULT_MAX_TOKENS) -> int:
    """Keep a token budget inside what every hosted endpoint accepts."""
    value = default if max_tokens is None else int(max_tokens)
    return max(1, min(value, MAX_TOKENS_CEILING))


def message_text(payload: Mapping[str, Any] | None, *, provider: str) -> str:
    """Pull the assistant's text out of an OpenAI-shaped chat completion.

    Some reasoning models put their chain of thought in a sibling field and leave
    `content` empty when the token budget runs out mid-thought. That is a real
    condition with a specific fix, so it gets its own message rather than
    "empty response".
    """
    choices = (payload or {}).get("choices") or []
    if not choices or not isinstance(choices[0], Mapping):
        raise ProviderError(
            f"{provider} returned no choices; the model may have refused the request.",
            provider=provider,
            retryable=True,
        )

    message = choices[0].get("message") or {}
    content = message.get("content")

    if isinstance(content, list):
        # Some gateways return content as typed parts rather than a string.
        content = "".join(
            str(part.get("text", "")) for part in content if isinstance(part, Mapping)
        )

    text = (content or "").strip() if isinstance(content, str) else ""
    if text:
        return text

    reasoning = str(
        message.get("reasoning_content") or message.get("reasoning") or ""
    ).strip()
    if reasoning:
        raise ProviderError(
            f"{provider} spent its whole token budget on reasoning and returned no "
            "answer. Raise the token budget or lower the reasoning effort.",
            provider=provider,
            retryable=False,
        )
    raise ProviderError(
        f"{provider} returned an empty message.", provider=provider, retryable=True
    )


def parse_json_object(text: str, *, provider: str) -> dict[str, Any]:
    """Parse the single JSON object in a model reply.

    Markdown fences and a sentence of preamble are stripped, because instruct
    models add them constantly; anything else that is not an object is an error.
    """
    candidate = _FENCE.sub("", text or "").strip()
    try:
        parsed = json.loads(candidate)
    except json.JSONDecodeError:
        start, end = candidate.find("{"), candidate.rfind("}")
        if start == -1 or end <= start:
            raise ProviderError(
                f"{provider} did not return JSON (got {candidate[:80]!r}).",
                provider=provider,
                retryable=True,
            ) from None
        try:
            parsed = json.loads(candidate[start : end + 1])
        except json.JSONDecodeError as exc:
            raise ProviderError(
                f"{provider} returned malformed JSON: {exc}.",
                provider=provider,
                retryable=True,
            ) from exc

    if not isinstance(parsed, dict):
        raise ProviderError(
            f"{provider} returned a {type(parsed).__name__}, not a JSON object.",
            provider=provider,
            retryable=True,
        )
    return parsed


def _matches_type(value: Any, expected: str) -> bool:
    types = _JSON_TYPES.get(expected)
    if types is None:
        return True
    # `bool` is a subclass of `int` in Python, so an integer field must reject it.
    if expected in {"number", "integer"} and isinstance(value, bool):
        return False
    return isinstance(value, types)


def apply_schema(data: dict[str, Any], schema: Mapping[str, Any] | None, *, provider: str) -> dict[str, Any]:
    """Shallow schema check: required keys present, declared types correct.

    Deliberately shallow — the caller validates its own models afterwards, and a
    second full JSON-Schema implementation here would be a second place to be
    wrong. Two things it does do:

    - a missing required key is a retryable failure, not a silent `None` later;
    - keys the schema does not declare are dropped, because a model adding
      `"notes": "..."` to a plan should not fail a whole shoot.
    """
    if not schema:
        return data

    missing = [key for key in (schema.get("required") or []) if key not in data]
    if missing:
        raise ProviderError(
            f"{provider} omitted required field(s): {', '.join(map(str, missing))}.",
            provider=provider,
            retryable=True,
        )

    properties = schema.get("properties") or {}
    if not properties:
        return data

    checked: dict[str, Any] = {}
    for key, spec in properties.items():
        if key not in data:
            continue
        expected = str((spec or {}).get("type") or "")
        if expected and not _matches_type(data[key], expected):
            raise ProviderError(
                f"{provider} returned {type(data[key]).__name__} for {key!r}, "
                f"expected {expected}.",
                provider=provider,
                retryable=True,
            )
        checked[key] = data[key]
    return checked


def parse_structured(
    payload: Mapping[str, Any] | None, schema: Mapping[str, Any] | None, *, provider: str
) -> tuple[dict[str, Any], str]:
    """Text -> JSON object -> schema-checked object, plus the raw text for audit."""
    text = message_text(payload, provider=provider)
    return apply_schema(parse_json_object(text, provider=provider), schema, provider=provider), text


__all__ = [
    "DEFAULT_MAX_TOKENS",
    "MAX_TOKENS_CEILING",
    "SCHEMA_NAME",
    "apply_schema",
    "clamp_tokens",
    "message_text",
    "parse_json_object",
    "parse_structured",
]
