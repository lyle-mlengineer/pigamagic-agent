"""Groq `LLMProvider`: the planner, on `openai/gpt-oss-120b`.

Why Groq for the planning half: it serves open-weight GPT-OSS 120B on its own
inference hardware with a free tier and an OpenAI-compatible API, so the model
that writes the shot list costs nothing and needs no vendor SDK.

Three details are not guessable and are handled explicitly:

1. **Structured output is a request field, not a prompt style.** Groq accepts
   `response_format={"type": "json_schema", "json_schema": {...}}` for GPT-OSS,
   which is strictly better than hoping the model emits JSON. Some schemas are
   not accepted, so a 400 falls back to `json_object` mode once and says so in
   the log — silently degrading to "please output JSON" is how a planner starts
   returning prose.
2. **`max_tokens` is deprecated in favour of `max_completion_tokens`,** and
   GPT-OSS is a *reasoning* model: with a small budget it spends everything
   thinking and returns an empty `content`. `chat_json.message_text` turns that
   into a specific, actionable error.
3. **Reasoning effort is capped at `low`.** Planning a shot list does not need
   deep deliberation, and low effort is what keeps a free-tier run inside its
   rate limit.

Free tier means zero spend: `estimate_cost` reports 0 USD, so the image budget
ledger is never involved.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, ClassVar, Mapping

from pigamagic_agent.agent.providers.base import (
    CostEstimate,
    LLMResult,
    ProviderError,
    ProviderNotConfigured,
)
from pigamagic_agent.agent.providers.chat_json import (
    SCHEMA_NAME,
    clamp_tokens,
    parse_structured,
)
from pigamagic_agent.agent.providers.http_base import HttpProvider

logger = logging.getLogger(__name__)

GROQ_CHAT_URL = "https://api.groq.com/openai/v1/chat/completions"

#: The model requested in the brief: open-weight GPT-OSS 120B on Groq.
DEFAULT_MODEL = "openai/gpt-oss-120b"

#: Reasoning effort for planning. `low` is enough for a shot list and keeps the
#: free tier's tokens-per-minute budget usable.
DEFAULT_REASONING_EFFORT = "low"


@dataclass
class GroqLLMProvider(HttpProvider):
    """Structured JSON planning through Groq's OpenAI-compatible endpoint."""

    name: ClassVar[str] = "groq"
    base_url: str = GROQ_CHAT_URL
    model: str = DEFAULT_MODEL
    reasoning_effort: str = DEFAULT_REASONING_EFFORT

    def require_credentials(self) -> None:
        if not self.api_key.strip():
            raise ProviderNotConfigured(
                "groq needs an API key. Set AGENT_LLM_API_KEY (free at "
                "console.groq.com), or use the fixture provider."
            )

    async def complete_json(
        self,
        *,
        system: str,
        user: str,
        schema: dict[str, Any],
        model: str | None = None,
        temperature: float = 0.0,
        max_tokens: int | None = None,
    ) -> LLMResult:
        chosen = self.resolve_model(model)
        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ]
        tokens = clamp_tokens(max_tokens)

        try:
            payload = await self._call(messages, chosen, schema, temperature, tokens, strict=True)
        except ProviderError as exc:
            # A schema Groq cannot compile is a 400, not an outage. Degrade to
            # JSON mode once, loudly, and let our own check do the work.
            if exc.status != 400 or not schema:
                raise
            logger.info(
                "groq rejected the json_schema request (%s); retrying in json_object mode", exc
            )
            payload = await self._call(messages, chosen, schema, temperature, tokens, strict=False)

        data, raw_text = parse_structured(payload, schema, provider=self.name)
        return LLMResult(data=data, raw_text=raw_text, provider=self.name, model=chosen)

    async def estimate_cost(self, *, user: str = "", model: str | None = None) -> CostEstimate:
        """Groq's free tier costs nothing; the budget ledger stays out of it."""
        return CostEstimate(credits=0, usd=0.0)

    # -- internals ---------------------------------------------------------

    async def _call(
        self,
        messages: list[dict[str, Any]],
        model: str,
        schema: Mapping[str, Any],
        temperature: float,
        max_tokens: int,
        *,
        strict: bool,
    ) -> Mapping[str, Any]:
        response_format: dict[str, Any] = {"type": "json_object"}
        if strict:
            response_format = {
                "type": "json_schema",
                "json_schema": {
                    "name": SCHEMA_NAME,
                    "schema": dict(schema),
                    # Strict mode is what makes the schema binding rather than
                    # advisory. It requires `additionalProperties: false` and a
                    # complete `required` list, which is exactly why the 400
                    # fallback above exists.
                    "strict": True,
                },
            }

        return await self.request_json(
            "POST",
            self.base_url or GROQ_CHAT_URL,
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            },
            json_body={
                "model": model,
                "messages": messages,
                "temperature": temperature,
                # GPT-OSS is a reasoning model; low effort keeps it inside the
                # free tier's token budget without hurting a shot list.
                "reasoning_effort": self.reasoning_effort,
                "max_completion_tokens": max_tokens,
                "response_format": response_format,
            },
            estimate=CostEstimate.zero(),
        )


__all__ = ["DEFAULT_MODEL", "DEFAULT_REASONING_EFFORT", "GROQ_CHAT_URL", "GroqLLMProvider"]
