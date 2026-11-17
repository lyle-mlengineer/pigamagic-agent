"""Brave Search adapter: real venue research over the open web.

`GET https://api.search.brave.com/res/v1/web/search` with the key in
`X-Subscription-Token`. Brave is the default *paid-tier-friendly* option here
because it returns clean JSON, allows a free monthly quota, and has no per-query
SDK to pull in.

Only `search` is vendor-specific: page extraction and image download are shared
in `web_common.py`, so a Brave hit and a Wikipedia hit produce identical
`WebPage`/`WebImage` values downstream.

Budget: the research stage caps the number of searches
(`AGENT_MAX_SEARCH_CALLS`); this adapter reports the per-query price so a host
that sets a `BudgetLedger` limit still sees the spend.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import ClassVar

from pigamagic_agent.agent.providers.base import CostEstimate, ProviderNotConfigured, SearchHit
from pigamagic_agent.agent.providers.web_common import (
    HttpWebProvider,
    missing_key_error,
    require_http_url,
    strip_tags,
)

BRAVE_ENDPOINT = "https://api.search.brave.com/res/v1/web/search"

#: Brave list price is $5 per 1000 queries; the first ~2000/month are free.
BRAVE_USD_PER_QUERY = 0.005

#: Brave caps `count` at 20 per request.
MAX_RESULTS = 20


@dataclass
class BraveWebProvider(HttpWebProvider):
    """Open-web search via the Brave Search API."""

    name: ClassVar[str] = "brave"
    base_url: str = BRAVE_ENDPOINT
    price_per_query_usd: float = BRAVE_USD_PER_QUERY

    def require_credentials(self) -> None:
        if not self.api_key.strip():
            raise missing_key_error("brave", "AGENT_WEB_API_KEY")

    async def search(self, *, query: str, limit: int = 5) -> list[SearchHit]:
        self.require_credentials()
        if not query.strip():
            return []

        count = max(1, min(int(limit), MAX_RESULTS))
        payload = await self.request_json(
            "GET",
            self.base_url,
            headers={
                "X-Subscription-Token": self.api_key,
                "Accept": "application/json",
                "User-Agent": self.user_agent,
            },
            params={
                "q": query,
                "count": count,
                "safesearch": "strict",
                "text_decorations": 0,
                "spellcheck": 0,
            },
            estimate=CostEstimate(usd=self.price_per_query_usd),
        )

        rows = ((payload or {}).get("web") or {}).get("results") or []
        hits: list[SearchHit] = []
        for row in rows:
            if not isinstance(row, dict):
                continue
            url = require_http_url(str(row.get("url") or ""))
            if not url:
                continue
            hits.append(
                SearchHit(
                    url=url,
                    title=strip_tags(str(row.get("title") or "")),
                    snippet=strip_tags(str(row.get("description") or row.get("snippet") or "")),
                )
            )
            if len(hits) >= count:
                break
        return hits


__all__ = ["BRAVE_ENDPOINT", "BRAVE_USD_PER_QUERY", "MAX_RESULTS", "BraveWebProvider"]
