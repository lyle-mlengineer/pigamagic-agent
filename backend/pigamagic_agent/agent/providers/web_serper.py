"""Serper `WebProvider`: Google's index as clean JSON.

Serper wraps Google search results behind `POST https://google.serper.dev/search`
with the key in `X-API-KEY`. It is here because it is the cheapest way to get
*Google-grade* venue research (a restaurant's own site, a wedding directory, a
local news piece) without a Google Cloud project, and because its response is
already structured — `organic[].{title,link,snippet}` — so nothing has to be
scraped.

Notes that come from the API rather than from guesswork:

- the endpoint is **POST with a JSON body**, not a query string, and `num` is
  capped at 100; asking for more returns a 400 rather than a bigger page;
- `organic` is absent (not empty) when Google returns nothing, so both shapes
  are handled;
- each search consumes one credit. The per-query price is declared so a host
  that sets a `BudgetLedger` limit sees the spend before it happens.

Page extraction and image download are shared with every other adapter
(`web_common.py`), so a Serper hit and a Wikipedia hit produce the same
`WebPage`.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import ClassVar

from pigamagic_agent.agent.providers.base import CostEstimate, SearchHit
from pigamagic_agent.agent.providers.web_common import (
    HttpWebProvider,
    missing_key_error,
    require_http_url,
    strip_tags,
)

SERPER_ENDPOINT = "https://google.serper.dev/search"

#: Serper bills per search credit; 50k credits cost about $50.
SERPER_USD_PER_QUERY = 0.001

#: Serper rejects `num` above 100 with a 400.
MAX_RESULTS = 100


@dataclass
class SerperWebProvider(HttpWebProvider):
    """Open-web search through Serper's Google endpoint."""

    name: ClassVar[str] = "serper"
    base_url: str = SERPER_ENDPOINT
    price_per_query_usd: float = SERPER_USD_PER_QUERY
    country: str = "us"
    language: str = "en"

    def require_credentials(self) -> None:
        if not self.api_key.strip():
            raise missing_key_error("serper", "AGENT_WEB_API_KEY")

    async def search(self, *, query: str, limit: int = 5) -> list[SearchHit]:
        self.require_credentials()
        if not query.strip():
            return []

        count = max(1, min(int(limit), MAX_RESULTS))
        payload = await self.request_json(
            "POST",
            self.base_url or SERPER_ENDPOINT,
            headers={
                "X-API-KEY": self.api_key,
                "Accept": "application/json",
                "Content-Type": "application/json",
                "User-Agent": self.user_agent,
            },
            json_body={
                "q": query,
                "num": count,
                "gl": self.country,
                "hl": self.language,
            },
            estimate=CostEstimate(usd=self.price_per_query_usd),
        )

        hits: list[SearchHit] = []
        for row in (payload or {}).get("organic") or []:
            if not isinstance(row, dict):
                continue
            url = require_http_url(str(row.get("link") or row.get("url") or ""))
            if not url:
                continue
            hits.append(
                SearchHit(
                    url=url,
                    title=strip_tags(str(row.get("title") or "")),
                    snippet=strip_tags(str(row.get("snippet") or row.get("description") or "")),
                )
            )
            if len(hits) >= count:
                break
        return hits


__all__ = [
    "MAX_RESULTS",
    "SERPER_ENDPOINT",
    "SERPER_USD_PER_QUERY",
    "SerperWebProvider",
]
