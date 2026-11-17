"""SearXNG adapter: keyless, self-hostable web search.

A SearXNG instance exposes `GET /search?q=…&format=json`. That makes it the only
option here that needs **no account and no key** while still searching the open
web, which is why it is the recommended free path for local development.

Two setup facts that cause almost every failure:

1. `format=json` must be enabled on the instance
   (`search.formats: [html, json]` in `settings.yml`). Without it the instance
   answers with an HTML page and this adapter raises a message that says so,
   rather than "non-JSON content".
2. Public instances rate-limit and block bots. Point `AGENT_WEB_BASE_URL` at
   your own instance (Docker: `searxng/searxng`) for repeatable research.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import ClassVar

from pigamagic_agent.agent.providers.base import ProviderError, ProviderNotConfigured, SearchHit
from pigamagic_agent.agent.providers.web_common import HttpWebProvider, require_http_url, strip_tags

#: SearXNG has no fixed public endpoint; the operator supplies one.
DEFAULT_LANGUAGE = "en"


@dataclass
class SearxngWebProvider(HttpWebProvider):
    """Open-web search through a SearXNG instance's JSON API."""

    name: ClassVar[str] = "searxng"
    language: str = DEFAULT_LANGUAGE

    def require_credentials(self) -> None:
        """SearXNG needs a URL, not a key."""
        return None

    def _base(self) -> str:
        base = (self.base_url or getattr(self.settings, "web_base_url", "") or "").strip()
        if not base:
            raise ProviderNotConfigured(
                "searxng needs an instance URL. Set AGENT_WEB_BASE_URL to your "
                "instance, e.g. http://localhost:8080, or choose the wikipedia "
                "provider for a keyless public API."
            )
        return base.rstrip("/")

    async def search(self, *, query: str, limit: int = 5) -> list[SearchHit]:
        if not query.strip():
            return []
        count = max(1, min(int(limit), 20))
        try:
            payload = await self.request_json(
                "GET",
                f"{self._base()}/search",
                headers={"Accept": "application/json", "User-Agent": self.user_agent},
                params={
                    "q": query,
                    "format": "json",
                    "categories": "general",
                    "language": self.language,
                    "safesearch": 1,
                    "pageno": 1,
                },
            )
        except ProviderError as exc:
            if "non-JSON" in str(exc):
                raise ProviderNotConfigured(
                    "this SearXNG instance returned HTML, so its JSON API is "
                    "disabled. Add `json` to `search.formats` in the instance's "
                    "settings.yml, or use AGENT_IMAGE_PROVIDER-style fixture data."
                ) from exc
            raise

        rows = (payload or {}).get("results") or []
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
                    snippet=strip_tags(str(row.get("content") or row.get("snippet") or "")),
                )
            )
            if len(hits) >= count:
                break
        return hits


__all__ = ["DEFAULT_LANGUAGE", "SearxngWebProvider"]
