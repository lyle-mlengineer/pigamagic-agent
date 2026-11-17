"""Wikipedia adapter: keyless, real venue facts for notable places.

Why ship an encyclopedia adapter in a "web research" step: many of the venues
people actually book — heritage hotels, churches, gardens, museums, campuses —
have an article that describes their architecture and grounds in exactly the
vocabulary the planner needs ("atrium", "colonnade", "stained glass",
"terraces", "dance floor"). It needs no key, no account and no rate-limit
negotiation, so the real evidence path is demonstrable before anyone has a
search-vendor subscription.

What it is not: a general web search. A restaurant with no article returns no
hits, and the research stage then says so honestly (low confidence, no invented
signature details) rather than guessing.

Page extraction and image download come from `web_common.py`, so an article and
a Brave hit produce the same `WebPage` shape.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import ClassVar
from urllib.parse import quote, unquote

from pigamagic_agent.agent.providers.base import ProviderError, SearchHit, WebPage
from pigamagic_agent.agent.providers.web_common import (
    DEAD_PAGE_STATUS,
    MAX_TEXT_CHARS,
    HttpWebProvider,
    now_iso,
    require_http_url,
    strip_tags,
)

#: Read the MediaWiki API documentation before changing these parameters.
WIKIPEDIA_API = "https://en.wikipedia.org/w/api.php"

MAX_RESULTS = 20


@dataclass
class WikipediaWebProvider(HttpWebProvider):
    """Search a MediaWiki wiki and read its articles as venue evidence."""

    name: ClassVar[str] = "wikipedia"
    base_url: str = WIKIPEDIA_API

    def require_credentials(self) -> None:
        """Public API; no key. The User-Agent is the politeness contract."""
        return None

    def article_base(self) -> str:
        base = (self.base_url or WIKIPEDIA_API).rstrip("/")
        suffix = "/w/api.php"
        if base.endswith(suffix):
            return base[: -len(suffix)]
        return "https://en.wikipedia.org"

    def title_from_url(self, url: str) -> str:
        """Extract the article title from a `/wiki/...` URL."""
        marker = "/wiki/"
        if marker not in url:
            return ""
        slug = url.split(marker, 1)[1].split("#", 1)[0].split("?", 1)[0]
        return unquote(slug).replace("_", " ").strip()

    async def fetch_page(self, *, url: str) -> WebPage:
        """Read article prose through the API, not through page chrome.

        A rendered Wikipedia page is mostly navigation, edit links and a
        bibliography; extracting it as HTML gives the planner menu labels as
        "architecture". `prop=extracts&explaintext` returns the article body
        alone, and `pageimages` returns a usable lead photo in the same call.
        """
        target = require_http_url(url)
        if not target or not target.startswith(self.article_base()):
            return await super().fetch_page(url=url)

        title = self.title_from_url(target)
        if not title:
            return await super().fetch_page(url=url)

        try:
            payload = await self.request_json(
                "GET",
                self.base_url or WIKIPEDIA_API,
                headers={"Accept": "application/json", "User-Agent": self.user_agent},
                params={
                    "action": "query",
                    "format": "json",
                    "formatversion": 2,
                    "prop": "extracts|pageimages",
                    "explaintext": 1,
                    "redirects": 1,
                    "piprop": "original|thumbnail",
                    "pithumbsize": 1280,
                    "titles": title,
                },
            )
        except ProviderError as exc:
            if exc.status in DEAD_PAGE_STATUS:
                return WebPage(url=target, title="", text="", fetched_at=now_iso())
            raise

        pages = ((payload or {}).get("query") or {}).get("pages") or []
        page = pages[0] if pages and isinstance(pages[0], dict) else {}
        text = str(page.get("extract") or "")
        if page.get("missing") or not text:
            return WebPage(url=target, title="", text="", fetched_at=now_iso())

        images: list[str] = []
        for key in ("original", "thumbnail"):
            source = page.get(key)
            url_value = source.get("source") if isinstance(source, dict) else ""
            if url_value and url_value not in images:
                images.append(str(url_value))

        return WebPage(
            url=target,
            title=str(page.get("title") or title),
            text=text[:MAX_TEXT_CHARS],
            fetched_at=now_iso(),
            images=tuple(images),
        )

    async def search(self, *, query: str, limit: int = 5) -> list[SearchHit]:
        if not query.strip():
            return []
        count = max(1, min(int(limit), MAX_RESULTS))
        payload = await self.request_json(
            "GET",
            self.base_url or WIKIPEDIA_API,
            headers={"Accept": "application/json", "User-Agent": self.user_agent},
            params={
                "action": "query",
                "format": "json",
                "formatversion": 2,
                "list": "search",
                "srsearch": query,
                "srlimit": count,
                # Articles only: a File: or Category: hit is not evidence.
                "srnamespace": 0,
                "srprop": "snippet",
            },
        )

        rows = ((payload or {}).get("query") or {}).get("search") or []
        hits: list[SearchHit] = []
        for row in rows:
            if not isinstance(row, dict):
                continue
            title = str(row.get("title") or "").strip()
            if not title:
                continue
            slug = quote(title.replace(" ", "_"), safe="")
            hits.append(
                SearchHit(
                    url=f"{self.article_base()}/wiki/{slug}",
                    title=title,
                    snippet=strip_tags(str(row.get("snippet") or "")),
                )
            )
            if len(hits) >= count:
                break
        return hits


__all__ = ["MAX_RESULTS", "WIKIPEDIA_API", "WikipediaWebProvider"]
