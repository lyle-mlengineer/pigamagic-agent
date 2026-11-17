"""DuckDuckGo `WebProvider`: keyless open-web search with no account.

`https://lite.duckduckgo.com/lite/` is the plainest HTML the service serves —
no JavaScript, no API key, no signup — which makes it the only open-web search
that works out of the box on a machine with nothing configured. It is slower and
less fresh than a paid index, and it rate-limits aggressively, so it is the
*convenience* provider: `wikipedia` is better evidence for notable venues and
`serper`/`brave` are better for everything else.

Two behaviours that matter:

- **The redirect wrapper must be unwrapped.** Result links are
  `//duckduckgo.com/l/?uddg=<percent-encoded target>`. Passing that to
  `fetch_page` downloads a redirect stub instead of the venue's site, so every
  hit is decoded back to its real URL and ad/sponsored links are dropped.
- **A challenge page is reported as such.** DuckDuckGo answers datacenter IPs
  with an "anomaly" page and no results. That is not "this venue has no web
  presence", and treating it as such would produce a confidently empty venue
  profile; it raises instead, naming the alternatives.
"""

from __future__ import annotations

from dataclasses import dataclass
from html.parser import HTMLParser
from typing import ClassVar
from urllib.parse import parse_qs, unquote, urlparse

from pigamagic_agent.agent.providers.base import ProviderError, SearchHit
from pigamagic_agent.agent.providers.http_base import is_retryable_exception
from pigamagic_agent.agent.providers.web_common import (
    HttpWebProvider,
    require_http_url,
    strip_tags,
)

LITE_ENDPOINT = "https://lite.duckduckgo.com/lite/"

MAX_RESULTS = 20

#: Values of `uddg` that point back at DuckDuckGo itself (ads, "next page").
_DDG_HOSTS = frozenset({"duckduckgo.com", "html.duckduckgo.com", "lite.duckduckgo.com"})


class _LiteParser(HTMLParser):
    """Collect result links and snippets from the DuckDuckGo Lite page.

    Structural, not regex-based: the page is a table of `<a class="result-link">`
    followed by `<td class="result-snippet">`, and pairing them by document
    order is what keeps titles attached to the right snippets.
    """

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.links: list[tuple[str, str]] = []
        self.snippets: list[str] = []
        self._href = ""
        self._in_link = False
        self._in_snippet = False
        self._buffer: list[str] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        attributes = {name.lower(): (value or "") for name, value in attrs}
        classes = attributes.get("class", "").split()
        if tag == "a" and "result-link" in classes:
            self._href = attributes.get("href", "")
            self._in_link = True
            self._buffer = []
        elif tag == "td" and "result-snippet" in classes:
            self._in_snippet = True
            self._buffer = []

    def handle_endtag(self, tag: str) -> None:
        if tag == "a" and self._in_link:
            self.links.append((self._href, strip_tags("".join(self._buffer))))
            self._in_link = False
            self._buffer = []
        elif tag == "td" and self._in_snippet:
            self.snippets.append(strip_tags("".join(self._buffer)))
            self._in_snippet = False
            self._buffer = []

    def handle_data(self, data: str) -> None:
        if self._in_link or self._in_snippet:
            self._buffer.append(data)


def unwrap_redirect(href: str) -> str:
    """Turn a DuckDuckGo result href into the URL it actually points at."""
    raw = (href or "").strip()
    if not raw:
        return ""
    if raw.startswith("//"):
        raw = "https:" + raw
    parsed = urlparse(raw)
    if parsed.netloc.endswith("duckduckgo.com"):
        target = parse_qs(parsed.query).get("uddg", [""])[0]
        if not target:
            return ""  # an ad or an internal link, not a result
        return require_http_url(unquote(target))
    return require_http_url(raw)


@dataclass
class DuckDuckGoWebProvider(HttpWebProvider):
    """Open-web search through DuckDuckGo's keyless HTML endpoint."""

    name: ClassVar[str] = "duckduckgo"
    base_url: str = LITE_ENDPOINT

    def require_credentials(self) -> None:
        """No key, no account, no signup."""
        return None

    async def search(self, *, query: str, limit: int = 5) -> list[SearchHit]:
        if not query.strip():
            return []

        count = max(1, min(int(limit), MAX_RESULTS))
        self.guard_offline()
        url = self.base_url or LITE_ENDPOINT
        try:
            response = await self._http().get(
                url,
                params={"q": query},
                headers=self._headers("text/html"),
                follow_redirects=True,
            )
        except Exception as exc:  # noqa: BLE001 - narrowed by the helper
            raise ProviderError(
                f"{self.name} could not search: {exc}",
                provider=self.name,
                retryable=is_retryable_exception(exc),
            ) from exc

        body = response.text or ""
        if response.status_code >= 400:
            raise ProviderError(
                f"{self.name} returned {response.status_code}",
                provider=self.name,
                retryable=response.status_code >= 500 or response.status_code == 429,
                status=response.status_code,
            )

        parser = _LiteParser()
        try:
            parser.feed(body)
            parser.close()
        except Exception:  # noqa: BLE001 - malformed markup is normal on the web
            pass

        hits: list[SearchHit] = []
        for index, (href, title) in enumerate(parser.links):
            target = unwrap_redirect(href)
            if not target or urlparse(target).netloc.endswith(tuple(_DDG_HOSTS)):
                continue
            snippet = parser.snippets[index] if index < len(parser.snippets) else ""
            hits.append(SearchHit(url=target, title=title, snippet=snippet))
            if len(hits) >= count:
                break

        if not hits and _is_challenge(body):
            raise ProviderError(
                f"{self.name} served its bot-challenge page instead of results. "
                "DuckDuckGo rate-limits datacenter IPs; retry later, or use the "
                "searxng/serper/wikipedia providers.",
                provider=self.name,
                retryable=True,
            )
        return hits


def _is_challenge(body: str) -> bool:
    lowered = (body or "").casefold()
    return "anomaly" in lowered or "bots use duckduckgo" in lowered


__all__ = ["LITE_ENDPOINT", "MAX_RESULTS", "DuckDuckGoWebProvider", "unwrap_redirect"]
