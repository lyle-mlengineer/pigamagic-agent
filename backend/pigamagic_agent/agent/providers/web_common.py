"""Shared scaffolding for real web-research adapters (design.md §6.1, Step 15).

Every real `WebProvider` has the same two jobs after the search call: turn a
page's HTML into clean text plus absolute image URLs, and download an image
candidate. Only `search()` is vendor-specific, so it is the only method an
adapter implements here.

Why this module exists rather than three copies:

- one HTML extractor means one place where `<script>` content, lazy-loaded
  images and relative URLs are handled correctly;
- one place owns the offline guard, the dead-page policy and the size caps, so
  a broken link cannot take down a whole research run;
- the extraction rules are testable without any network at all.

**Dead pages are data, not errors.** Links rot and sites block hotlinking. A
404 page or a 403 image returns an empty value, which the research stage already
knows how to ignore; a 429 or a 5xx is raised, because those carry information
about our budget and credentials.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import datetime, timezone
from html.parser import HTMLParser
from typing import ClassVar
from urllib.parse import urljoin, urlparse

from pigamagic_agent.agent.providers.base import (
    CostEstimate,
    ProviderError,
    ProviderNotConfigured,
    SearchHit,
    WebImage,
    WebPage,
)
from pigamagic_agent.agent.providers.http_base import (
    RETRYABLE_STATUS,
    BudgetLedger,
    HttpProvider,
    is_retryable_exception,
    sniff_image_type,
)

#: Wikimedia asks for a descriptive agent; a generic browser string gets blocked.
DEFAULT_USER_AGENT = (
    "PlusOneAgent/1.0 (venue research; +https://github.com/lyle-mlengineer/pigamagic-agent)"
)

#: Pages bigger than this are almost never venue evidence, and decoding them is
#: how a research run turns into a memory spike.
MAX_PAGE_BYTES = 2_000_000

#: One venue photo is a few hundred KB. Anything above this is a wallpaper.
MAX_IMAGE_BYTES = 8_000_000

#: Extracted text is capped so a link farm cannot flood the prompt budget.
MAX_TEXT_CHARS = 60_000

#: Statuses that mean "this page/image is gone" rather than "we are being
#: throttled". Research treats them as empty evidence.
DEAD_PAGE_STATUS = frozenset({403, 404, 410, 451})

_IMAGE_TYPES = frozenset({"image/png", "image/jpeg", "image/webp"})
_SKIP_TAGS = frozenset({"script", "style", "noscript", "template", "svg", "iframe", "head"})
_BLOCK_TAGS = frozenset(
    {
        "address",
        "article",
        "aside",
        "blockquote",
        "br",
        "dd",
        "div",
        "dl",
        "dt",
        "figcaption",
        "figure",
        "footer",
        "h1",
        "h2",
        "h3",
        "h4",
        "h5",
        "h6",
        "header",
        "hr",
        "li",
        "main",
        "nav",
        "ol",
        "p",
        "pre",
        "section",
        "table",
        "td",
        "th",
        "tr",
        "ul",
    }
)
_SKIP_IMAGE_SUFFIXES = (".svg", ".gif", ".ico")
_WHITESPACE = re.compile(r"[ \t\r\f\v\u00a0]+")
_TAG = re.compile(r"<[^>]+>")


# --------------------------------------------------------------------------
# Small helpers
# --------------------------------------------------------------------------


def now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def strip_tags(text: str) -> str:
    """Remove markup from a snippet and normalise its whitespace."""
    cleaned = _TAG.sub(" ", text or "")
    cleaned = _WHITESPACE.sub(" ", cleaned)
    return " ".join(cleaned.split()).strip()


def require_http_url(url: str) -> str:
    """Return `url` if it is a usable http(s) URL, else an empty string.

    Search results occasionally contain `javascript:` or `mailto:` entries; a
    provider that forwards them to `httpx` produces a confusing transport error
    instead of an honest "not a page".
    """
    candidate = (url or "").strip()
    if not candidate:
        return ""
    parsed = urlparse(candidate)
    if parsed.scheme not in {"http", "https"} or not parsed.netloc:
        return ""
    return candidate


# --------------------------------------------------------------------------
# HTML extraction
# --------------------------------------------------------------------------


@dataclass(slots=True)
class ExtractedPage:
    title: str = ""
    text: str = ""
    images: tuple[str, ...] = ()


class _PageExtractor(HTMLParser):
    """Collect title, visible text and image URLs from one page.

    Deliberately structural, not heuristic: block elements become line breaks,
    skipped elements are dropped whole, and `<img>` attributes (including
    lazy-loading ones) become absolute URLs.
    """

    def __init__(self, base_url: str) -> None:
        super().__init__(convert_charrefs=True)
        self.base_url = base_url
        self.title_parts: list[str] = []
        self.parts: list[str] = []
        self.images: list[str] = []
        self._skip_depth = 0
        self._in_title = False

    # -- HTMLParser hooks -------------------------------------------------

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag in _SKIP_TAGS:
            self._skip_depth += 1
            return
        if tag == "title":
            self._in_title = True
            return
        if tag in _BLOCK_TAGS:
            self.parts.append("\n")
        if tag == "img":
            self._collect_image(attrs)

    def handle_startendtag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        # `<img />` does not get an end tag, so it must be handled here too.
        if tag == "img":
            self._collect_image(attrs)
            return
        self.handle_starttag(tag, attrs)
        self.handle_endtag(tag)

    def handle_endtag(self, tag: str) -> None:
        if tag in _SKIP_TAGS:
            self._skip_depth = max(0, self._skip_depth - 1)
            return
        if tag == "title":
            self._in_title = False
            return
        if tag in _BLOCK_TAGS:
            self.parts.append("\n")

    def handle_data(self, data: str) -> None:
        if not data:
            return
        # `<title>` lives inside `<head>`, which is itself a skipped element, so
        # this check must come first. Testing the skip depth first is what made
        # every real page report `title == ""`.
        if self._in_title:
            self.title_parts.append(data)
            return
        if self._skip_depth:
            return
        self.parts.append(data)

    # -- internals --------------------------------------------------------

    def _collect_image(self, attrs: list[tuple[str, str | None]]) -> None:
        # Script and `<noscript>` bodies are skipped containers: their images are
        # tracking pixels and print fallbacks, not venue photography.
        if self._skip_depth:
            return
        attributes = {name.lower(): (value or "") for name, value in attrs}
        for key in ("src", "data-src", "data-original", "data-lazy-src"):
            url = self._absolute(attributes.get(key, ""))
            if url:
                self.images.append(url)
                return
        srcset = attributes.get("srcset") or attributes.get("data-srcset") or ""
        for candidate in srcset.split(","):
            url = self._absolute(candidate.strip().split(" ")[0])
            if url:
                self.images.append(url)
                return

    def _absolute(self, raw: str) -> str:
        raw = (raw or "").strip()
        if not raw:
            # `urljoin(base, "")` returns the *base*, so an `<img>` with no `src`
            # (ordinary for lazy loading) used to be reported as an image
            # candidate pointing at the page itself.
            return ""
        url = require_http_url(urljoin(self.base_url, raw))
        if not url:
            return ""
        if urlparse(url).path.lower().endswith(_SKIP_IMAGE_SUFFIXES):
            return ""
        return url


def extract_page(html: str, base_url: str) -> ExtractedPage:
    """Turn one page's markup into title, readable text and image candidates."""
    parser = _PageExtractor(base_url)
    try:
        parser.feed(html or "")
        parser.close()
    except Exception:  # noqa: BLE001 - malformed markup is normal on the web
        # A partially parsed page is still evidence; HTMLParser only raises on
        # pathological input, and losing the whole page would be worse.
        pass

    lines: list[str] = []
    for raw_line in "".join(parser.parts).splitlines():
        line = " ".join(_WHITESPACE.sub(" ", raw_line).split())
        if line:
            lines.append(line)
    text = "\n".join(lines)[:MAX_TEXT_CHARS]

    seen: set[str] = set()
    images: list[str] = []
    for url in parser.images:
        if url not in seen:
            seen.add(url)
            images.append(url)

    title = " ".join(" ".join(parser.title_parts).split())
    return ExtractedPage(title=title, text=text, images=tuple(images))


# --------------------------------------------------------------------------
# Provider base
# --------------------------------------------------------------------------


@dataclass
class HttpWebProvider(HttpProvider):
    """Base for real search adapters. Subclasses implement `search` only."""

    name: ClassVar[str] = "web"
    user_agent: str = DEFAULT_USER_AGENT
    accept_language: str = "en"
    max_page_bytes: int = MAX_PAGE_BYTES
    max_image_bytes: int = MAX_IMAGE_BYTES

    def __post_init__(self) -> None:
        # A host-supplied budget must survive; an auto-created one must not
        # inherit AGENT_IMAGE_BUDGET_USD.
        supplied_budget = self.budget is not None
        super().__post_init__()
        # Web research is bounded by call counts (`max_search_calls`), not by the
        # image budget: a research run must never be able to refuse a photo render.
        if not supplied_budget:
            self.budget = BudgetLedger(limit_usd=0.0)

    # -- credentials ------------------------------------------------------

    def require_credentials(self) -> None:
        """No key by default. Keyed adapters (Brave) override this."""
        return None

    # -- internet ---------------------------------------------------------

    async def search(self, *, query: str, limit: int = 5) -> list[SearchHit]:
        raise NotImplementedError("each web adapter implements search()")

    async def fetch_page(self, *, url: str) -> WebPage:
        """Extract one page. A dead or unreadable page yields an empty result."""
        target = require_http_url(url)
        if not target:
            return WebPage(url=url, title="", text="", fetched_at=now_iso())
        self.guard_offline()
        try:
            response = await self._http().get(target, headers=self._headers("text/html,application/xhtml+xml"))
        except Exception as exc:  # noqa: BLE001 - narrowed by the helper
            raise ProviderError(
                f"{self.name} could not fetch {target}: {exc}",
                provider=self.name,
                retryable=is_retryable_exception(exc),
            ) from exc

        if response.status_code in DEAD_PAGE_STATUS:
            # Rot, hotlink protection and robots-style blocks are all
            # "no evidence here", not "research is broken".
            return WebPage(url=target, title="", text="", fetched_at=now_iso())
        if response.status_code >= 400:
            raise ProviderError(
                f"{self.name} page fetch returned {response.status_code} for {target}",
                provider=self.name,
                retryable=response.status_code in RETRYABLE_STATUS,
                status=response.status_code,
            )

        content_type = (response.headers.get("content-type") or "").split(";")[0].strip().casefold()
        if content_type and not (content_type.startswith("text/") or content_type in {"application/xhtml+xml"}):
            # A PDF or an image is not text evidence.
            return WebPage(url=target, title="", text="", fetched_at=now_iso())

        html = response.content[: self.max_page_bytes].decode("utf-8", errors="replace")
        extracted = extract_page(html, target)
        return WebPage(
            url=target,
            title=extracted.title,
            text=extracted.text,
            fetched_at=now_iso(),
            images=extracted.images,
        )

    async def fetch_image(self, *, url: str) -> WebImage:
        """Download an image candidate. Any failure means "skip this one"."""
        target = require_http_url(url)
        if not target:
            return WebImage(url=url)
        self.guard_offline()
        try:
            response = await self._http().get(target, headers=self._headers("image/png,image/jpeg,image/webp"))
        except Exception as exc:  # noqa: BLE001
            if is_retryable_exception(exc):
                raise ProviderError(
                    f"{self.name} could not fetch image {target}: {exc}",
                    provider=self.name,
                    retryable=True,
                ) from exc
            return WebImage(url=target)

        if response.status_code >= 400:
            return WebImage(url=target)

        data = response.content
        if not data or len(data) > self.max_image_bytes:
            return WebImage(url=target)
        kind = sniff_image_type(data, fallback="")
        if kind not in _IMAGE_TYPES:
            return WebImage(url=target)
        return WebImage(url=target, content=data, content_type=kind)

    async def estimate_cost(self, *, calls: int = 1) -> CostEstimate:
        return CostEstimate(credits=0, usd=0.0)

    # -- internals --------------------------------------------------------

    def _headers(self, accept: str) -> dict[str, str]:
        return {
            "User-Agent": self.user_agent,
            "Accept": accept,
            "Accept-Language": self.accept_language,
        }


def missing_key_error(name: str, env_var: str = "AGENT_WEB_API_KEY") -> ProviderNotConfigured:
    """One phrasing for every keyed adapter, so setup errors read the same."""
    return ProviderNotConfigured(
        f"{name} needs a search API key. Set {env_var} in .env, or use the "
        "keyless duckduckgo/wikipedia/searxng providers, or stay on the offline fixture."
    )


__all__ = [
    "DEAD_PAGE_STATUS",
    "DEFAULT_USER_AGENT",
    "ExtractedPage",
    "HttpWebProvider",
    "MAX_IMAGE_BYTES",
    "MAX_PAGE_BYTES",
    "MAX_TEXT_CHARS",
    "extract_page",
    "missing_key_error",
    "now_iso",
    "require_http_url",
    "strip_tags",
]
