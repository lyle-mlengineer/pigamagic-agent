"""Deterministic offline `WebProvider` for venue research (Steps 4-14).

Real web research lands at Step 15. Until then this fixture must exercise the
whole research pipeline — budgets, extraction, image ranking, provenance — with
no network, deterministically.

Design: a **corpus** maps query substrings onto canned search hits and pages.
Anything not in the corpus returns no hits, which is how the "research found
nothing" path gets tested for free.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable

from pigamagic_agent.agent.providers.base import CostEstimate, SearchHit, WebImage, WebPage

# A 1x1 PNG. Real bytes so downstream integrity checks behave normally.
_TINY_PNG = bytes.fromhex(
    "89504e470d0a1a0a0000000d49484452000000010000000108060000001f15c489"
    "0000000a49444154789c6360000002000100ffff03000006000557bfabd40000"
    "000049454e44ae426082"
)


@dataclass(frozen=True, slots=True)
class FixturePage:
    """One canned page in the fixture corpus."""

    url: str
    title: str
    text: str
    images: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class FixtureSite:
    """A canned website: searchable by `keywords`, with pages under `hits`."""

    keywords: tuple[str, ...]
    hits: tuple[SearchHit, ...]
    pages: dict[str, FixturePage] = field(default_factory=dict)


def _site(keyword: str, title: str, text: str, image_count: int = 2) -> FixtureSite:
    """Build a single-page fixture site from a keyword."""
    slug = keyword.replace(" ", "-")
    url = f"https://fixture.test/{slug}"
    images = tuple(f"{url}/photo-{index}.png" for index in range(image_count))
    page = FixturePage(url=url, title=title, text=text, images=images)
    hit = SearchHit(url=url, title=title, snippet=text[:140])
    return FixtureSite(keywords=(keyword,), hits=(hit,), pages={url: page})


# Venues that "exist" in the fixture world. Keys are lowercased substrings.
DEFAULT_CORPUS: tuple[FixtureSite, ...] = (
    _site(
        "sarova stanley",
        "The Sarova Stanley — heritage hotel in Nairobi",
        "The Sarova Stanley is a colonial-era hotel in Nairobi's central business "
        "district. Its atrium lobby rises through several floors under a glass roof, "
        "with a long reflecting pool at ground level and an arched colonnade along the "
        "east side. Interiors use pale oak panelling and travertine floors. The ballroom "
        "seats 300 under warm uplighting, with tall north-facing windows.",
        image_count=3,
    ),
    _site(
        "kempinski",
        "Kempinski Roof Gardens — rooftop venue",
        "A rooftop garden venue above the city, with a lawn terrace and open sky. "
        "Pergola structure with climbing plants, and a glass balustrade. Sunset views "
        "across the skyline; evening events use festoon lighting.",
        image_count=2,
    ),
    _site(
        "catholic church",
        "Holy Family Basilica — Catholic church",
        "A Catholic basilica with Gothic arches, stained glass windows along the north "
        "aisle and a long central nave of pale stone. Wooden pews, a marble altar and "
        "soft light from clerestory windows.",
        image_count=2,
    ),
    _site(
        "karura forest",
        "Karura Forest — trails and clearings",
        "An urban forest with marked trails under tall canopy, a river crossing and "
        "open grass clearings. Dappled light through the canopy, red earth paths.",
        image_count=2,
    ),
)

# Queries the fixture deliberately answers with nothing, for the fallback path.
NO_RESULT_MARKERS: tuple[str, ...] = ("nonexistent", "hideout", "my aunt", "unknown", "zzz")


class FixtureWebProvider:
    """Offline, deterministic `WebProvider`.

    Tracks call counts so tests can assert the search budget is respected.
    """

    name = "fixture"

    def __init__(self, corpus: Iterable[FixtureSite] | None = None) -> None:
        self._sites = tuple(corpus if corpus is not None else DEFAULT_CORPUS)
        self.search_calls: list[str] = []
        self.page_calls: list[str] = []
        self.image_calls: list[str] = []

    # --- WebProvider ------------------------------------------------------

    async def search(self, *, query: str, limit: int = 5) -> list[SearchHit]:
        self.search_calls.append(query)
        needle = query.casefold()
        if any(marker in needle for marker in NO_RESULT_MARKERS):
            return []

        hits: list[SearchHit] = []
        for site in self._sites:
            if any(keyword in needle for keyword in site.keywords):
                hits.extend(site.hits)
            if len(hits) >= limit:
                break
        return hits[:limit]

    async def fetch_page(self, *, url: str) -> WebPage:
        self.page_calls.append(url)
        for site in self._sites:
            page = site.pages.get(url)
            if page is not None:
                return WebPage(
                    url=page.url,
                    title=page.title,
                    text=page.text,
                    fetched_at="2024-01-01T00:00:00Z",
                    images=page.images,
                )
        # Unknown URLs are a real condition (links rot); do not raise.
        return WebPage(url=url, title="", text="", fetched_at="2024-01-01T00:00:00Z")

    async def fetch_image(self, *, url: str) -> WebImage:
        self.image_calls.append(url)
        alt = url.rsplit("/", 1)[-1].replace(".png", "").replace("-", " ")
        return WebImage(url=url, page_url=url.rsplit("/", 1)[0], alt_text=alt, content=_TINY_PNG, content_type="image/png")

    async def estimate_cost(self, *, calls: int = 1) -> CostEstimate:
        return CostEstimate(credits=0, usd=0.0)

    # --- test helpers -----------------------------------------------------

    def image_urls_for(self, venue_keyword: str) -> tuple[str, ...]:
        for site in self._sites:
            if venue_keyword in site.keywords:
                urls: list[str] = []
                for page in site.pages.values():
                    urls.extend(page.images)
                return tuple(urls)
        return ()


__all__ = [
    "DEFAULT_CORPUS",
    "FixturePage",
    "FixtureSite",
    "FixtureWebProvider",
    "NO_RESULT_MARKERS",
]
