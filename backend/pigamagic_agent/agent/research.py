"""Venue research: what does this specific place actually look like?

(design.md §6.1)

The pipeline is deliberately boring and budgeted:

    normalize key → cache hit? return
      ↓ miss
    1. plan 3-5 queries from the venue name and event
    2. search, capped by settings.max_search_calls
    3. extract the top pages into text
    4. collect image candidates, cap by settings.max_images_per_venue
    5. vision pass over the best images for architectural vocabulary
    6. synthesise a VenueProfile with confidence + sources
    7. cache it

Two behaviours matter more than the happy path:

**Honesty over invention.** When research finds nothing, we return a profile
with `confidence < 0.5`, empty `signature_details`, and a warning. We never
manufacture architecture from the event type — that is precisely how a
"hotel ballroom" ends up looking like generic stock. (Step 1's `VenueProfile`
invariant enforces this at the schema level too.)

**Shared cache.** Venues are shared, expensive and rarely change, so the cache
key is the normalised venue, not the user. The hundredth wedding at the same
hotel costs nothing.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Protocol, Sequence

from pigamagic_agent.agent.config import AgentSettings, get_settings
from pigamagic_agent.agent.conventions import ConventionPack
from pigamagic_agent.agent.hashing import normalize_venue_key, venue_research_key
from pigamagic_agent.agent.profiles import VenueProfile
from pigamagic_agent.agent.providers.base import SearchHit, VisionProvider, WebImage, WebProvider

# Bumped when the research strategy changes shape (invalidates cached profiles).
RESEARCH_VERSION = "1"

# Confidence bands. Deliberately coarse and documented so UI copy can be
# written against them ("we found this venue", "we guessed").
CONFIDENCE_RESEARCHED = 0.7
CONFIDENCE_PARTIAL = 0.45
CONFIDENCE_NONE = 0.15

# Words that describe a venue *kind*, used to seed queries and to label the
# profile. Kept short and generic.
# Marker placed in `sources` when nothing was found, so provenance is explicit
# rather than an empty list that could mean "forgot to record".
NO_SOURCES_MARKER = "no-sources"
_VENUE_KIND_HINTS: tuple[tuple[str, str], ...] = (
    ("hotel", "hotel"),
    ("ballroom", "ballroom"),
    ("church", "church"),
    ("basilica", "church"),
    ("cathedral", "cathedral"),
    ("garden", "garden"),
    ("park", "park"),
    ("beach", "beach"),
    ("restaurant", "restaurant"),
    ("rooftop", "rooftop"),
    ("hall", "hall"),
    ("campus", "campus"),
    ("forest", "forest"),
    ("trail", "trail"),
    ("club", "club"),
    ("resort", "resort"),
    ("lodge", "lodge"),
    ("museum", "museum"),
    ("gallery", "gallery"),
    ("stadium", "stadium"),
)

# Architectural vocabulary extracted from page text. Matching plain words keeps
# this deterministic and cheap; the vision pass refines it.
_ARCHITECTURE_TERMS: tuple[str, ...] = (
    "gothic arch",
    "stained glass",
    "colonnade",
    "atrium",
    "glass roof",
    "pergola",
    "balustrade",
    "marble",
    "travertine",
    "pale oak",
    "panelling",
    "vaulted",
    "nave",
    "pews",
    "terrace",
    "lawn",
    "courtyard",
    "reflecting pool",
    "hardwood",
    "brick",
    "thatched",
    "floor-to-ceiling windows",
    "ballroom",
    "chandelier",
    "rooftop",
    "pool",
    "spa",
    "lounge",
)

_LIGHTING_TERMS: tuple[str, ...] = (
    "uplighting",
    "festoon",
    "candlelight",
    "window light",
    "clerestory",
    "sunset",
    "dappled",
    "floodlit",
    "spotlit",
    "north-facing",
)

_COLOUR_TERMS: tuple[str, ...] = (
    "ivory",
    "cream",
    "pale oak",
    "travertine",
    "marble",
    "green",
    "deep green",
    "sage",
    "terracotta",
    "burgundy",
    "navy",
    "gold",
    "white",
    "stone",
)

_SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+")

# Words that make a sentence a usable visual anchor. Matched on word boundaries:
# substring matching turned "Archived from…" into a `signature_detail` because
# "archived" contains "arch", which is exactly the kind of fake evidence this
# module exists to prevent.
_DETAIL_TERMS: tuple[str, ...] = (
    "pool",
    "colonnade",
    "atrium",
    "arch",
    "arches",
    "garden",
    "terrace",
    "window",
    "windows",
    "floor",
    "floors",
    "ballroom",
    "ceiling",
    "marble",
    "staircase",
    "stairs",
    "fountain",
    "lawn",
    "courtyard",
    "chandelier",
    "stage",
    "dance floor",
    "rooftop",
)
_DETAIL_PATTERN = re.compile(r"\b(?:" + "|".join(_DETAIL_TERMS) + r")\b")

# Citation furniture that survives naive text extraction. A bibliography is not
# a description of a room, so a sentence containing these is never an anchor.
_BOILERPLATE_MARKERS: tuple[str, ...] = (
    "archived from",
    "retrieved",
    "isbn",
    "doi:",
    "http://",
    "https://",
    "↑",
    "[edit]",
    "edit]",
    "citation needed",
    "external links",
)

# Tokens that say nothing about which venue a page is about.
_VENUE_STOPWORDS: frozenset[str] = frozenset(
    {"the", "and", "for", "with", "hotel", "venue", "hall", "centre", "center"}
)


# --------------------------------------------------------------------------
# Cache
# --------------------------------------------------------------------------


class VenueResearchCache(Protocol):
    def get(self, key: str) -> VenueProfile | None: ...

    def put(self, key: str, profile: VenueProfile, sources: Sequence[str]) -> None: ...


@dataclass(slots=True)
class CachedVenue:
    profile: VenueProfile
    sources: list[str] = field(default_factory=list)
    fetched_at: str = ""


class InMemoryVenueCache:
    """Process-local cache. A Postgres-backed one replaces this at Step 11."""

    def __init__(self) -> None:
        self._store: dict[str, CachedVenue] = {}

    def get(self, key: str) -> VenueProfile | None:
        entry = self._store.get(key)
        return entry.profile if entry else None

    def put(self, key: str, profile: VenueProfile, sources: Sequence[str]) -> None:
        self._store[key] = CachedVenue(profile=profile, sources=list(sources))

    def __len__(self) -> int:
        return len(self._store)


# --------------------------------------------------------------------------
# Result
# --------------------------------------------------------------------------


@dataclass(slots=True)
class VenueResearchOutcome:
    profile: VenueProfile
    sources: list[str]
    warnings: list[str] = field(default_factory=list)
    search_calls: int = 0
    page_fetches: int = 0
    image_fetches: int = 0
    cache_hit: bool = False

    @property
    def has_research(self) -> bool:
        return self.profile.has_research() and not self.cache_hit or self.cache_hit


# --------------------------------------------------------------------------
# Query planning
# --------------------------------------------------------------------------


def plan_queries(venue: str, event: str = "", *, event_kind: str = "") -> list[str]:
    """Build 3-5 search queries for a venue. Deterministic and deduplicated.

    Deliberately not LLM-driven: query planning is cheap, must be reproducible,
    and has a fixed budget. An LLM here would add cost and variance for no gain.
    """
    base = venue.strip()
    if not base:
        return []

    context = (event_kind or event).strip()
    candidates = [
        base,
        f"{base} interior",
        f"{base} photos",
        f"{base} wedding venue" if "wedding" in context.casefold() else f"{base} event space",
        f"{base} architecture exterior",
    ]

    queries: list[str] = []
    seen: set[str] = set()
    for query in candidates:
        key = query.casefold()
        if key in seen:
            continue
        seen.add(key)
        queries.append(query)
    return queries


def detect_venue_kind(venue: str, text_corpus: str = "") -> str:
    """Infer a coarse venue kind from the name, then from text."""
    haystack = f"{venue} {text_corpus}".casefold()
    for term, kind in _VENUE_KIND_HINTS:
        if term in haystack:
            return kind
    return "venue"


# --------------------------------------------------------------------------
# Extraction
# --------------------------------------------------------------------------


def extract_terms(text: str, terms: Sequence[str]) -> list[str]:
    """Return the terms present in `text`, in canonical order, deduplicated."""
    haystack = text.casefold()
    found: list[str] = []
    for term in terms:
        if re.search(rf"\b{re.escape(term)}\b", haystack) and term not in found:
            found.append(term)
    return found


def extract_details(pages: Sequence[str]) -> list[str]:
    """Pull short, quotable venue details out of page text.

    These become `signature_details`, which the planner uses as concrete
    anchors ("the long reflecting pool") rather than vague adjectives. A
    sentence only qualifies when it names a *visual* feature and is not
    citation furniture.
    """
    details: list[str] = []
    seen: list[str] = []
    for page in pages:
        for sentence in _SENTENCE_SPLIT.split(page):
            candidate = _clean_detail(sentence)
            if not 25 <= len(candidate) <= 220:
                continue
            lowered = candidate.casefold()
            if not _DETAIL_PATTERN.search(lowered):
                continue
            if any(marker in lowered for marker in _BOILERPLATE_MARKERS):
                continue
            key = _detail_key(candidate)
            # Real pages repeat themselves: a menu label is echoed by the panel
            # it opens, and a blurb appears in both the card and the footer. Two
            # anchors that say the same thing waste one of six slots and make the
            # planner look like it has more evidence than it does.
            if any(key == other or key in other or other in key for other in seen):
                continue
            seen.append(key)
            details.append(candidate)
            if len(details) >= 6:
                return details
    return details


# UI labels that precede real content in extracted page text. They are chrome,
# not architecture, and they make otherwise identical anchors look different.
_DETAIL_LABELS: tuple[str, ...] = (
    "more details",
    "read more",
    "learn more",
    "view more",
    "see more",
    "book now",
    "skip to content",
)

# A label that immediately repeats itself is one phrase plus the panel it
# opens: "Meetings & Events Meetings & Events From high-tech rooms...".
_DOUBLED_PHRASE = re.compile(r"^(.{5,60}?)\s+\1(?![\w])", re.IGNORECASE)


def _clean_detail(sentence: str) -> str:
    """Strip page chrome from one candidate anchor."""
    candidate = " ".join(sentence.split())
    changed = True
    while changed:
        changed = False
        for label in _DETAIL_LABELS:
            if candidate.casefold().startswith(label + " "):
                candidate = candidate[len(label) + 1 :].lstrip()
                changed = True
    doubled = _DOUBLED_PHRASE.match(candidate)
    if doubled:
        candidate = candidate[len(doubled.group(1)) :].lstrip()
    return candidate


def _detail_key(candidate: str) -> str:
    """Alphanumeric-only form used to spot an anchor that says the same thing."""
    return " ".join(re.findall(r"[a-z0-9]+", candidate.casefold()))


def rank_hits(venue: str, hits: Sequence[SearchHit]) -> list[SearchHit]:
    """Order search hits by how much their title/URL looks like this venue.

    Search engines happily return a beer brand for a hotel query. Those pages
    are not evidence, and letting them in contaminates the palette and detail
    lists with another subject's vocabulary. When at least two hits share a
    token with the venue we keep only those; otherwise the original order is
    preserved, because an unrelated-looking title is not proof of irrelevance.

    Stable for equal scores, so a profile stays reproducible.
    """
    tokens = {
        token
        for token in re.findall(r"[a-z0-9]{3,}", venue.casefold())
        if token not in _VENUE_STOPWORDS
    }
    if not tokens:
        return list(hits)

    scored: list[tuple[int, int, SearchHit]] = []
    for index, hit in enumerate(hits):
        haystack = f"{hit.title} {hit.url}".casefold()
        score = sum(1 for token in tokens if token in haystack)
        scored.append((-score, index, hit))
    scored.sort(key=lambda row: (row[0], row[1]))

    relevant = [hit for negative, _, hit in scored if -negative > 0]
    return relevant if len(relevant) >= 2 else [hit for _, _, hit in scored]


def rank_images(urls: Sequence[str], limit: int) -> list[str]:
    """Order image candidates deterministically and cap them.

    Deduplicates by URL and prefers URLs that look like the venue itself over
    generic site assets. Stable sort, so the same input always yields the same
    order (a requirement for reproducible profiles).
    """
    seen: set[str] = set()
    ranked: list[tuple[int, str]] = []
    for url in urls:
        if not url or url in seen:
            continue
        seen.add(url)
        lowered = url.casefold()
        # Prefer real-looking photos; deprioritise logos/icons/banners.
        penalty = 1 if any(bad in lowered for bad in ("logo", "icon", "banner", "favicon", "sprite")) else 0
        ranked.append((penalty, url))
    ranked.sort()
    return [url for _, url in ranked[:limit]]


# --------------------------------------------------------------------------
# Research
# --------------------------------------------------------------------------


async def research_venue(
    venue: str,
    event: str = "",
    *,
    web: WebProvider,
    vision: VisionProvider | None = None,
    cache: VenueResearchCache | None = None,
    settings: AgentSettings | None = None,
    pack: ConventionPack | None = None,
) -> VenueResearchOutcome:
    """Research a venue into a `VenueProfile`, or admit we could not."""
    active = settings or get_settings()
    store = cache if cache is not None else InMemoryVenueCache()

    venue_key = normalize_venue_key(venue, event)
    cache_key = venue_research_key(venue, event)

    cached = store.get(cache_key)
    if cached is not None:
        return VenueResearchOutcome(
            profile=cached,
            sources=list(cached.sources),
            warnings=[],
            cache_hit=True,
        )

    if not venue.strip():
        return _deal_with_ignorance(
            venue_key=venue_key,
            venue=venue,
            event=event,
            pack=pack,
            store=store,
            cache_key=cache_key,
            reason="No venue was provided.",
            search_calls=0,
        )

    queries = plan_queries(venue, event, event_kind=pack.event_key if pack else "")
    budget = max(active.max_search_calls, 0)
    queries = queries[:budget]

    hits: list[SearchHit] = []
    search_calls = 0
    for query in queries:
        if search_calls >= budget:
            break
        search_calls += 1
        found = await web.search(query=query, limit=5)
        for hit in found:
            if all(existing.url != hit.url for existing in hits):
                hits.append(hit)

    if not hits:
        return _deal_with_ignorance(
            venue_key=venue_key,
            venue=venue,
            event=event,
            pack=pack,
            store=store,
            cache_key=cache_key,
            reason=f"No web results were found for {venue!r}.",
            search_calls=search_calls,
            queries=queries,
        )

    # Extract the best pages. Cap so a link-heavy site cannot blow the budget.
    # Unrelated hits (a beer brand for a hotel query) are dropped first, so they
    # cannot donate their vocabulary to this venue's profile.
    hits = rank_hits(venue, hits)
    page_fetches = 0
    pages_text: list[str] = []
    image_urls: list[str] = []
    sources: list[str] = []
    for hit in hits[: active.max_pages_per_venue]:
        page = await web.fetch_page(url=hit.url)
        page_fetches += 1
        if page.url:
            sources.append(page.url)
        if page.text:
            pages_text.append(page.text)
        # Real adapters return absolute image URLs harvested from the page
        # markup. `_images_in_text` stays as a fallback for providers whose text
        # mentions image URLs directly.
        image_urls.extend(getattr(page, "images", ()) or ())
        image_urls.extend(_images_in_text(page.text))

    # Fetch a bounded number of images for the vision pass.
    images: list[WebImage] = []
    image_fetches = 0
    candidates = rank_images(image_urls, active.max_images_per_venue)
    for url in candidates[: active.max_images_per_venue]:
        image = await web.fetch_image(url=url)
        image_fetches += 1
        if image.content:
            images.append(image)

    joined = " ".join(pages_text)
    architecture = extract_terms(joined, _ARCHITECTURE_TERMS)
    lighting = extract_terms(joined, _LIGHTING_TERMS)
    palette = extract_terms(joined, _COLOUR_TERMS)
    details = extract_details(pages_text)
    kind = detect_venue_kind(venue, joined)

    # Vision pass over researched images: authoritative visual vocabulary.
    vision_notes: list[str] = []
    if vision is not None and images:
        for image in images:
            result = await vision.describe_image(
                image=image.content,
                instruction=(
                    "Describe the venue visible in this photograph: architecture, "
                    "materials, lighting and any distinctive feature. Omit what you "
                    "cannot see."
                ),
                schema={},
            )
            for cue in result.data.get("venue_cues", []) or []:
                text = str(cue).strip()
                if text and text not in vision_notes:
                    vision_notes.append(text)

    # Merge vision cues into the anchor lists.
    for cue in vision_notes:
        if cue.casefold() not in {item.casefold() for item in architecture}:
            architecture.append(cue)

    has_evidence = bool(architecture or details or vision_notes)
    confidence = CONFIDENCE_RESEARCHED if has_evidence else CONFIDENCE_PARTIAL
    if len(pages_text) <= 1:
        confidence = min(confidence, CONFIDENCE_PARTIAL)

    warnings: list[str] = []
    if not has_evidence:
        confidence = CONFIDENCE_NONE
        warnings.append(
            f"Pages about {venue!r} were found but contained no usable visual detail; "
            "venue appearance is estimated."
        )

    profile = VenueProfile(
        venue_key=venue_key[:200],
        name=venue.strip()[:200],
        kind=kind,
        indoor_outdoor=_guess_indoor_outdoor(joined),
        architecture=architecture[:10],
        materials=[term for term in architecture if term in {"marble", "travertine", "pale oak", "brick", "hardwood"}],
        palette=palette[:8],
        lighting=lighting[:8],
        signature_details=details[:6],
        layout=_first_layout_sentence(pages_text),
        time_of_day_evidence=[term for term in lighting if term in {"sunset", "dappled", "floodlit", "candlelight"}],
        confidence=confidence,
        sources=sources[:10] or [NO_SOURCES_MARKER],
    )

    store.put(cache_key, profile, sources)

    return VenueResearchOutcome(
        profile=profile,
        sources=sources,
        warnings=warnings,
        search_calls=search_calls,
        page_fetches=page_fetches,
        image_fetches=image_fetches,
        cache_hit=False,
    )


def _deal_with_ignorance(
    *,
    venue_key: str,
    venue: str,
    event: str,
    pack: ConventionPack | None,
    store: VenueResearchCache,
    cache_key: str,
    reason: str,
    search_calls: int,
    queries: Sequence[str] = (),
) -> VenueResearchOutcome:
    """Build an honest low-confidence profile when research finds nothing.

    Critically: `signature_details` stays EMPTY. Inventing anchors here is the
    single most damaging thing this module could do.
    """
    lighting: list[str] = []
    if pack is not None:
        lighting = list(pack.lighting_notes[:4])

    warnings = [
        f"{reason} Venue appearance is estimated from the event type, not researched.",
    ]
    if queries:
        warnings.append(f"Queried: {', '.join(queries)}")

    profile = VenueProfile(
        venue_key=venue_key[:200],
        name=venue.strip()[:200],
        kind=detect_venue_kind(venue),
        indoor_outdoor="mixed",
        architecture=[],
        materials=[],
        palette=[],
        lighting=lighting,
        signature_details=[],
        layout=None,
        time_of_day_evidence=[],
        confidence=CONFIDENCE_NONE,
        sources=[NO_SOURCES_MARKER],
    )
    store.put(cache_key, profile, profile.sources)
    return VenueResearchOutcome(
        profile=profile,
        sources=list(profile.sources),
        warnings=warnings,
        search_calls=search_calls,
        cache_hit=False,
    )


# --------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------


def unreachable_venue_outcome(
    venue: str,
    event: str,
    pack: ConventionPack | None = None,
    *,
    reason: str,
) -> VenueResearchOutcome:
    """The honest-ignorance outcome for a research provider that failed.

    A search provider that is keyless, rate-limited or down tells us nothing
    about the venue, which is exactly the same fact as "the search returned no
    results". Producing that same low-confidence profile keeps the product
    promise (a shot list with a warning) instead of turning a missing key into an
    error page.
    """
    return _deal_with_ignorance(
        venue_key=normalize_venue_key(venue, event),
        venue=venue,
        event=event,
        pack=pack,
        store=InMemoryVenueCache(),
        cache_key=venue_research_key(venue, event),
        reason=reason,
        search_calls=0,
    )


def _images_in_text(text: str) -> list[str]:
    """Find image-like URLs mentioned in extracted page text."""
    return re.findall(r"https?://\S+?\.(?:png|jpe?g|webp)", text)


def _guess_indoor_outdoor(text: str) -> str:
    lowered = text.casefold()
    indoor_hits = sum(
        1 for term in ("indoor", "interior", "atrium", "ballroom", "pews", "ceiling", "roof")
        if term in lowered
    )
    outdoor_hits = sum(
        1 for term in ("outdoor", "garden", "lawn", "terrace", "trail", "forest", "beach", "skyline")
        if term in lowered
    )
    if indoor_hits and outdoor_hits:
        return "mixed"
    if outdoor_hits > indoor_hits:
        return "outdoor"
    if indoor_hits > outdoor_hits:
        return "indoor"
    return "mixed"


def _first_layout_sentence(pages: Sequence[str]) -> str | None:
    """One sentence describing how the space is laid out, if available."""
    for page in pages:
        for sentence in _SENTENCE_SPLIT.split(page):
            candidate = sentence.strip()
            lowered = candidate.casefold()
            if 20 <= len(candidate) <= 300 and any(
                term in lowered for term in ("opens onto", "leads to", "connects", "at ground level", "along the")
            ):
                return candidate
    return None


__all__ = [
    "CachedVenue",
    "CONFIDENCE_NONE",
    "CONFIDENCE_PARTIAL",
    "CONFIDENCE_RESEARCHED",
    "InMemoryVenueCache",
    "RESEARCH_VERSION",
    "VenueResearchCache",
    "VenueResearchOutcome",
    "detect_venue_kind",
    "extract_details",
    "extract_terms",
    "plan_queries",
    "rank_hits",
    "rank_images",
    "research_venue",
    "unreachable_venue_outcome",
]
