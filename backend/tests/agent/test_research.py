"""Step 4: venue research with cache, budget and provenance.

Properties that matter (design.md §6.1):
- the same venue researched twice performs ONE research round
- the search budget is never exceeded, even for a pathological name
- a venue with no results yields low confidence + warning + NO invented detail
- every profile carries sources, or an explicit `no-sources` marker
- two descriptions of the same venue share one cache entry
"""

from typing import Any

import pytest

from pigamagic_agent.agent.config import AgentSettings
from pigamagic_agent.agent.conventions import get_pack
from pigamagic_agent.agent.profiles import VenueProfile
from pigamagic_agent.agent.providers.vision_fixture import FixtureVisionProvider
from pigamagic_agent.agent.providers.web_fixture import FixtureWebProvider
from pigamagic_agent.agent.research import (
    CONFIDENCE_NONE,
    CONFIDENCE_PARTIAL,
    CONFIDENCE_RESEARCHED,
    InMemoryVenueCache,
    detect_venue_kind,
    extract_details,
    extract_terms,
    plan_queries,
    rank_images,
    research_venue,
)


def _settings(**overrides: Any) -> AgentSettings:
    return AgentSettings(_env_file=None, **overrides)


# --------------------------------------------------------------------------
# Query planning
# --------------------------------------------------------------------------


def test_plan_queries_produces_a_small_deduplicated_set() -> None:
    queries = plan_queries("The Sarova Stanley", "wedding")
    assert 3 <= len(queries) <= 5
    assert len(set(queries)) == len(queries)
    assert queries[0] == "The Sarova Stanley"


def test_plan_queries_is_deterministic() -> None:
    assert plan_queries("Kempinski", "birthday") == plan_queries("Kempinski", "birthday")


def test_plan_queries_empty_for_blank_venue() -> None:
    assert plan_queries("") == []
    assert plan_queries("   ") == []


def test_plan_queries_uses_event_context() -> None:
    wedding = plan_queries("Kempinski", event_kind="wedding")
    assert any("wedding" in query for query in wedding)


# --------------------------------------------------------------------------
# Extraction helpers
# --------------------------------------------------------------------------


def test_extract_terms_finds_canonical_terms() -> None:
    text = "The atrium rises to a glass roof above travertine floors and an arched colonnade."
    # Terms are returned in the order of the supplied term list, which is what
    # makes extraction deterministic across callers.
    found = extract_terms(text, ("atrium", "glass roof", "travertine", "colonnade", "beach"))
    assert found == ["atrium", "glass roof", "travertine", "colonnade"]


def test_extract_terms_respects_word_boundaries() -> None:
    # "bar" must not match inside "barbecue".
    assert extract_terms("a barbecue area", ("bar",)) == []


def test_extract_details_pulls_quotable_sentences() -> None:
    pages = [
        "The hotel has a long reflecting pool at ground level. "
        "Guests arrive through a courtyard with tall windows."
    ]
    details = extract_details(pages)
    assert details
    assert all(20 <= len(detail) <= 160 for detail in details)


def test_extract_details_skips_irrelevant_sentences() -> None:
    assert extract_details(["Book online for the best rates available today."]) == []


def test_rank_images_deprioritises_assets_and_dedupes() -> None:
    urls = [
        "https://x.test/logo.png",
        "https://x.test/venue-1.png",
        "https://x.test/venue-1.png",
        "https://x.test/banner.png",
        "https://x.test/venue-2.png",
    ]
    ranked = rank_images(urls, limit=10)
    # Real-looking photos come first; asset URLs sort after them, alphabetically
    # within each tier so the order is stable.
    assert ranked == [
        "https://x.test/venue-1.png",
        "https://x.test/venue-2.png",
        "https://x.test/banner.png",
        "https://x.test/logo.png",
    ]


def test_rank_images_respects_limit_and_is_deterministic() -> None:
    urls = [f"https://x.test/p{index}.png" for index in range(10)]
    assert len(rank_images(urls, limit=3)) == 3
    assert rank_images(urls, limit=3) == rank_images(urls, limit=3)


def test_detect_venue_kind_from_name_then_text() -> None:
    assert detect_venue_kind("Kempinski") == "venue"
    assert detect_venue_kind("Holy Family Church") == "church"
    assert detect_venue_kind("Somewhere", "a rooftop garden") in {"garden", "rooftop"}


# --------------------------------------------------------------------------
# Happy path
# --------------------------------------------------------------------------


async def test_researches_a_known_venue_with_provenance() -> None:
    web = FixtureWebProvider()
    outcome = await research_venue("The Sarova Stanley", "wedding", web=web, settings=_settings())

    profile = outcome.profile
    assert profile.confidence >= CONFIDENCE_PARTIAL
    assert profile.sources
    assert "no-sources" not in profile.sources
    assert profile.signature_details, "a researched venue should yield concrete anchors"
    assert any("colonnade" in item or "atrium" in item or "pool" in item for item in profile.architecture)


async def test_vision_pass_adds_visual_vocabulary() -> None:
    web = FixtureWebProvider()
    vision = FixtureVisionProvider()
    outcome = await research_venue(
        "The Sarova Stanley", "wedding", web=web, vision=vision, settings=_settings()
    )
    assert vision.calls, "the vision pass should have been used"
    assert outcome.profile.confidence >= CONFIDENCE_PARTIAL


async def test_profile_is_indoor_or_outdoor_classified() -> None:
    outcome = await research_venue("Karura Forest", "road trip", web=FixtureWebProvider(), settings=_settings())
    assert outcome.profile.indoor_outdoor in {"indoor", "outdoor", "mixed"}


# --------------------------------------------------------------------------
# Caching
# --------------------------------------------------------------------------


async def test_second_research_is_a_cache_hit_with_no_web_calls() -> None:
    web = FixtureWebProvider()
    cache = InMemoryVenueCache()

    first = await research_venue("The Sarova Stanley", "wedding", web=web, cache=cache, settings=_settings())
    calls_after_first = len(web.search_calls)

    second = await research_venue("The Sarova Stanley", "wedding", web=web, cache=cache, settings=_settings())

    assert first.cache_hit is False
    assert second.cache_hit is True
    assert len(web.search_calls) == calls_after_first, "cache hit must not search"


async def test_two_descriptions_of_one_venue_share_a_cache_entry() -> None:
    """The main cost lever: the hundredth wedding at the same hotel is free."""
    web = FixtureWebProvider()
    cache = InMemoryVenueCache()

    await research_venue(
        "The Sarova Stanley Hotel, Nairobi", "Wedding", web=web, cache=cache, settings=_settings()
    )
    second = await research_venue(
        "  sarova stanley hotel   nairobi ", "wedding", web=web, cache=cache, settings=_settings()
    )

    assert second.cache_hit is True


async def test_different_venues_do_not_share_cache() -> None:
    web = FixtureWebProvider()
    cache = InMemoryVenueCache()

    await research_venue("The Sarova Stanley", "wedding", web=web, cache=cache, settings=_settings())
    second = await research_venue("Kempinski", "wedding", web=web, cache=cache, settings=_settings())

    assert second.cache_hit is False
    assert second.profile.name == "Kempinski"


# --------------------------------------------------------------------------
# Budgets
# --------------------------------------------------------------------------


async def test_search_budget_is_never_exceeded() -> None:
    web = FixtureWebProvider()
    settings = _settings(max_search_calls=2)

    outcome = await research_venue("The Sarova Stanley", "wedding", web=web, settings=settings)

    assert outcome.search_calls <= 2
    assert len(web.search_calls) <= 2


async def test_image_budget_is_never_exceeded() -> None:
    web = FixtureWebProvider()
    settings = _settings(max_images_per_venue=1)

    outcome = await research_venue("The Sarova Stanley", "wedding", web=web, settings=settings)

    assert outcome.image_fetches <= 1
    assert len(web.image_calls) <= 1


async def test_page_budget_is_never_exceeded() -> None:
    web = FixtureWebProvider()
    settings = _settings(max_pages_per_venue=1)

    outcome = await research_venue("The Sarova Stanley", "wedding", web=web, settings=settings)

    assert outcome.page_fetches <= 1


async def test_pathological_venue_name_still_respects_budget() -> None:
    web = FixtureWebProvider()
    settings = _settings(max_search_calls=3)
    pathological = "The " + "Wedding " * 200 + "Hotel"

    outcome = await research_venue(pathological, "wedding", web=web, settings=settings)

    assert outcome.search_calls <= 3
    assert len(web.search_calls) <= 3


# --------------------------------------------------------------------------
# Honest ignorance (the important failure path)
# --------------------------------------------------------------------------


async def test_unfindable_venue_admits_low_confidence() -> None:
    web = FixtureWebProvider()
    outcome = await research_venue(
        "Nonexistent Hideout Palace", "wedding", web=web, settings=_settings()
    )

    profile = outcome.profile
    assert profile.confidence < 0.5
    assert profile.confidence == CONFIDENCE_NONE
    assert outcome.warnings
    assert profile.sources == ["no-sources"]


async def test_unfindable_venue_does_not_invent_signature_details() -> None:
    """The single most damaging thing research could do (design.md §6.1)."""
    outcome = await research_venue(
        "Nonexistent Hideout Palace", "wedding", web=FixtureWebProvider(), settings=_settings()
    )

    assert outcome.profile.signature_details == []
    assert outcome.profile.architecture == []


async def test_unfindable_venue_still_returns_a_valid_profile() -> None:
    """The schema invariant must accept the honest profile."""
    outcome = await research_venue(
        "Nonexistent Hideout Palace", "wedding", web=FixtureWebProvider(), settings=_settings()
    )
    assert isinstance(outcome.profile, VenueProfile)
    assert outcome.profile.has_research() is False


async def test_missing_venue_is_handled_without_raising() -> None:
    outcome = await research_venue("", "wedding", web=FixtureWebProvider(), settings=_settings())
    assert outcome.profile.confidence == CONFIDENCE_NONE
    assert outcome.warnings
    assert outcome.search_calls == 0


async def test_low_confidence_profile_still_uses_convention_lighting() -> None:
    """Falling back to event conventions is allowed; inventing anchors is not."""
    pack = get_pack("wedding")
    outcome = await research_venue(
        "Nonexistent Hideout Palace",
        "wedding",
        web=FixtureWebProvider(),
        settings=_settings(),
        pack=pack,
    )
    assert outcome.profile.lighting
    assert outcome.profile.signature_details == []


# --------------------------------------------------------------------------
# Confidence bands reflect evidence
# --------------------------------------------------------------------------


async def test_researched_venue_confidence_is_high() -> None:
    outcome = await research_venue(
        "The Sarova Stanley", "wedding", web=FixtureWebProvider(), settings=_settings()
    )
    assert outcome.profile.confidence >= CONFIDENCE_RESEARCHED - 0.3


async def test_single_page_result_is_capped_at_partial_confidence() -> None:
    outcome = await research_venue(
        "The Sarova Stanley", "wedding", web=FixtureWebProvider(), settings=_settings(max_pages_per_venue=1)
    )
    assert outcome.profile.confidence <= CONFIDENCE_PARTIAL


# --------------------------------------------------------------------------
# Determinism and provenance
# --------------------------------------------------------------------------


async def test_research_is_deterministic_for_same_inputs() -> None:
    first = await research_venue(
        "The Sarova Stanley", "wedding", web=FixtureWebProvider(), settings=_settings()
    )
    second = await research_venue(
        "The Sarova Stanley", "wedding", web=FixtureWebProvider(), settings=_settings()
    )
    assert first.profile.model_dump() == second.profile.model_dump()


async def test_every_profile_has_sources_or_an_explicit_marker() -> None:
    for venue in ("The Sarova Stanley", "Nonexistent Hideout Palace", ""):
        outcome = await research_venue(venue, "wedding", web=FixtureWebProvider(), settings=_settings())
        assert outcome.profile.sources, f"{venue!r} produced no source record"


async def test_venue_key_is_normalised_on_the_profile() -> None:
    outcome = await research_venue(
        "The Sarova Stanley Hotel, Nairobi", "Wedding", web=FixtureWebProvider(), settings=_settings()
    )
    assert outcome.profile.venue_key == "sarova stanley hotel nairobi|wedding"
