"""Step 15: real web research providers (keyless Wikipedia/SearXNG + Brave).

What must hold, all of it without spending a search credit or touching the
network in CI:

- one HTML extractor serves every adapter, and it handles the things real pages
  do (script bodies, lazy-loaded images, relative URLs, `srcset`, malformed
  markup) rather than the things a hand-written fixture does
- a dead page (404/403/410/451) is *empty evidence*, not an error; a 429/5xx is
  an error, because it says something about our budget or credentials
- an image that cannot be used is skipped, never fatal
- keyless providers do not demand a key, and a keyed provider fails with a
  message naming the setting to fix
- web research can never consume the image budget
- research asks for images the page actually offers — the old code invented
  `{page}/photo-N.png` candidates, which 404 against every real website
"""

from __future__ import annotations

from typing import Any

import httpx
import pytest

from pigamagic_agent.agent.config import AgentSettings
from pigamagic_agent.agent.conventions import get_pack
from pigamagic_agent.agent.providers.base import (
    OfflineModeError,
    ProviderError,
    ProviderNotConfigured,
    SearchHit,
    WebImage,
    WebPage,
    WebProvider,
)
from pigamagic_agent.agent.providers.http_base import BudgetExceeded, BudgetLedger
from pigamagic_agent.agent.providers.image_fixture import render_fixture_png
from pigamagic_agent.agent.providers.registry import (
    available_web_providers,
    build_web_provider,
    describe_web_providers,
)
from pigamagic_agent.agent.providers.web_brave import BraveWebProvider
from pigamagic_agent.agent.providers.web_common import (
    DEAD_PAGE_STATUS,
    HttpWebProvider,
    extract_page,
    missing_key_error,
    require_http_url,
    strip_tags,
)
from pigamagic_agent.agent.providers.web_searxng import SearxngWebProvider
from pigamagic_agent.agent.providers.web_wikipedia import WikipediaWebProvider
from pigamagic_agent.agent.research import rank_hits, research_venue
from pigamagic_agent.agent.providers.vision_fixture import FixtureVisionProvider

PNG = render_fixture_png(prompt="venue", width=32, height=32)


def _settings(**overrides: Any) -> AgentSettings:
    return AgentSettings(_env_file=None, **overrides)


def _attach(provider: HttpWebProvider, handler: Any, **kwargs: Any) -> HttpWebProvider:
    """Point a real adapter at a mock transport.

    `max_attempts=1` by default so a retryable status fails on the first
    attempt instead of sleeping through the backoff ladder.
    """
    provider.max_attempts = kwargs.pop("max_attempts", 1)
    provider._client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    return provider


def _text_response(body: str, status: int = 200, content_type: str = "text/html") -> httpx.Response:
    return httpx.Response(status, text=body, headers={"content-type": content_type})


# --------------------------------------------------------------------------
# URL hygiene
# --------------------------------------------------------------------------


def test_require_http_url_keeps_real_pages() -> None:
    assert require_http_url("https://example.com/a") == "https://example.com/a"
    assert require_http_url("  http://example.com  ") == "http://example.com"


@pytest.mark.parametrize(
    "candidate",
    ["", "   ", "javascript:alert(1)", "mailto:a@b.test", "/relative/path", "example.com"],
)
def test_require_http_url_rejects_what_is_not_a_page(candidate: str) -> None:
    # Passing these to httpx produces a confusing transport error instead of an
    # honest "this is not a fetchable page".
    assert require_http_url(candidate) == ""


def test_strip_tags_removes_markup_and_normalises_whitespace() -> None:
    assert strip_tags("<b>Rooftop</b>\n  garden   venue") == "Rooftop garden venue"
    assert strip_tags("") == ""


# --------------------------------------------------------------------------
# HTML extraction
# --------------------------------------------------------------------------

PAGE_HTML = """
<html><head>
  <title>  The Grand  Ballroom </title>
  <style>p { color: red }</style>
  <script>var secret = "never evidence";</script>
</head><body>
  <h1>Grand Ballroom</h1>
  <p>A marble staircase and a long reflecting pool.</p>
  <p>Chandeliers under a glass ceiling.</p>
  <img src="/images/relative.jpg" alt="hall">
  <img data-src="https://cdn.example.com/lazy.jpg">
  <img srcset="https://cdn.example.com/small.jpg 480w, https://cdn.example.com/big.jpg 1280w">
  <img src="https://cdn.example.com/logo.svg">
  <img src="https://cdn.example.com/icon.ico">
  <img src="javascript:void(0)">
  <img src="/images/relative.jpg" alt="duplicate">
  <noscript><img src="https://cdn.example.com/noscript.jpg"></noscript>
</body></html>
"""


def test_extract_page_reads_title_and_visible_text() -> None:
    page = extract_page(PAGE_HTML, "https://venue.example.com/hall")
    assert page.title == "The Grand Ballroom"
    assert "marble staircase" in page.text
    assert "Chandeliers" in page.text


def test_extract_page_never_returns_script_or_style_as_evidence() -> None:
    page = extract_page(PAGE_HTML, "https://venue.example.com/hall")
    assert "never evidence" not in page.text
    assert "color: red" not in page.text


def test_extract_page_keeps_block_boundaries() -> None:
    # Run-together text is how "poolChandeliers" ends up in a signature detail.
    page = extract_page(PAGE_HTML, "https://venue.example.com/hall")
    assert "pool.\nChandeliers" in page.text


def test_extract_page_drops_images_inside_skipped_containers() -> None:
    # `<noscript>` holds print/tracking fallbacks, not venue photography.
    images = extract_page(PAGE_HTML, "https://venue.example.com/hall").images
    assert all("noscript" not in url for url in images)


def test_extract_page_never_reports_the_page_itself_as_an_image() -> None:
    # `urljoin(base, "")` returns the base, so an `<img>` with no `src` used to
    # become an image candidate pointing at the HTML page.
    images = extract_page('<img alt="lazy"><img src="">', "https://venue.example.com/hall").images
    assert images == ()


def test_extract_page_absolutises_and_filters_images() -> None:
    images = extract_page(PAGE_HTML, "https://venue.example.com/hall").images
    assert "https://venue.example.com/images/relative.jpg" in images
    assert "https://cdn.example.com/lazy.jpg" in images  # data-src
    assert "https://cdn.example.com/small.jpg" in images  # srcset
    assert all("logo.svg" not in url and "icon.ico" not in url for url in images)
    assert all(not url.startswith("javascript:") for url in images)
    assert "https://cdn.example.com/noscript.jpg" not in images


def test_extract_page_dedupes_images_in_document_order() -> None:
    images = extract_page(PAGE_HTML, "https://venue.example.com/hall").images
    assert len(images) == len(set(images))
    assert images[0].endswith("relative.jpg")


def test_extract_page_survives_malformed_markup() -> None:
    # Unclosed tags are the norm on the open web; losing the whole page would be
    # worse than losing a tag.
    page = extract_page("<div><p>Colonnade<p>Terrace</div", "https://x.test/")
    assert "Colonnade" in page.text and "Terrace" in page.text


def test_extract_page_caps_text_length() -> None:
    from pigamagic_agent.agent.providers.web_common import MAX_TEXT_CHARS

    page = extract_page("<p>" + ("x" * (MAX_TEXT_CHARS + 5_000)) + "</p>", "https://x.test/")
    assert len(page.text) <= MAX_TEXT_CHARS


# --------------------------------------------------------------------------
# Shared HTTP behaviour
# --------------------------------------------------------------------------


def _plain_provider(**overrides: Any) -> HttpWebProvider:
    """The base adapter itself: it has no `search`, but every other job."""
    provider = HttpWebProvider(settings=_settings(**overrides))
    provider.name = "web-test"  # type: ignore[misc]
    return provider


@pytest.mark.parametrize("status", sorted(DEAD_PAGE_STATUS))
async def test_a_dead_page_is_empty_evidence_not_an_error(status: int) -> None:
    provider = _attach(_plain_provider(), lambda request: httpx.Response(status))

    page = await provider.fetch_page(url="https://venue.example.com/gone")

    assert page.text == "" and page.images == ()


async def test_a_throttled_page_raises_because_it_says_something_about_our_budget() -> None:
    provider = _attach(_plain_provider(), lambda request: httpx.Response(429, text="slow down"))

    with pytest.raises(ProviderError) as excinfo:
        await provider.fetch_page(url="https://venue.example.com/hall")

    assert excinfo.value.retryable is True
    assert excinfo.value.status == 429


async def test_a_pdf_is_not_text_evidence() -> None:
    provider = _attach(
        _plain_provider(),
        lambda request: httpx.Response(200, content=b"%PDF-1.4", headers={"content-type": "application/pdf"}),
    )

    page = await provider.fetch_page(url="https://venue.example.com/brochure.pdf")

    assert page.text == ""


async def test_page_bytes_are_capped() -> None:
    provider = _plain_provider()
    provider.max_page_bytes = 120
    _attach(provider, lambda request: _text_response("<p>" + ("word " * 500) + "</p>", content_type="text/html"))

    page = await provider.fetch_page(url="https://venue.example.com/huge")

    assert 0 < len(page.text) <= 120


async def test_an_unfetchable_url_never_makes_a_request() -> None:
    calls: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(str(request.url))
        return httpx.Response(200)

    provider = _attach(_plain_provider(), handler)

    page = await provider.fetch_page(url="javascript:alert(1)")
    image = await provider.fetch_image(url="mailto:a@b.test")

    assert calls == []
    assert page.text == "" and image.content == b""


async def test_transport_failure_is_reported_as_retryable() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("no route", request=request)

    provider = _attach(_plain_provider(), handler)

    with pytest.raises(ProviderError) as excinfo:
        await provider.fetch_page(url="https://venue.example.com/hall")

    assert excinfo.value.retryable is True


async def test_fetch_image_returns_real_bytes_with_a_sniffed_type() -> None:
    provider = _attach(
        _plain_provider(),
        lambda request: httpx.Response(200, content=PNG, headers={"content-type": "image/png"}),
    )

    image = await provider.fetch_image(url="https://cdn.example.com/hall.png")

    assert image.content == PNG
    assert image.content_type == "image/png"


@pytest.mark.parametrize(
    "response",
    [
        httpx.Response(403),
        httpx.Response(404),
        httpx.Response(200, content=b"not an image at all"),
        httpx.Response(200, content=b""),
        httpx.Response(200, content=b"<html><body>not an image</body></html>"),
    ],
)
async def test_an_unusable_image_is_skipped_not_fatal(response: httpx.Response) -> None:
    provider = _attach(_plain_provider(), lambda request: response)

    image = await provider.fetch_image(url="https://cdn.example.com/hall.png")

    assert image.url == "https://cdn.example.com/hall.png"
    assert image.content == b""


async def test_image_bytes_win_over_a_lying_content_type_header() -> None:
    # CDNs serve real JPEGs as `application/octet-stream` all the time. Sniffing
    # the bytes is the only trustworthy answer.
    provider = _attach(
        _plain_provider(),
        lambda request: httpx.Response(200, content=PNG, headers={"content-type": "text/html"}),
    )

    image = await provider.fetch_image(url="https://cdn.example.com/hall.png")

    assert image.content == PNG and image.content_type == "image/png"


async def test_an_oversized_image_is_skipped() -> None:
    provider = _plain_provider()
    provider.max_image_bytes = 10
    _attach(provider, lambda request: httpx.Response(200, content=PNG, headers={"content-type": "image/png"}))

    assert (await provider.fetch_image(url="https://cdn.example.com/big.png")).content == b""


async def test_offline_mode_refuses_before_any_request() -> None:
    calls: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(str(request.url))
        return httpx.Response(200)

    provider = _attach(_plain_provider(offline=True), handler)

    with pytest.raises(OfflineModeError):
        await provider.fetch_page(url="https://venue.example.com/hall")
    with pytest.raises(OfflineModeError):
        await provider.fetch_image(url="https://cdn.example.com/hall.png")

    assert calls == []


# --------------------------------------------------------------------------
# Web research must never consume the image budget
# --------------------------------------------------------------------------


def test_web_providers_do_not_inherit_the_image_budget() -> None:
    # AGENT_IMAGE_BUDGET_USD is for pixels. A research run that spent it would
    # stop the next photo render from being affordable.
    provider = HttpWebProvider(settings=_settings(image_budget_usd=5.0))
    assert provider.budget is not None
    assert provider.budget.limit_usd == 0.0


def test_a_host_supplied_budget_survives() -> None:
    ledger = BudgetLedger(limit_usd=1.0)
    provider = HttpWebProvider(settings=_settings(image_budget_usd=5.0), budget=ledger)
    assert provider.budget is ledger


async def test_web_estimates_are_free() -> None:
    assert (await HttpWebProvider(settings=_settings()).estimate_cost(calls=3)).usd == 0.0


# --------------------------------------------------------------------------
# Wikipedia — keyless, so the real evidence path is demonstrable for free
# --------------------------------------------------------------------------


def _wikipedia() -> WikipediaWebProvider:
    return WikipediaWebProvider(settings=_settings())


async def test_wikipedia_search_maps_articles_to_real_urls() -> None:
    seen: list[httpx.URL] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request.url)
        return httpx.Response(
            200,
            json={
                "query": {
                    "search": [
                        {
                            "title": "Stanley Hotel, Nairobi",
                            "snippet": '<span class="searchmatch">Hotel</span> in Nairobi',
                        }
                    ]
                }
            },
        )

    hits = await _attach(_wikipedia(), handler).search(query="Stanley Hotel Nairobi", limit=5)

    assert [hit.url for hit in hits] == ["https://en.wikipedia.org/wiki/Stanley_Hotel%2C_Nairobi"]
    assert hits[0].snippet == "Hotel in Nairobi", "markup must not reach the planner"
    params = seen[0].params
    assert params["srnamespace"] == "0", "a File: or Category: hit is not evidence"
    assert params["srlimit"] == "5"


async def test_wikipedia_search_caps_the_limit_at_the_api_maximum() -> None:
    seen: list[httpx.URL] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request.url)
        return httpx.Response(200, json={"query": {"search": []}})

    await _attach(_wikipedia(), handler).search(query="anything", limit=500)

    assert seen[0].params["srlimit"] == "20"


async def test_an_empty_query_makes_no_request() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        raise AssertionError("an empty query must not reach the network")

    assert await _attach(_wikipedia(), handler).search(query="   ") == []


async def test_wikipedia_reads_article_prose_through_the_api() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.params["explaintext"] == "1"
        return httpx.Response(
            200,
            json={
                "query": {
                    "pages": [
                        {
                            "title": "Sarova Stanley",
                            "extract": "The hotel has a heated rooftop swimming pool and a ballroom.",
                            "original": {"source": "https://upload.example.com/stanley.jpg"},
                        }
                    ]
                }
            },
        )

    page = await _attach(_wikipedia(), handler).fetch_page(
        url="https://en.wikipedia.org/wiki/Sarova_Stanley"
    )

    assert page.title == "Sarova Stanley"
    assert "rooftop swimming pool" in page.text
    assert page.images == ("https://upload.example.com/stanley.jpg",)


async def test_a_missing_article_is_empty_evidence() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"query": {"pages": [{"missing": True}]}})

    page = await _attach(_wikipedia(), handler).fetch_page(
        url="https://en.wikipedia.org/wiki/No_Such_Venue"
    )

    assert page.text == "" and page.images == ()


async def test_a_non_wikipedia_url_falls_back_to_html_extraction() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return _text_response("<title>Venue</title><p>A terrace and a lawn.</p>")

    page = await _attach(_wikipedia(), handler).fetch_page(url="https://venue.example.com/about")

    assert "terrace and a lawn" in page.text


@pytest.mark.parametrize(
    ("url", "expected"),
    [
        ("https://en.wikipedia.org/wiki/Sarova_Stanley", "Sarova Stanley"),
        ("https://en.wikipedia.org/wiki/Stanley_Hotel%2C_Nairobi#History", "Stanley Hotel, Nairobi"),
        ("https://en.wikipedia.org/wiki/Main_Page?x=1", "Main Page"),
        ("https://example.com/nope", ""),
    ],
)
def test_wikipedia_title_extraction(url: str, expected: str) -> None:
    assert _wikipedia().title_from_url(url) == expected


# --------------------------------------------------------------------------
# SearXNG — keyless open web, needs an instance
# --------------------------------------------------------------------------


def test_searxng_without_an_instance_says_which_setting_to_set() -> None:
    provider = SearxngWebProvider(settings=_settings())
    with pytest.raises(ProviderNotConfigured) as excinfo:
        provider._base()
    assert "AGENT_WEB_BASE_URL" in str(excinfo.value)


async def test_searxng_parses_results_and_skips_non_pages() -> None:
    provider = SearxngWebProvider(settings=_settings(web_base_url="http://localhost:8080"))

    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.params["format"] == "json", "without this the instance answers HTML"
        return httpx.Response(
            200,
            json={
                "results": [
                    {
                        "url": "https://venue.example.com/hall",
                        "title": "Hall",
                        "content": "A <b>ballroom</b>",
                    },
                    {"url": "javascript:void(0)", "title": "junk"},
                    {"url": "https://guide.example.com/hall", "title": "Guide"},
                ]
            },
        )

    hits = await _attach(provider, handler).search(query="hall venue", limit=5)

    assert [hit.url for hit in hits] == [
        "https://venue.example.com/hall",
        "https://guide.example.com/hall",
    ]
    assert hits[0].snippet == "A ballroom"


async def test_searxng_html_response_names_the_fix() -> None:
    provider = SearxngWebProvider(settings=_settings(web_base_url="http://localhost:8080"))
    _attach(provider, lambda request: _text_response("<html>results</html>"))

    with pytest.raises(ProviderNotConfigured) as excinfo:
        await provider.search(query="hall")

    message = str(excinfo.value)
    assert "search.formats" in message and "json" in message


# --------------------------------------------------------------------------
# Brave — the paid-tier option, keyed
# --------------------------------------------------------------------------


def test_brave_without_a_key_names_the_setting() -> None:
    provider = BraveWebProvider(settings=_settings())
    with pytest.raises(ProviderNotConfigured) as excinfo:
        provider.require_credentials()
    assert "AGENT_WEB_API_KEY" in str(excinfo.value)


async def test_brave_parses_results_and_caps_the_count() -> None:
    seen: list[httpx.URL] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request.url)
        assert request.headers["x-subscription-token"] == "key-123"
        return httpx.Response(
            200,
            json={
                "web": {
                    "results": [
                        {
                            "url": "https://venue.example.com/hall",
                            "title": "<b>Hall</b>",
                            "description": "A colonnade",
                        },
                        {"url": "ftp://venue.example.com/x", "title": "nope"},
                    ]
                }
            },
        )

    provider = BraveWebProvider(api_key="key-123", settings=_settings())
    hits = await _attach(provider, handler).search(query="hall", limit=100)

    assert [hit.url for hit in hits] == ["https://venue.example.com/hall"]
    assert hits[0].title == "Hall"
    assert seen[0].params["count"] == "20", "Brave caps a page at 20"


async def test_brave_reports_its_spend_to_the_ledger() -> None:
    ledger = BudgetLedger(limit_usd=1.0)
    provider = BraveWebProvider(api_key="key-123", settings=_settings(), budget=ledger)
    _attach(provider, lambda request: httpx.Response(200, json={"web": {"results": []}}))

    await provider.search(query="hall")

    assert ledger.spent_usd == pytest.approx(0.005)


async def test_brave_refuses_before_spending_past_the_budget() -> None:
    calls: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(str(request.url))
        return httpx.Response(200, json={"web": {"results": []}})

    ledger = BudgetLedger(limit_usd=0.001)
    provider = BraveWebProvider(api_key="key-123", settings=_settings(), budget=ledger)
    _attach(provider, handler)

    with pytest.raises(BudgetExceeded):
        await provider.search(query="hall")

    assert calls == [], "a refused search must cost nothing"


# --------------------------------------------------------------------------
# Registry and configuration
# --------------------------------------------------------------------------


@pytest.mark.parametrize("name", available_web_providers())
def test_every_web_provider_builds_and_satisfies_the_protocol(name: str) -> None:
    provider = build_web_provider(
        _settings(web_provider=name, web_api_key="test", web_base_url="http://localhost:8080")
    )
    assert isinstance(provider, WebProvider)
    assert provider.name == name


def test_the_default_web_provider_is_offline() -> None:
    # CI and the CLI must never reach the network by accident.
    assert build_web_provider(_settings()).name == "fixture"


@pytest.mark.parametrize(
    ("alias", "expected"),
    [
        ("wiki", "wikipedia"),
        ("searx", "searxng"),
        ("brave-search", "brave"),
        ("offline", "fixture"),
    ],
)
def test_web_provider_aliases(alias: str, expected: str) -> None:
    assert build_web_provider(_settings(), name=alias).name == expected


def test_an_unknown_web_provider_lists_the_options() -> None:
    with pytest.raises(ProviderNotConfigured) as excinfo:
        build_web_provider(_settings(), name="bing")
    message = str(excinfo.value)
    assert "bing" in message
    for name in available_web_providers():
        assert name in message


def test_the_capability_table_tells_the_truth() -> None:
    # A table that claims a provider is keyless while it demands a key is how a
    # setup guide becomes wrong.
    for row in describe_web_providers():
        assert row["name"] in available_web_providers()
        provider = build_web_provider(_settings(), name=row["name"])
        check = getattr(provider, "require_credentials", None)
        if row["needs_key"]:
            assert check is not None, f"{row['name']} claims it needs a key but never checks"
            with pytest.raises(ProviderNotConfigured):
                check()
        elif check is not None:
            check()


def test_missing_credentials_reports_only_what_is_actually_missing() -> None:
    # Wikipedia needs nothing at all; treating "non-fixture" as "needs a key"
    # would report a missing credential for a perfectly valid setup.
    assert _settings(web_provider="wikipedia").missing_credentials() == []
    assert "brave" in " ".join(_settings(web_provider="brave").missing_credentials())
    assert "searxng" in " ".join(_settings(web_provider="searxng").missing_credentials())
    assert (
        _settings(
            web_provider="searxng", web_base_url="http://localhost:8080"
        ).missing_credentials()
        == []
    )
    assert _settings(web_provider="brave", web_api_key="k").missing_credentials() == []


def test_missing_key_message_is_shared_by_every_keyed_adapter() -> None:
    assert "AGENT_WEB_API_KEY" in str(missing_key_error("brave"))


# --------------------------------------------------------------------------
# Research integration: sorting hits, and images the page really offers
# --------------------------------------------------------------------------


def test_rank_hits_drops_results_about_something_else() -> None:
    # A search for a hotel happily returns a beer brand. Those pages are not
    # evidence, and letting them in donates another subject's vocabulary.
    hits = [
        SearchHit(url="https://beer.example.com/tusker", title="Tusker Lager — our brewery"),
        SearchHit(url="https://venue.example.com/stanley", title="The Stanley Hotel Nairobi"),
        SearchHit(url="https://guide.example.com/hall", title="Stanley Hotel review"),
    ]

    ranked = rank_hits("Stanley Hotel, Nairobi", hits)

    assert [hit.url for hit in ranked] == [
        "https://venue.example.com/stanley",
        "https://guide.example.com/hall",
    ]


def test_rank_hits_keeps_everything_when_too_little_looks_relevant() -> None:
    # An unrelated-looking title is not proof of irrelevance; dropping the only
    # real hit would turn a researchable venue into "no evidence".
    hits = [
        SearchHit(url="https://a.example.com/x", title="Wedding venues in Nairobi"),
        SearchHit(url="https://b.example.com/y", title="Tusker Lager — our brewery"),
        SearchHit(url="https://c.example.com/z", title="Reviews"),
    ]

    assert [hit.url for hit in rank_hits("Stanley Hotel, Nairobi", hits)] == [
        "https://a.example.com/x",
        "https://b.example.com/y",
        "https://c.example.com/z",
    ]


def test_rank_hits_is_stable_for_equal_scores() -> None:
    hits = [
        SearchHit(url="https://venue.example.com/b", title="Stanley Hotel second"),
        SearchHit(url="https://venue.example.com/a", title="Stanley Hotel first"),
    ]
    assert [hit.url for hit in rank_hits("Stanley Hotel", hits)] == [hit.url for hit in hits]


class _ImageWebProvider:
    """A provider whose pages expose real image URLs, like a real adapter.

    Records every image URL it is asked for, which is what makes the old
    synthetic `{page}/photo-N.png` candidates observable.
    """

    name = "stub-web"

    def __init__(self, hits: list[SearchHit], pages: dict[str, WebPage]) -> None:
        self._hits = hits
        self._pages = pages
        self.image_calls: list[str] = []

    async def search(self, *, query: str, limit: int = 5) -> list[SearchHit]:
        return self._hits[:limit]

    async def fetch_page(self, *, url: str) -> WebPage:
        return self._pages.get(url, WebPage(url=url, title="", text=""))

    async def fetch_image(self, *, url: str) -> WebImage:
        self.image_calls.append(url)
        return WebImage(url=url, content=PNG, content_type="image/png")


async def test_research_requests_only_images_the_pages_actually_offer() -> None:
    page_url = "https://venue.example.com/stanley"
    web = _ImageWebProvider(
        hits=[SearchHit(url=page_url, title="Stanley Hotel Nairobi")],
        pages={
            page_url: WebPage(
                url=page_url,
                title="Stanley Hotel Nairobi",
                text=(
                    "The Stanley Hotel Nairobi has an atrium lobby with a long reflecting "
                    "pool and an arched colonnade along the east side."
                ),
                images=(
                    "https://cdn.example.com/atrium.jpg",
                    "https://cdn.example.com/colonnade.jpg",
                ),
            )
        },
    )

    outcome = await research_venue(
        "Stanley Hotel, Nairobi",
        "wedding",
        web=web,
        vision=FixtureVisionProvider(),
        settings=_settings(max_images_per_venue=1),
        pack=get_pack("wedding"),
    )

    assert web.image_calls == ["https://cdn.example.com/atrium.jpg"]
    assert outcome.profile.has_research()


# --------------------------------------------------------------------------
# The CLI surface (`python -m pigamagic_agent.agent.cli research`)
# --------------------------------------------------------------------------


def _run_cli(args: list[str], **env: str):
    """Run the real CLI in a subprocess, offline, with no inherited secrets."""
    import os
    import subprocess
    import sys

    environment = dict(os.environ)
    for key in ("AGENT_WEB_PROVIDER", "AGENT_WEB_API_KEY", "AGENT_WEB_BASE_URL"):
        environment.pop(key, None)
    environment.update(env)

    return subprocess.run(
        [sys.executable, "-m", "pigamagic_agent.agent.cli", "research", *args],
        capture_output=True,
        text=True,
        timeout=180,
        env=environment,
    )


def test_cli_research_exits_zero_on_a_researched_venue() -> None:
    result = _run_cli(
        ["--venue", "The Sarova Stanley, Nairobi", "--event", "wedding", "--provider", "fixture", "--quiet"]
    )
    assert result.returncode == 0, result.stderr


def test_cli_research_exits_one_when_nothing_was_found() -> None:
    # "looked and found nothing" is a legitimate, honest outcome — not an error.
    result = _run_cli(
        ["--venue", "Nonexistent Hideout Ballroom", "--event", "wedding", "--provider", "fixture", "--quiet"]
    )
    assert result.returncode == 1, result.stderr


@pytest.mark.parametrize(
    ("provider", "env"),
    [
        ("searxng", {"AGENT_WEB_BASE_URL": ""}),
        ("brave", {"AGENT_WEB_API_KEY": ""}),
    ],
)
def test_cli_research_reports_misconfiguration_as_setup_failure(
    provider: str, env: dict[str, str]
) -> None:
    # Exit 3 means "fix a setting"; exit 2 means "the provider failed". Sending
    # an operator hunting for an outage when a URL is missing wastes their day.
    result = _run_cli(["--venue", "Anywhere", "--provider", provider, "--quiet"], **env)
    assert result.returncode == 3, result.stderr
    assert result.stderr.strip(), "a setup failure must say what to set"


def test_cli_research_refuses_offline() -> None:
    result = _run_cli(
        ["--venue", "The Sarova Stanley, Nairobi", "--provider", "wikipedia", "--quiet"],
        AGENT_OFFLINE="true",
    )
    assert result.returncode == 2
    assert "offline" in result.stderr.casefold()
