"""The real chat and search providers: Groq planner, NVIDIA vision, DDG, Serper.

All offline. What these pin, in the order the pipeline uses them:

- the planner sends a *binding* structured-output request and refuses to trust
  chat prose: a reply that is not a JSON object is a retryable failure, and a
  missing required field is never silently `None`
- a reasoning model that burns its budget thinking returns a specific error
  naming the fix, not "empty response"
- Groq's json_schema rejection degrades to json_object **once**, visibly
- the vision adapter sends the image as a base64 data URI (SR-5: a private
  family photo is never uploaded to a public URL) and keeps the appearance
  policy in the system turn, which NIM requires to be a plain string
- DuckDuckGo's redirect wrapper is unwrapped, because passing it to `fetch_page`
  downloads a redirect stub instead of the venue
- a DuckDuckGo bot-challenge page is an error, not "this venue has no web
  presence" — otherwise a rate limit becomes a confidently empty profile
- Serper is a POST with the key in `X-API-KEY` and reports its per-query price
"""

from __future__ import annotations

import base64
import logging
from typing import Any

import httpx
import pytest

from pigamagic_agent.agent.config import AgentSettings
from pigamagic_agent.agent.providers.base import (
    LLMProvider,
    OfflineModeError,
    ProviderError,
    ProviderNotConfigured,
    VisionProvider,
    WebProvider,
)
from pigamagic_agent.agent.providers.chat_json import parse_json_object
from pigamagic_agent.agent.providers.http_base import BudgetLedger
from pigamagic_agent.agent.providers.image_fixture import render_fixture_png
from pigamagic_agent.agent.providers.llm_groq import DEFAULT_MODEL as GROQ_DEFAULT_MODEL
from pigamagic_agent.agent.providers.llm_groq import GroqLLMProvider
from pigamagic_agent.agent.providers.registry import (
    available_llm_providers,
    available_vision_providers,
    available_web_providers,
    build_image_provider,
    build_llm_provider,
    build_vision_provider,
    build_web_provider,
    describe_chat_providers,
    describe_web_providers,
)
from pigamagic_agent.agent.providers.vision_nvidia import (
    ANALYSIS_SCHEMA,
    DEFAULT_MODEL as VISION_DEFAULT_MODEL,
    NvidiaVisionProvider,
)
from pigamagic_agent.agent.providers.web_duckduckgo import (
    DuckDuckGoWebProvider,
    unwrap_redirect,
)
from pigamagic_agent.agent.providers.web_serper import SerperWebProvider

PNG = render_fixture_png(prompt="reference", width=32, height=32)

PLAN_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {"shots": {"type": "array"}, "confidence": {"type": "number"}},
    "required": ["shots"],
}


def _settings(**overrides: Any) -> AgentSettings:
    return AgentSettings(_env_file=None, **overrides)


def _attach(provider: Any, handler: Any) -> Any:
    provider.max_attempts = 1
    provider._client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    return provider


def _chat(content: Any, **extra: Any) -> httpx.Response:
    message: dict[str, Any] = {"role": "assistant", "content": content}
    message.update(extra)
    return httpx.Response(200, json={"choices": [{"message": message}]})


# --------------------------------------------------------------------------
# Groq planner
# --------------------------------------------------------------------------


async def test_groq_requests_binding_structured_output() -> None:
    seen: dict[str, Any] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        import json

        seen["url"] = str(request.url)
        seen["auth"] = request.headers["authorization"]
        seen["body"] = json.loads(request.content)
        return _chat('{"shots": [], "confidence": 0.8}')

    provider = _attach(GroqLLMProvider(api_key="gsk-test", settings=_settings()), handler)
    result = await provider.complete_json(
        system="plan it", user="a wedding", schema=PLAN_SCHEMA
    )

    assert result.provider == "groq"
    assert result.data == {"shots": [], "confidence": 0.8}
    assert seen["url"].startswith("https://api.groq.com/openai/v1/chat/completions")
    assert seen["auth"] == "Bearer gsk-test"
    assert seen["body"]["model"] == GROQ_DEFAULT_MODEL
    assert seen["body"]["response_format"]["type"] == "json_schema"
    assert seen["body"]["response_format"]["json_schema"]["strict"] is True
    # A reasoning model with a small budget returns nothing at all.
    assert seen["body"]["max_completion_tokens"] >= 1024
    assert seen["body"]["reasoning_effort"] == "low"


async def test_groq_falls_back_to_json_object_when_the_schema_is_rejected(caplog) -> None:
    bodies: list[dict[str, Any]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        import json

        body = json.loads(request.content)
        bodies.append(body)
        if body["response_format"]["type"] == "json_schema":
            return httpx.Response(400, json={"error": {"message": "schema unsupported"}})
        return _chat('{"shots": []}')

    provider = _attach(GroqLLMProvider(api_key="gsk-test", settings=_settings()), handler)
    with caplog.at_level(logging.INFO):
        result = await provider.complete_json(system="s", user="u", schema=PLAN_SCHEMA)

    assert result.data == {"shots": []}
    assert [body["response_format"]["type"] for body in bodies] == ["json_schema", "json_object"]
    # Degrading silently is how a planner starts returning prose nobody noticed.
    assert "json_schema" in caplog.text


async def test_groq_does_not_retry_json_object_when_there_is_no_schema() -> None:
    calls: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(str(request.url))
        return httpx.Response(400, json={"error": {"message": "nope"}})

    provider = _attach(GroqLLMProvider(api_key="gsk-test", settings=_settings()), handler)
    with pytest.raises(ProviderError):
        await provider.complete_json(system="s", user="u", schema={})

    assert len(calls) == 1


async def test_groq_rejects_prose_instead_of_parsing_it() -> None:
    provider = _attach(
        GroqLLMProvider(api_key="gsk-test", settings=_settings()),
        lambda request: _chat("Sure! Here is a lovely plan for the wedding."),
    )

    with pytest.raises(ProviderError) as excinfo:
        await provider.complete_json(system="s", user="u", schema={})

    assert excinfo.value.retryable is True
    assert "did not return JSON" in str(excinfo.value)


async def test_a_reasoning_model_that_never_answered_says_why() -> None:
    provider = _attach(
        GroqLLMProvider(api_key="gsk-test", settings=_settings()),
        lambda request: _chat("", reasoning_content="weighing the options…"),
    )

    with pytest.raises(ProviderError) as excinfo:
        await provider.complete_json(system="s", user="u", schema={})

    message = str(excinfo.value)
    assert "reasoning" in message and "token budget" in message
    assert excinfo.value.retryable is False


async def test_a_missing_required_field_is_a_failure_not_a_none() -> None:
    provider = _attach(
        GroqLLMProvider(api_key="gsk-test", settings=_settings()),
        lambda request: _chat('{"confidence": 0.5}'),
    )

    with pytest.raises(ProviderError) as excinfo:
        await provider.complete_json(system="s", user="u", schema=PLAN_SCHEMA)

    assert "shots" in str(excinfo.value)


async def test_undeclared_fields_are_dropped_so_chatter_does_not_break_a_shoot() -> None:
    provider = _attach(
        GroqLLMProvider(api_key="gsk-test", settings=_settings()),
        lambda request: _chat('{"shots": [], "notes": "I hope this helps!"}'),
    )

    result = await provider.complete_json(system="s", user="u", schema=PLAN_SCHEMA)

    assert result.data == {"shots": []}


async def test_groq_fenced_json_is_accepted() -> None:
    provider = _attach(
        GroqLLMProvider(api_key="gsk-test", settings=_settings()),
        lambda request: _chat('```json\n{"shots": [1, 2]}\n```'),
    )

    result = await provider.complete_json(system="s", user="u", schema=PLAN_SCHEMA)

    assert result.data == {"shots": [1, 2]}


def test_groq_without_a_key_names_the_setting() -> None:
    with pytest.raises(ProviderNotConfigured) as excinfo:
        GroqLLMProvider(api_key="", settings=_settings()).require_credentials()
    assert "AGENT_LLM_API_KEY" in str(excinfo.value)


def test_parse_json_object_reports_a_bare_array() -> None:
    with pytest.raises(ProviderError):
        parse_json_object("[1, 2, 3]", provider="test")


# --------------------------------------------------------------------------
# NVIDIA vision
# --------------------------------------------------------------------------


async def test_nvidia_vision_sends_a_base64_data_uri_not_a_public_url() -> None:
    seen: dict[str, Any] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        import json

        seen["auth"] = request.headers["authorization"]
        seen["body"] = json.loads(request.content)
        return _chat('{"people": [{"label": "the bride", "hair": {"style": "braided updo"}}]}')

    provider = _attach(NvidiaVisionProvider(api_key="nvapi-test", settings=_settings()), handler)
    result = await provider.describe_image(image=PNG, instruction="describe", schema={})

    parts = seen["body"]["messages"][-1]["content"]
    image_part = next(part for part in parts if part["type"] == "image_url")
    url = image_part["image_url"]["url"]
    assert url.startswith("data:image/png;base64,")
    assert base64.b64decode(url.split(",", 1)[1]) == PNG
    # SR-5: the reference photo is private; it must never be a hosted URL.
    assert "http" not in url
    assert seen["body"]["model"] == VISION_DEFAULT_MODEL
    assert result.provider == "nemotron"
    assert "braided updo" in result.description


async def test_nvidia_vision_keeps_the_appearance_policy_in_a_string_system_turn() -> None:
    seen: dict[str, Any] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        import json

        seen["body"] = json.loads(request.content)
        return _chat('{"people": []}')

    provider = _attach(NvidiaVisionProvider(api_key="nvapi-test", settings=_settings()), handler)
    await provider.describe_image(image=PNG, instruction="describe", schema={})

    system = seen["body"]["messages"][0]
    assert system["role"] == "system"
    # NIM rejects content parts on non-user roles.
    assert isinstance(system["content"], str)
    assert "identity" in system["content"]
    # And it has no response_format field, so we must not pretend it does.
    assert "response_format" not in seen["body"]


async def test_nvidia_vision_requires_a_people_list() -> None:
    provider = _attach(
        NvidiaVisionProvider(api_key="nvapi-test", settings=_settings()),
        lambda request: _chat('{"scene_quality": "usable"}'),
    )

    with pytest.raises(ProviderError) as excinfo:
        await provider.describe_image(image=PNG, instruction="describe", schema={})

    assert "people" in str(excinfo.value)


async def test_nvidia_vision_refuses_bytes_that_are_not_an_image() -> None:
    provider = NvidiaVisionProvider(api_key="nvapi-test", settings=_settings())
    with pytest.raises(ValueError):
        await provider.describe_image(image=b"not an image", instruction="d", schema={})
    with pytest.raises(ValueError):
        await provider.describe_image(image=b"", instruction="d", schema={})


def test_nvidia_vision_without_a_key_names_the_setting() -> None:
    with pytest.raises(ProviderNotConfigured) as excinfo:
        NvidiaVisionProvider(api_key="", settings=_settings()).require_credentials()
    assert "AGENT_VISION_API_KEY" in str(excinfo.value)


def test_the_default_analysis_schema_asks_for_people() -> None:
    assert ANALYSIS_SCHEMA["required"] == ["people"]


# --------------------------------------------------------------------------
# DuckDuckGo
# --------------------------------------------------------------------------

DDG_HTML = """
<html><body><table>
  <tr><td><a rel="nofollow" class='result-link'
      href="//duckduckgo.com/l/?uddg=https%3A%2F%2Fwww.sarovahotels.com%2Fstanley%2Dnairobi%2F&amp;rut=abc"
      >Sarova Stanley - 5 Star Hotel</a></td></tr>
  <tr><td class='result-snippet'>A <b>colonial</b> hotel with a rooftop pool.</td></tr>
  <tr><td><a rel="nofollow" class='result-link'
      href="https://duckduckgo.com/y.js?ad=1">Sponsored thing</a></td></tr>
  <tr><td class='result-snippet'>Buy now.</td></tr>
  <tr><td><a rel="nofollow" class='result-link'
      href="//duckduckgo.com/l/?uddg=https%3A%2F%2Fguide.example.com%2Fstanley"
      >Stanley Hotel review</a></td></tr>
  <tr><td class='result-snippet'>What to expect.</td></tr>
</table></body></html>
"""


async def test_duckduckgo_unwraps_redirects_and_drops_ads() -> None:
    seen: dict[str, Any] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["q"] = request.url.params.get("q")
        return httpx.Response(200, text=DDG_HTML, headers={"content-type": "text/html"})

    provider = _attach(DuckDuckGoWebProvider(settings=_settings()), handler)
    hits = await provider.search(query="Sarova Stanley Nairobi", limit=5)

    assert seen["q"] == "Sarova Stanley Nairobi"
    assert [hit.url for hit in hits] == [
        "https://www.sarovahotels.com/stanley-nairobi/",
        "https://guide.example.com/stanley",
    ]
    assert hits[0].title == "Sarova Stanley - 5 Star Hotel"
    assert hits[0].snippet == "A colonial hotel with a rooftop pool."


@pytest.mark.parametrize(
    ("href", "expected"),
    [
        ("//duckduckgo.com/l/?uddg=https%3A%2F%2Fa.example.com%2Fx&rut=1", "https://a.example.com/x"),
        ("https://b.example.com/y", "https://b.example.com/y"),
        ("https://duckduckgo.com/y.js?ad=1", ""),
        ("", ""),
        ("javascript:void(0)", ""),
    ],
)
def test_duckduckgo_redirect_unwrapping(href: str, expected: str) -> None:
    assert unwrap_redirect(href) == expected


async def test_duckduckgo_challenge_page_is_an_error_not_an_empty_venue() -> None:
    # "No results" and "we were rate-limited" are different facts. Conflating
    # them yields a confidently empty venue profile.
    provider = _attach(
        DuckDuckGoWebProvider(settings=_settings()),
        lambda request: httpx.Response(202, text="<html>anomaly detected</html>"),
    )

    with pytest.raises(ProviderError) as excinfo:
        await provider.search(query="anywhere")

    assert excinfo.value.retryable is True
    assert "challenge" in str(excinfo.value)


async def test_duckduckgo_genuinely_empty_results_are_empty() -> None:
    provider = _attach(
        DuckDuckGoWebProvider(settings=_settings()),
        lambda request: httpx.Response(200, text="<html><body>No results.</body></html>"),
    )

    assert await provider.search(query="nothing at all") == []


def test_duckduckgo_needs_no_credentials() -> None:
    DuckDuckGoWebProvider(settings=_settings()).require_credentials()


# --------------------------------------------------------------------------
# Serper
# --------------------------------------------------------------------------

SERPER_JSON = {
    "organic": [
        {"title": "Sarova <b>Stanley</b>", "link": "https://www.sarovahotels.com/stanley", "snippet": "Colonial hotel"},
        {"title": "No link", "link": "javascript:void(0)"},
        {"title": "Review", "link": "https://guide.example.com/stanley", "snippet": "What to expect"},
    ]
}


def test_serper_without_a_key_names_the_setting() -> None:
    with pytest.raises(ProviderNotConfigured) as excinfo:
        SerperWebProvider(settings=_settings()).require_credentials()
    assert "AGENT_WEB_API_KEY" in str(excinfo.value)


async def test_serper_posts_with_the_key_in_the_header() -> None:
    seen: dict[str, Any] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        import json

        seen["method"] = request.method
        seen["url"] = str(request.url)
        seen["key"] = request.headers["x-api-key"]
        seen["body"] = json.loads(request.content)
        return httpx.Response(200, json=SERPER_JSON)

    provider = _attach(SerperWebProvider(api_key="serper-test", settings=_settings()), handler)
    hits = await provider.search(query="Sarova Stanley", limit=5)

    assert seen["method"] == "POST"
    assert seen["url"].startswith("https://google.serper.dev/search")
    assert seen["key"] == "serper-test"
    assert seen["body"]["q"] == "Sarova Stanley"
    assert seen["body"]["num"] == 5
    assert [hit.url for hit in hits] == [
        "https://www.sarovahotels.com/stanley",
        "https://guide.example.com/stanley",
    ]
    assert hits[0].title == "Sarova Stanley"


async def test_serper_caps_num_at_the_api_maximum() -> None:
    seen: dict[str, Any] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        import json

        seen.update(json.loads(request.content))
        return httpx.Response(200, json={"organic": []})

    provider = _attach(SerperWebProvider(api_key="k", settings=_settings()), handler)
    await provider.search(query="x", limit=500)

    assert seen["num"] == 100


async def test_serper_tolerates_a_missing_organic_key() -> None:
    provider = _attach(
        SerperWebProvider(api_key="k", settings=_settings()),
        lambda request: httpx.Response(200, json={"searchParameters": {}}),
    )

    assert await provider.search(query="x") == []


async def test_serper_reports_its_spend() -> None:
    ledger = BudgetLedger(limit_usd=1.0)
    provider = SerperWebProvider(api_key="k", settings=_settings(), budget=ledger)
    _attach(provider, lambda request: httpx.Response(200, json={"organic": []}))

    await provider.search(query="x")

    assert ledger.spent_usd == pytest.approx(0.001)


# --------------------------------------------------------------------------
# Registry wiring: the configured names are the ones actually built
# --------------------------------------------------------------------------


@pytest.mark.parametrize("name", available_llm_providers())
def test_every_llm_provider_builds_and_satisfies_the_protocol(name: str) -> None:
    provider = build_llm_provider(_settings(llm_provider=name, llm_api_key="k"))
    assert isinstance(provider, LLMProvider)
    assert provider.name == name


@pytest.mark.parametrize("name", available_vision_providers())
def test_every_vision_provider_builds_and_satisfies_the_protocol(name: str) -> None:
    provider = build_vision_provider(_settings(vision_provider=name, vision_api_key="k"))
    assert isinstance(provider, VisionProvider)
    assert provider.name == name


@pytest.mark.parametrize("name", available_web_providers())
def test_every_web_provider_builds_and_satisfies_the_protocol(name: str) -> None:
    provider = build_web_provider(
        _settings(web_provider=name, web_api_key="k", web_base_url="http://localhost:8080")
    )
    assert isinstance(provider, WebProvider)
    assert provider.name == name


def test_the_requested_stack_is_what_the_config_selects() -> None:
    """image=nvidia, vision=nemotron, llm=groq, web=duckduckgo (as configured)."""
    settings = _settings(
        llm_provider="groq",
        vision_provider="nemotron",
        image_provider="nvidia",
        web_provider="duckduckgo",
        llm_api_key="gsk",
        nvidia_api_key="nvapi",
    )

    llm = build_llm_provider(settings)
    vision = build_vision_provider(settings)
    image = build_image_provider(settings)

    assert llm.name == "groq" and llm.model == GROQ_DEFAULT_MODEL
    assert vision.name == "nemotron" and vision.model == VISION_DEFAULT_MODEL
    assert image.name == "nvidia"
    assert build_web_provider(settings).name == "duckduckgo"
    # One NVIDIA key covers both NIM surfaces; the planner key is separate.
    assert image.api_key == "nvapi" and vision.api_key == "nvapi"
    assert settings.missing_credentials() == []


def test_registry_aliases() -> None:
    assert build_llm_provider(_settings(), name="gpt-oss").name == "groq"
    assert build_vision_provider(_settings(), name="nvidia").name == "nemotron"
    assert build_web_provider(_settings(), name="ddg").name == "duckduckgo"


def test_a_blank_model_setting_keeps_the_pinned_default() -> None:
    # Passing "" straight through overrode the adapter's pin, so every request
    # went out with an empty model name.
    assert (
        build_llm_provider(_settings(llm_provider="groq", llm_api_key="k")).model
        == GROQ_DEFAULT_MODEL
    )
    assert (
        build_vision_provider(_settings(vision_provider="nemotron", vision_api_key="k")).model
        == VISION_DEFAULT_MODEL
    )


def test_an_explicit_model_setting_wins() -> None:
    llm = build_llm_provider(
        _settings(llm_provider="groq", llm_api_key="k", llm_model="llama-3.3-70b-versatile")
    )
    vision = build_vision_provider(
        _settings(vision_provider="nemotron", vision_api_key="k", vision_model="nvidia/other-vl")
    )
    assert llm.model == "llama-3.3-70b-versatile"
    assert vision.model == "nvidia/other-vl"


def test_serper_accepts_its_own_key_name() -> None:
    settings = _settings(web_provider="serper", serper_api_key="serper-env-name")
    assert settings.missing_credentials() == []
    assert build_web_provider(settings).api_key == "serper-env-name"


def test_unknown_names_list_the_options() -> None:
    with pytest.raises(ProviderNotConfigured):
        build_llm_provider(_settings(), name="gpt-5")
    with pytest.raises(ProviderNotConfigured):
        build_vision_provider(_settings(), name="clip")
    with pytest.raises(ProviderNotConfigured):
        build_web_provider(_settings(), name="bing")


def test_missing_credentials_understand_the_aliases() -> None:
    assert _settings(llm_provider="groq").missing_credentials() == ["groq (key missing)"]
    assert _settings(llm_provider="groq", groq_api_key="k").missing_credentials() == []
    assert _settings(vision_provider="nemotron").missing_credentials() == ["nemotron (key missing)"]
    assert _settings(vision_provider="nemotron", nvidia_api_key="k").missing_credentials() == []
    assert _settings(web_provider="serper").missing_credentials() == ["serper (key missing)"]
    assert _settings(web_provider="duckduckgo").missing_credentials() == []


def test_the_capability_tables_only_name_real_providers() -> None:
    web_rows = describe_web_providers()
    assert {row["name"] for row in web_rows} == set(available_web_providers())

    chat_rows = describe_chat_providers()
    assert {row["name"] for row in chat_rows if row["kind"] == "llm"} == set(
        available_llm_providers()
    )
    assert {row["name"] for row in chat_rows if row["kind"] == "vision"} == set(
        available_vision_providers()
    )


# --------------------------------------------------------------------------
# Misconfiguration must degrade, not crash (found while wiring real providers)
# --------------------------------------------------------------------------


def test_misconfiguration_is_a_provider_error_with_no_retry() -> None:
    # While ProviderNotConfigured lived outside the ProviderError hierarchy,
    # every degrade-gracefully ladder missed it and the process died.
    error = ProviderNotConfigured("no key")
    assert isinstance(error, ProviderError)
    assert error.retryable is False


async def test_a_keyless_planner_degrades_to_a_procedural_plan() -> None:
    from pigamagic_agent.agent.compile import compile_plan
    from pigamagic_agent.agent.conventions import get_pack
    from pigamagic_agent.agent.pipeline import load_brief
    from pigamagic_agent.agent.research import unreachable_venue_outcome

    brief = load_brief("wedding")
    pack = get_pack(brief.event)
    offline = _settings(offline=False)
    compile_result = await compile_plan(
        brief,
        cast=[],
        venue=unreachable_venue_outcome(brief.venue_name, brief.event, pack, reason="no venue").profile,
        # No key, no network: `require_credentials` fails before any request.
        llm=GroqLLMProvider(api_key="", settings=offline),
        pack=pack,
        settings=offline,
    )

    assert compile_result.plan.shots, "a keyless planner must still yield shots"
    assert compile_result.used_fallback is True
    assert any("groq needs an API key" in warning for warning in compile_result.plan.warnings)


async def test_a_keyless_vision_and_web_provider_still_produce_an_artifact() -> None:
    from pigamagic_agent.agent.pipeline import load_brief, run_pipeline
    from pigamagic_agent.agent.analyze import ReferenceImage
    from pigamagic_agent.agent.providers.llm_fixture import FixtureLLMProvider

    class _OfflineWeb:
        name = "offline-web"

        async def search(self, *, query: str, limit: int = 5) -> list[Any]:
            raise OfflineModeError("outbound HTTP is disabled (offline mode)")

        async def fetch_page(self, *, url: str) -> Any:
            raise OfflineModeError("outbound HTTP is disabled (offline mode)")

        async def fetch_image(self, *, url: str) -> Any:
            raise OfflineModeError("outbound HTTP is disabled (offline mode)")

    brief = load_brief("wedding")
    artifact = await run_pipeline(
        brief,
        llm=FixtureLLMProvider(),
        vision=NvidiaVisionProvider(api_key="", settings=_settings(offline=True)),
        web=_OfflineWeb(),
        settings=_settings(offline=True),
        references=[ReferenceImage(media_id="med_1", content=PNG, description="the couple")],
    )

    joined = " ".join(artifact.warnings)
    assert artifact.plan.shots
    assert "could not be analysed" in joined
    assert "Venue appearance is estimated" in joined
