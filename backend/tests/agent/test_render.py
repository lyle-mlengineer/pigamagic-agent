"""Step 9: render pipeline + blob storage (local, S3, Backblaze B2).

Properties that matter:
- content-addressed and immutable: identical bytes never create a second object
- a partial failure does not lose the photos that already succeeded
- an edit creates a CHILD photo; the delivered photo is never mutated
- the budget stops the run instead of overspending
- B2/S3 adapters map our calls onto real SDK calls (verified with a fake client,
  so no network and no credentials are needed)
"""

from __future__ import annotations

import asyncio
import base64
import hashlib
import io
from typing import Any

import pytest

from pigamagic_agent.agent.cli import _demo_venue, demo_cast
from pigamagic_agent.agent.config import AgentSettings
from pigamagic_agent.agent.pipeline import load_brief, run_pipeline
from pigamagic_agent.agent.plan import CameraSpec, ShotSpec
from pigamagic_agent.agent.providers.base import CostEstimate, GeneratedImage, ProviderError
from pigamagic_agent.agent.providers.http_base import BudgetLedger
from pigamagic_agent.agent.providers.image_fixture import FixtureImageProvider, render_fixture_png
from pigamagic_agent.agent.providers.llm_fixture import FixtureLLMProvider
from pigamagic_agent.agent.providers.vision_fixture import FixtureVisionProvider
from pigamagic_agent.agent.providers.web_fixture import FixtureWebProvider
from pigamagic_agent.agent.render import (
    RenderOutcome,
    RenderedPhoto,
    default_seed,
    identity_refs_for,
    render_edit,
    render_plan,
)
from pigamagic_agent.agent.storage import (
    BackblazeB2Store,
    BlobStore,
    LocalBlobStore,
    S3BlobStore,
    StorageError,
    b2_endpoint,
    b2_region_from_endpoint,
    build_blob_store,
    content_key,
)


def _settings(**overrides: Any) -> AgentSettings:
    return AgentSettings(_env_file=None, **overrides)


def _artifact(max_shots: int = 4):
    async def build():
        settings = _settings()
        brief = load_brief("wedding")
        return await run_pipeline(
            brief,
            llm=FixtureLLMProvider(),
            vision=FixtureVisionProvider(),
            web=FixtureWebProvider(),
            settings=settings,
            cast_override=demo_cast("wedding"),
            venue_override=_demo_venue(brief),
        )

    return asyncio.run(build())


# --------------------------------------------------------------------------
# Key building
# --------------------------------------------------------------------------


def test_content_key_is_addressed_by_hash() -> None:
    content = b"some image bytes"
    digest = hashlib.sha256(content).hexdigest()
    key = content_key(content, content_type="image/png")
    assert digest in key
    assert key.endswith(".png")
    # Two-level fan-out keeps directories small.
    assert key.startswith(f"{digest[:2]}/{digest[2:4]}/")


def test_content_key_includes_prefix() -> None:
    key = content_key(b"x", content_type="image/png", prefix="wedding")
    assert key.startswith("wedding/")


def test_identical_content_gives_identical_key() -> None:
    assert content_key(b"abc", content_type="image/png") == content_key(
        b"abc", content_type="image/png"
    )


def test_different_content_gives_different_key() -> None:
    assert content_key(b"abc", content_type="image/png") != content_key(
        b"abd", content_type="image/png"
    )


# --------------------------------------------------------------------------
# Local store
# --------------------------------------------------------------------------


def test_local_store_round_trip(tmp_path) -> None:
    store = LocalBlobStore(root=tmp_path)
    png = render_fixture_png(prompt="a", width=64, height=64)

    stored = asyncio.run(store.put(png, content_type="image/png", prefix="wedding"))
    assert stored.size == len(png)
    assert stored.sha256 == hashlib.sha256(png).hexdigest()
    assert asyncio.run(store.exists(stored.key)) is True
    assert asyncio.run(store.get(stored.key)) == png


def test_local_store_is_immutable(tmp_path) -> None:
    """Writing the same key twice must not create a second file."""
    store = LocalBlobStore(root=tmp_path)
    png = render_fixture_png(prompt="same", width=64, height=64)

    first = asyncio.run(store.put(png, content_type="image/png"))
    second = asyncio.run(store.put(png, content_type="image/png"))

    assert first.key == second.key
    files = list(tmp_path.rglob("*.png"))
    assert len(files) == 1


def test_local_store_never_leaves_partial_files(tmp_path) -> None:
    store = LocalBlobStore(root=tmp_path)
    asyncio.run(store.put(b"payload", content_type="image/png"))
    assert not list(tmp_path.rglob("*.part"))


def test_local_store_rejects_empty_content(tmp_path) -> None:
    store = LocalBlobStore(root=tmp_path)
    with pytest.raises(StorageError):
        asyncio.run(store.put(b"", content_type="image/png"))


def test_local_store_rejects_path_traversal(tmp_path) -> None:
    store = LocalBlobStore(root=tmp_path)
    for key in ("../escape.png", "a/../../escape.png"):
        with pytest.raises(StorageError):
            asyncio.run(store.get(key))


def test_local_store_missing_key_raises(tmp_path) -> None:
    store = LocalBlobStore(root=tmp_path)
    with pytest.raises(StorageError):
        asyncio.run(store.get("aa/bb/nope.png"))


def test_local_store_list_and_delete(tmp_path) -> None:
    store = LocalBlobStore(root=tmp_path)
    stored = asyncio.run(store.put(b"one", content_type="image/png", prefix="p"))
    asyncio.run(store.put(b"two", content_type="image/png", prefix="p"))

    blobs = asyncio.run(store.list())
    assert len(blobs) == 2
    assert asyncio.run(store.delete(stored.key)) is True
    assert asyncio.run(store.exists(stored.key)) is False
    assert len(asyncio.run(store.list())) == 1


def test_local_store_satisfies_the_protocol(tmp_path) -> None:
    assert isinstance(LocalBlobStore(root=tmp_path), BlobStore)


# --------------------------------------------------------------------------
# S3 / Backblaze mapping (fake client: no network, no credentials)
# --------------------------------------------------------------------------


class FakeS3Client:
    """Minimal stand-in for a boto3 S3 client."""

    def __init__(self) -> None:
        self.objects: dict[str, bytes] = {}
        self.calls: list[tuple[str, dict[str, Any]]] = []

    def head_object(self, Bucket: str, Key: str) -> dict[str, Any]:  # noqa: N803
        self.calls.append(("head_object", {"Bucket": Bucket, "Key": Key}))
        if Key not in self.objects:
            raise KeyError(Key)
        return {"ContentLength": len(self.objects[Key])}

    def put_object(self, **kwargs: Any) -> None:  # noqa: N802
        self.calls.append(("put_object", kwargs))
        self.objects[kwargs["Key"]] = kwargs["Body"]

    def get_object(self, Bucket: str, Key: str) -> dict[str, Any]:  # noqa: N803
        self.calls.append(("get_object", {"Bucket": Bucket, "Key": Key}))
        body = io.BytesIO(self.objects[Key])
        return {"Body": body}

    def delete_object(self, Bucket: str, Key: str) -> None:  # noqa: N803
        self.calls.append(("delete_object", {"Bucket": Bucket, "Key": Key}))
        self.objects.pop(Key, None)

    def get_paginator(self, name: str) -> Any:
        class _Paginator:
            def __init__(self, parent: FakeS3Client) -> None:
                self.parent = parent

            def paginate(self, Bucket: str, Prefix: str = "") -> list[dict[str, Any]]:  # noqa: N803
                contents = [
                    {"Key": key, "Size": len(value)}
                    for key, value in sorted(self.parent.objects.items())
                    if key.startswith(Prefix)
                ]
                return [{"Contents": contents}]

        return _Paginator(self)


def test_s3_store_uploads_with_content_type_and_digest() -> None:
    client = FakeS3Client()
    store = S3BlobStore(bucket="bucket", client=client)
    png = render_fixture_png(prompt="a", width=32, height=32)

    stored = asyncio.run(store.put(png, content_type="image/png", prefix="wedding"))

    put = next(call for call in client.calls if call[0] == "put_object")
    assert put[1]["Bucket"] == "bucket"
    assert put[1]["ContentType"] == "image/png"
    assert put[1]["Metadata"]["sha256"] == hashlib.sha256(png).hexdigest()
    assert stored.key.startswith("wedding/")


def test_s3_store_applies_prefix() -> None:
    client = FakeS3Client()
    store = S3BlobStore(bucket="bucket", prefix="plusone", client=client)
    asyncio.run(store.put(b"data", content_type="image/png"))
    key = next(iter(client.objects))
    assert key.startswith("plusone/")


def test_s3_store_skips_upload_when_object_exists() -> None:
    client = FakeS3Client()
    store = S3BlobStore(bucket="bucket", client=client)
    png = render_fixture_png(prompt="a", width=32, height=32)

    asyncio.run(store.put(png, content_type="image/png"))
    puts_before = sum(1 for call in client.calls if call[0] == "put_object")
    asyncio.run(store.put(png, content_type="image/png"))
    puts_after = sum(1 for call in client.calls if call[0] == "put_object")

    assert puts_before == puts_after == 1, "second put must be a no-op"


def test_s3_store_round_trip_and_delete() -> None:
    client = FakeS3Client()
    store = S3BlobStore(bucket="bucket", client=client)
    stored = asyncio.run(store.put(b"payload", content_type="image/png"))

    assert asyncio.run(store.get(stored.key)) == b"payload"
    assert asyncio.run(store.exists(stored.key)) is True
    assert asyncio.run(store.delete(stored.key)) is True
    assert asyncio.run(store.exists(stored.key)) is False


def test_s3_store_list_strips_prefix() -> None:
    client = FakeS3Client()
    store = S3BlobStore(bucket="bucket", prefix="plusone", client=client)
    stored = asyncio.run(store.put(b"payload", content_type="image/png"))

    listed = asyncio.run(store.list())
    assert [item.key for item in listed] == [stored.key]


def test_s3_store_requires_credentials() -> None:
    store = S3BlobStore(bucket="bucket")
    with pytest.raises(StorageError) as info:
        asyncio.run(store.put(b"x", content_type="image/png"))
    assert "access key" in str(info.value).casefold()


def test_s3_store_satisfies_the_protocol() -> None:
    assert isinstance(S3BlobStore(bucket="b", client=FakeS3Client()), BlobStore)


# --------------------------------------------------------------------------
# Backblaze B2 specifics
# --------------------------------------------------------------------------


def test_b2_endpoint_is_derived_from_region() -> None:
    assert b2_endpoint("us-west-004") == "https://s3.us-west-004.backblazeb2.com"


def test_b2_region_can_be_recovered_from_an_endpoint() -> None:
    assert b2_region_from_endpoint("https://s3.us-west-004.backblazeb2.com") == "us-west-004"
    assert b2_region_from_endpoint("eu-central-003") == ""


def test_b2_store_derives_its_endpoint() -> None:
    store = BackblazeB2Store(
        bucket="pigamagic",
        region="us-west-004",
        access_key_id="keyid",
        secret_access_key="appkey",
        client=FakeS3Client(),
    )
    assert store.endpoint_url == "https://s3.us-west-004.backblazeb2.com"
    assert store.addressing_style == "path", "B2 requires path-style addressing"


def test_b2_requires_a_region() -> None:
    with pytest.raises(StorageError) as info:
        BackblazeB2Store(bucket="b", region="")
    assert "region" in str(info.value).casefold()


def test_b2_round_trip_through_a_fake_client() -> None:
    client = FakeS3Client()
    store = BackblazeB2Store(
        bucket="pigamagic",
        region="eu-central-003",
        access_key_id="k",
        secret_access_key="s",
        prefix="plusone",
        client=client,
    )
    png = render_fixture_png(prompt="b2", width=32, height=32)
    stored = asyncio.run(store.put(png, content_type="image/png", prefix="wedding"))

    assert stored.bucket == "pigamagic"
    assert stored.key.startswith("wedding/")
    assert asyncio.run(store.get(stored.key)) == png


def test_b2_registry_builds_from_settings() -> None:
    store = build_blob_store(
        _settings(
            storage_provider="backblaze",
            storage_bucket="pigamagic",
            storage_region="us-west-002",
            storage_access_key_id="keyid",
            storage_secret_access_key="appkey",
        )
    )
    assert isinstance(store, BackblazeB2Store)
    assert store.endpoint_url == "https://s3.us-west-002.backblazeb2.com"


def test_b2_region_can_come_from_a_pasted_endpoint() -> None:
    store = build_blob_store(
        _settings(
            storage_provider="b2",
            storage_bucket="pigamagic",
            storage_endpoint="https://s3.us-west-004.backblazeb2.com",
            storage_access_key_id="k",
            storage_secret_access_key="s",
        )
    )
    assert store.region == "us-west-004"


def test_registry_defaults_to_local(tmp_path) -> None:
    store = build_blob_store(_settings(media_dir=str(tmp_path)))
    assert isinstance(store, LocalBlobStore)


def test_registry_rejects_unknown_provider() -> None:
    with pytest.raises(StorageError):
        build_blob_store(_settings(), name="dropbox")


# --------------------------------------------------------------------------
# Render pipeline
# --------------------------------------------------------------------------


def test_render_writes_one_photo_per_shot(tmp_path) -> None:
    artifact = _artifact()
    store = LocalBlobStore(root=tmp_path)

    outcome = asyncio.run(
        render_plan(
            artifact,
            provider=FixtureImageProvider(settings=_settings()),
            store=store,
            settings=_settings(),
        )
    )

    assert outcome.count == len(artifact.plan.shots)
    assert outcome.complete is True
    assert all(photo.blob.size > 0 for photo in outcome.photos)
    assert len(list(tmp_path.rglob("*.png"))) == outcome.count


def test_render_is_idempotent(tmp_path) -> None:
    """Re-rendering must not duplicate objects; photos are content-addressed."""
    artifact = _artifact()
    store = LocalBlobStore(root=tmp_path)
    provider = FixtureImageProvider(settings=_settings())

    first = asyncio.run(render_plan(artifact, provider=provider, store=store, settings=_settings()))
    second = asyncio.run(render_plan(artifact, provider=provider, store=store, settings=_settings()))

    assert first.digest() == second.digest()
    assert len(list(tmp_path.rglob("*.png"))) == first.count


def test_render_records_lineage_and_provenance(tmp_path) -> None:
    artifact = _artifact()
    outcome = asyncio.run(
        render_plan(
            artifact,
            provider=FixtureImageProvider(settings=_settings()),
            store=LocalBlobStore(root=tmp_path),
            settings=_settings(),
        )
    )
    photo = outcome.photos[0]
    assert photo.shoot_id == artifact.brief.shoot_id
    assert photo.node_id
    assert photo.prompt_hash
    assert photo.prompt_version == "v1"
    assert photo.parent_id is None
    assert photo.created_at


def test_render_seed_is_deterministic() -> None:
    shot = ShotSpec(
        node_id="s00",
        phase="ceremony",
        moment="vows",
        activity="exchanging vows",
        description="d",
        subjects=["p1"],
        framing="medium",
        camera=CameraSpec(),
    )
    assert default_seed(shot, "shoot-1") == default_seed(shot, "shoot-1")
    assert default_seed(shot, "shoot-1") != default_seed(shot, "shoot-2")


def test_render_respects_max_shots(tmp_path) -> None:
    artifact = _artifact()
    outcome = asyncio.run(
        render_plan(
            artifact,
            provider=FixtureImageProvider(settings=_settings()),
            store=LocalBlobStore(root=tmp_path),
            settings=_settings(),
            max_shots=2,
        )
    )
    assert outcome.count == 2


def test_render_can_target_specific_nodes(tmp_path) -> None:
    artifact = _artifact()
    target = artifact.plan.shots[1].node_id
    outcome = asyncio.run(
        render_plan(
            artifact,
            provider=FixtureImageProvider(settings=_settings()),
            store=LocalBlobStore(root=tmp_path),
            settings=_settings(),
            node_ids=[target],
        )
    )
    assert outcome.count == 1
    assert outcome.photos[0].node_id == target


def test_identity_refs_only_include_people_in_frame() -> None:
    artifact = _artifact()
    shot = artifact.plan.shots[0]
    refs = {key: [b"img"] for key in artifact.plan.person_keys()}
    selected = identity_refs_for(shot, refs)

    assert selected is not None
    assert set(selected) <= set(shot.subjects)


def test_identity_refs_returns_none_without_references() -> None:
    artifact = _artifact()
    assert identity_refs_for(artifact.plan.shots[0], None) is None


def _png(width: int = 32, height: int = 32) -> bytes:
    return render_fixture_png(prompt="reference", width=width, height=height)


def test_render_drops_refs_for_a_provider_that_cannot_keep_likeness(tmp_path) -> None:
    """NVIDIA's hosted NIM is text-to-image; a shoot must still render.

    Passing references to it would raise on every shot, so the run would come
    back empty. Dropping them and saying so is the honest outcome: real photos,
    no likeness, and a warning that names it.
    """
    from pigamagic_agent.agent.providers.image_nvidia import NvidiaImageProvider

    artifact = _artifact(max_shots=2)
    refs = {key: [_png()] for key in artifact.plan.person_keys()}
    provider = NvidiaImageProvider(api_key="nvapi-test", settings=_settings())
    seen: list[dict[str, list[bytes]] | None] = []

    async def fake_request(method, url, **kwargs):
        return {"artifacts": [{"base64": base64.b64encode(_png()).decode("ascii")}]}

    provider.request_json = fake_request  # type: ignore[method-assign]
    original_generate = provider.generate

    async def spy_generate(**kwargs):
        seen.append(kwargs.get("identity_refs"))
        return await original_generate(**kwargs)

    provider.generate = spy_generate  # type: ignore[method-assign]

    store = LocalBlobStore(root=tmp_path)
    outcome = asyncio.run(
        render_plan(
            artifact,
            provider=provider,
            store=store,
            settings=_settings(),
            reference_images=refs,
            max_shots=2,
        )
    )

    assert outcome.count == 2
    assert not outcome.failures
    assert all(ref is None for ref in seen)
    assert any("likeness" in warning.casefold() for warning in outcome.warnings)


def test_render_keeps_refs_for_a_provider_that_supports_them(tmp_path) -> None:
    """The same shoot through a capable provider still anchors identity."""
    artifact = _artifact(max_shots=1)
    refs = {key: [_png()] for key in artifact.plan.person_keys()}
    provider = FixtureImageProvider(settings=_settings())
    seen: list[dict[str, list[bytes]] | None] = []

    original_generate = provider.generate

    async def spy_generate(**kwargs):
        seen.append(kwargs.get("identity_refs"))
        return await original_generate(**kwargs)

    provider.generate = spy_generate  # type: ignore[method-assign]

    store = LocalBlobStore(root=tmp_path)
    outcome = asyncio.run(
        render_plan(
            artifact,
            provider=provider,
            store=store,
            settings=_settings(),
            reference_images=refs,
            max_shots=1,
        )
    )

    assert outcome.count == 1
    assert seen and seen[0]
    assert not any("likeness" in warning.casefold() for warning in outcome.warnings)


# --------------------------------------------------------------------------
# Failure handling
# --------------------------------------------------------------------------


class FlakyProvider(FixtureImageProvider):
    """Fails the first N calls, then succeeds."""

    def __init__(self, failures: int = 1, retryable: bool = True) -> None:
        super().__init__(settings=_settings())
        self.remaining = failures
        self.retryable = retryable
        self.calls = 0

    async def generate(self, **kwargs: Any) -> GeneratedImage:
        self.calls += 1
        if self.remaining > 0:
            self.remaining -= 1
            raise ProviderError("transient", provider="flaky", retryable=self.retryable)
        return await super().generate(**kwargs)


class AlwaysFailProvider(FixtureImageProvider):
    async def generate(self, **kwargs: Any) -> GeneratedImage:
        raise ProviderError("always fails", provider="dead")


def test_retryable_failure_is_retried(tmp_path) -> None:
    artifact = _artifact()
    provider = FlakyProvider(failures=1, retryable=True)
    outcome = asyncio.run(
        render_plan(
            artifact,
            provider=provider,
            store=LocalBlobStore(root=tmp_path),
            settings=_settings(),
            max_shots=1,
        )
    )
    assert outcome.count == 1
    assert outcome.photos[0].attempts == 2


def test_non_retryable_failure_is_not_retried(tmp_path) -> None:
    artifact = _artifact()
    provider = FlakyProvider(failures=1, retryable=False)
    outcome = asyncio.run(
        render_plan(
            artifact,
            provider=provider,
            store=LocalBlobStore(root=tmp_path),
            settings=_settings(),
            max_shots=1,
        )
    )
    assert outcome.count == 0
    assert provider.calls == 1


def test_total_failure_still_returns_a_report(tmp_path) -> None:
    artifact = _artifact()
    outcome = asyncio.run(
        render_plan(
            artifact,
            provider=AlwaysFailProvider(settings=_settings()),
            store=LocalBlobStore(root=tmp_path),
            settings=_settings(),
            max_shots=2,
        )
    )
    assert outcome.count == 0
    assert outcome.complete is False
    assert len(outcome.failures) == 2
    assert outcome.as_dict()["failures"]


def test_partial_failure_keeps_successful_photos(tmp_path) -> None:
    """A late failure must not discard photos already delivered."""
    artifact = _artifact()

    class SecondFails(FlakyProvider):
        async def generate(self, **kwargs: Any) -> GeneratedImage:
            self.calls += 1
            if self.calls == 2:
                raise ProviderError("boom", provider="x", retryable=False)
            return await FixtureImageProvider.generate(self, **kwargs)

    outcome = asyncio.run(
        render_plan(
            artifact,
            provider=SecondFails(),
            store=LocalBlobStore(root=tmp_path),
            settings=_settings(),
            max_shots=3,
        )
    )
    assert outcome.count == 2
    assert len(outcome.failures) == 1
    assert outcome.complete is False


def test_budget_stops_the_run(tmp_path) -> None:
    artifact = _artifact()

    class PricyProvider(FixtureImageProvider):
        async def generate(self, **kwargs: Any) -> GeneratedImage:
            image = await super().generate(**kwargs)
            return GeneratedImage(
                content=image.content,
                content_type=image.content_type,
                width=image.width,
                height=image.height,
                seed=image.seed,
                provider="pricy",
                model="pricy-1",
                cost=CostEstimate(credits=10, usd=0.50),
            )

    outcome = asyncio.run(
        render_plan(
            artifact,
            provider=PricyProvider(settings=_settings()),
            store=LocalBlobStore(root=tmp_path),
            settings=_settings(),
            budget=BudgetLedger(limit_usd=0.60),
        )
    )
    assert outcome.count == 1, "only one photo fits the budget"
    assert any("budget" in failure["reason"] for failure in outcome.failures)
    assert outcome.ledger["spent_usd"] == pytest.approx(0.50)


# --------------------------------------------------------------------------
# Lineage: edits create children, never mutate
# --------------------------------------------------------------------------


def test_render_edit_creates_a_child_photo(tmp_path) -> None:
    artifact = _artifact()
    store = LocalBlobStore(root=tmp_path)
    provider = FixtureImageProvider(settings=_settings())

    parent_run = asyncio.run(
        render_plan(artifact, provider=provider, store=store, settings=_settings(), max_shots=1)
    )
    original = parent_run.photos[0]
    original_bytes = asyncio.run(store.get(original.blob.key))

    edit_run = asyncio.run(
        render_edit(
            original,
            artifact,
            instruction="make the lighting warmer",
            provider=provider,
            store=store,
            settings=_settings(),
        )
    )

    assert edit_run.count == 1
    child = edit_run.photos[0]
    assert child.parent_id == original.photo_id
    assert child.node_id == original.node_id
    # The delivered photo must be byte-identical to before the edit.
    assert asyncio.run(store.get(original.blob.key)) == original_bytes


def test_render_outcome_carries_the_artifact_it_rendered(tmp_path) -> None:
    """A host must be able to persist what actually ran, not recompile it.

    Storing the original artifact after an edit would let a second edit build on
    the unedited prompt and silently discard the first one.
    """
    artifact = _artifact()
    store = LocalBlobStore(root=tmp_path)
    provider = FixtureImageProvider(settings=_settings())

    parent_run = asyncio.run(
        render_plan(artifact, provider=provider, store=store, settings=_settings(), max_shots=1)
    )
    assert parent_run.artifact is not None

    edit_run = asyncio.run(
        render_edit(
            parent_run.photos[0],
            artifact,
            instruction="make the lighting warmer",
            provider=provider,
            store=store,
            settings=_settings(),
        )
    )
    assert edit_run.artifact is not None
    scoped = edit_run.artifact.prompts[parent_run.photos[0].node_id]
    assert scoped.metadata.get("edited") is True
    assert "make the lighting warmer" in scoped.prompt
    # The untouched nodes keep their original prompts.
    others = {
        node: prompt.prompt
        for node, prompt in edit_run.artifact.prompts.items()
        if node != parent_run.photos[0].node_id
    }
    for node, prompt in others.items():
        assert prompt == artifact.prompts[node].prompt


def test_two_edits_of_one_original_are_two_photographs(tmp_path) -> None:
    """Takes are siblings, not re-renders: they must not share the parent's seed.

    `default_seed` keys on shoot id + node id, both of which an edit inherits, so
    an edit that simply reused it would produce the *same* pixels as the original
    and every take of a shot would be one image with different captions.
    """
    from pigamagic_agent.agent.render import edit_seed

    artifact = _artifact()
    store = LocalBlobStore(root=tmp_path)
    provider = FixtureImageProvider(settings=_settings())

    parent_run = asyncio.run(
        render_plan(artifact, provider=provider, store=store, settings=_settings(), max_shots=1)
    )
    original = parent_run.photos[0]
    shot = next(s for s in artifact.plan.shots if s.node_id == original.node_id)

    warmer = asyncio.run(
        render_edit(
            original,
            artifact,
            instruction="make it warmer",
            provider=provider,
            store=store,
            settings=_settings(),
        )
    ).photos[0]
    closer = asyncio.run(
        render_edit(
            original,
            artifact,
            instruction="move closer",
            provider=provider,
            store=store,
            settings=_settings(),
        )
    ).photos[0]

    # Different instructions -> different takes of the same node.
    assert warmer.seed != closer.seed
    assert warmer.blob.key != closer.blob.key
    # And neither is a duplicate of the delivered original.
    assert warmer.seed != original.seed
    assert warmer.blob.key != original.blob.key
    assert closer.blob.key != original.blob.key

    # Still deterministic: the same instruction twice reproduces the same take.
    repeat = asyncio.run(
        render_edit(
            original,
            artifact,
            instruction="make it warmer",
            provider=provider,
            store=store,
            settings=_settings(),
        )
    ).photos[0]
    assert repeat.seed == warmer.seed
    assert repeat.blob.key == warmer.blob.key
    # Whitespace/case never forks the seed, or a mistyped retry costs a photo.
    assert edit_seed(shot, artifact.brief.shoot_id, "  Make  it WARMER ") == warmer.seed


def test_editing_an_already_edited_prompt_is_refused(tmp_path) -> None:
    """Two takes of one original must not stack edit instructions."""
    from pigamagic_agent.agent.providers.base import AgentError

    artifact = _artifact()
    store = LocalBlobStore(root=tmp_path)
    provider = FixtureImageProvider(settings=_settings())

    parent_run = asyncio.run(
        render_plan(artifact, provider=provider, store=store, settings=_settings(), max_shots=1)
    )
    original = parent_run.photos[0]
    edit_run = asyncio.run(
        render_edit(
            original,
            artifact,
            instruction="brighter",
            provider=provider,
            store=store,
            settings=_settings(),
        )
    )
    with pytest.raises(AgentError, match="already carries an edit"):
        asyncio.run(
            render_edit(
                original,
                edit_run.artifact,
                instruction="move closer",
                provider=provider,
                store=store,
                settings=_settings(),
            )
        )


def test_edit_instruction_reaches_the_prompt() -> None:
    from pigamagic_agent.agent.prompts import AssembledPrompt
    from pigamagic_agent.agent.render import _apply_instruction

    base = AssembledPrompt(
        prompt="[STYLE] x\n[ACTION] exchanging vows",
        negative_prompt="blurry",
        reference_notes="",
        prompt_version="v1",
        truncated_reference=False,
        estimated_tokens=10,
    )
    edited = _apply_instruction(base, "make it warmer")
    assert "make it warmer" in edited.prompt
    assert edited.metadata["edited"] is True
    # The negative prompt is untouched by a positive instruction.
    assert edited.negative_prompt == "blurry"


def test_empty_edit_instruction_is_rejected() -> None:
    from pigamagic_agent.agent.providers.base import AgentError
    from pigamagic_agent.agent.prompts import AssembledPrompt
    from pigamagic_agent.agent.render import _apply_instruction

    base = AssembledPrompt(
        prompt="[ACTION] x",
        negative_prompt="",
        reference_notes="",
        prompt_version="v1",
        truncated_reference=False,
        estimated_tokens=1,
    )
    with pytest.raises(AgentError):
        _apply_instruction(base, "   ")


def test_render_outcome_serialises_to_json() -> None:
    import json

    artifact = _artifact()
    outcome = asyncio.run(
        render_plan(
            artifact,
            provider=FixtureImageProvider(settings=_settings()),
            store=LocalBlobStore(root="./var/test_render_outcome"),
            settings=_settings(),
            max_shots=1,
        )
    )
    json.dumps(outcome.as_dict())
    import shutil

    shutil.rmtree("./var/test_render_outcome", ignore_errors=True)
