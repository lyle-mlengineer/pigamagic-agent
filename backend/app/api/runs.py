"""Runs API: submit a generation run, poll it, and stream it over SSE.

Three endpoints (steps.md Step 13):

- `POST /shoots/{shoot_id}/generate` — reserve credits, enqueue the job, return
  the run. Honours `Idempotency-Key`, so a double-tap on a slow phone returns
  the *same* run instead of charging twice.
- `GET /runs/{run_id}` — the persisted run state, which is what survives a page
  reload.
- `GET /runs/{run_id}/events` — `text/event-stream` over the run's event log,
  with `Last-Event-ID` replay.

The design rule these endpoints exist to honour (design.md §10.1): **the stream
is a notification channel over persisted state, never the transport for
pixels.** Photo bytes live in the blob store; an event carries a URL. That is
why a dropped connection loses nothing: the client re-reads state and replays
from the log.

Two smaller decisions worth stating because they are easy to get wrong:

- **No `sse-starlette`.** SSE is a `text/event-stream` body with `id:`/`event:`/
  `data:` lines and a blank-line terminator. Emitting that directly is ~20 lines
  and keeps the dependency list honest for one endpoint.
- **Replay is exclusive of the cursor.** `Last-Event-ID: 7` means "everything
  from 8". Re-sending 7 would duplicate a feed card.
"""

from __future__ import annotations

import asyncio
import json
from dataclasses import dataclass, field
from typing import Any, Callable, Mapping, Sequence

from fastapi import FastAPI, Header, Query, Request
from fastapi.responses import JSONResponse, StreamingResponse

from ..agent.compile import ShootBrief
from ..agent.config import AgentSettings, get_settings
from ..agent.events import (
    EventLog,
    EventRecorder,
    InMemoryEventLog,
    parse_last_event_id,
    sse_frame,
)
from ..agent.feed import (
    DEFAULT_PAGE_SIZE,
    MAX_PAGE_SIZE,
    FeedError,
    cards_for_outcome,
    page_cards,
    photo_card,
)
from ..agent.lineage import (
    InMemoryPhotoStore,
    LineageError,
    PhotoStore,
    lineage_tree,
    compare_with_parent,
)
from ..agent.hashing import canonical_json
from ..agent.ledger import CreditLedger, InsufficientCredits
from ..agent.pipeline import BriefNotFound, load_brief
from ..agent.purge import FlagRegistry, purge_account as purge_account_rows
from ..agent.providers.base import AgentError
from ..agent.run import (
    GenerationRun,
    InMemoryRunStore,
    RunInProgress,
    RunResult,
    RunStore,
    start_run,
)
from ..agent.worker import InMemoryJobQueue, enqueue_run_job
from .admin import install_admin_routes

#: How long an SSE connection waits for new events before emitting a comment
#: keep-alive. Long enough to be cheap, short enough to hold the socket open
#: through proxies that time out idle connections at 30-60s.
SSE_POLL_SECONDS = 0.5

#: Keep-alive cadence, in polls. 20 * 0.5s = one `: keepalive` every 10s.
SSE_KEEPALIVE_POLLS = 20


# --------------------------------------------------------------------------
# Dependencies
# --------------------------------------------------------------------------


@dataclass
class ApiDependencies:
    """Everything the endpoints need, injected rather than imported.

    Injecting this is what lets the tests drive the API end to end with fixture
    providers and an in-memory queue, and lets the product shell swap in
    Postgres-backed stores without touching the routes.
    """

    ledger: CreditLedger
    store: RunStore
    events: EventLog
    settings: AgentSettings
    #: Built lazily by the caller: a real adapter needs credentials.
    providers: "ProviderBundle"
    queue: InMemoryJobQueue | None = None
    #: Delivered photos, for lineage/compare. Defaults to an in-memory store.
    photos: PhotoStore | None = None
    #: Photo flag state (SR-4). A flagged photo is unreachable publicly.
    flags: FlagRegistry | None = None
    #: Reference analyses, so an account purge deletes derived data too (SR-5).
    analyses: Any = None
    #: Blob bytes are already persisted content-addressed, so a URL is enough.
    media_url_template: str = "/api/v1/media/{key}"
    #: Called with the finished `RunResult` so the shell can persist artifacts.
    on_run_finished: Callable[[RunResult], None] | None = None

    @property
    def event_store(self) -> EventLog:
        return self.events

    @property
    def flag_registry(self) -> FlagRegistry:
        """Flag state, lazily created so a read-only app carries no extra state."""
        if self.flags is None:
            self.flags = FlagRegistry()
        return self.flags

    async def purge(self, account_id: str) -> Any:
        """Delete everything held about an account (SR-5).

        Lives on the dependencies rather than the route so the same call is
        usable from a CLI, a support script or an account-deletion worker — the
        deletion path is too important to exist only behind one HTTP verb.
        """
        return await purge_account_rows(
            account_id,
            ledger=self.ledger,
            runs=self.store,
            events=self.events,
            photos=self.photo_store,
            blobs=self.providers.blobs,
            analyses=self.analyses,
            flags=self.flags,
        )

    @property
    def photo_store(self) -> PhotoStore:
        """Where delivered photos live. Lazily created so the field stays optional."""
        if self.photos is None:
            self.photos = InMemoryPhotoStore()
        return self.photos


@dataclass
class ProviderBundle:
    """The four provider seams, bundled so one object crosses the boundary."""

    llm: Any
    vision: Any
    web: Any
    image: Any
    blobs: Any
    budget: Any = None


def build_dependencies(
    *,
    settings: AgentSettings | None = None,
    ledger: CreditLedger | None = None,
    store: RunStore | None = None,
    events: EventLog | None = None,
    photos: PhotoStore | None = None,
    flags: FlagRegistry | None = None,
    analyses: Any = None,
    providers: ProviderBundle | None = None,
) -> ApiDependencies:
    """Wire the API with sensible defaults; override any part in tests."""
    active = settings or get_settings()
    if providers is None:
        raise ValueError("build_dependencies needs a ProviderBundle (llm/vision/web/image/blobs)")
    return ApiDependencies(
        # `is None`, not `or`: a store that defines `__len__` is falsy when empty,
        # and `x or Default()` would silently discard a caller-supplied store.
        ledger=ledger if ledger is not None else CreditLedger(),
        store=store if store is not None else InMemoryRunStore(),
        events=events if events is not None else InMemoryEventLog(),
        settings=active,
        providers=providers,
        photos=photos if photos is not None else InMemoryPhotoStore(),
        # `or` would be wrong here: `FlagRegistry` defines `__len__`, so an empty
        # registry is falsy and `flags or FlagRegistry()` silently discarded the
        # caller's registry — flags written to it would never be read by the app.
        flags=flags if flags is not None else FlagRegistry(),
        analyses=analyses,
        queue=InMemoryJobQueue(),
    )


# --------------------------------------------------------------------------
# Response shaping
# --------------------------------------------------------------------------


def run_payload(run: GenerationRun, deps: ApiDependencies | None = None) -> dict[str, Any]:
    """The `GET /runs/{id}` body. Additive over `run.as_dict()`.

    `cards` are built by the same `photo_card()` the SSE stream uses, so a
    settled card and a streamed card cannot diverge (steps.md Step 14).
    """
    body = run.as_dict()
    body["events_url"] = f"/api/v1/runs/{run.run_id}/events"
    body["photo_urls"] = {
        photo_id: _media_url(deps, photo_id) for photo_id in run.photo_ids
    }
    if deps is not None:
        # A flagged photo is private everywhere, and the run view is a public
        # route: it must not become the one place a card leaks through (SR-4).
        flagged = deps.flag_registry
        body["cards"] = [
            card
            for card in cached_cards(deps, run.run_id)
            if not flagged.blocked(str(card.get("id") or ""))
        ]
    else:
        body["cards"] = []
    return body


def cached_cards(deps: "ApiDependencies", run_id: str) -> list[dict[str, Any]]:
    """Cards for a run, rebuilt from its `photo` events with `photo_card()`.

    Rebuilt rather than reused on purpose. Returning `event.data["card"]` would
    make the settled and streamed shapes agree *by construction*, so the Step 14
    gate would pass even if the stream had quietly invented its own shape. This
    re-derives the card from the photo fields, which gives the gate two
    independent paths to compare.
    """
    cards: list[dict[str, Any]] = []
    for event in deps.events.events_after(run_id, 0):
        if event.kind != "photo":
            continue
        data = event.data
        cards.append(
            photo_card(
                {
                    "photo_id": data.get("photo_id", ""),
                    "shoot_id": data.get("shoot_id", ""),
                    "node_id": data.get("node_id", ""),
                    "moment": data.get("moment", ""),
                    "activity": data.get("activity", ""),
                    "framing": data.get("framing", ""),
                    "caption": data.get("caption", ""),
                    "blob": {"key": data.get("blob_key", "")},
                    "created_at": data.get("created_at", ""),
                    "parent_id": data.get("parent_id"),
                },
                media_url_template=deps.media_url_template,
                developing=bool(data.get("developing", False)),
            )
        )
    return cards


def _media_url(deps: ApiDependencies | None, key_or_id: str) -> str:
    template = deps.media_url_template if deps is not None else "/api/v1/media/{key}"
    return template.format(key=key_or_id) if key_or_id else ""


def _error(status: int, code: str, message: str, **extra: Any) -> JSONResponse:
    """Uniform error body. A client should never have to parse prose."""
    payload: dict[str, Any] = {"error": {"code": code, "message": message}}
    payload["error"].update(extra)
    return JSONResponse(status_code=status, content=payload)


# --------------------------------------------------------------------------
# Run job
# --------------------------------------------------------------------------


async def execute_run(
    *,
    deps: ApiDependencies,
    brief: ShootBrief,
    account_id: str,
    shot_count: int | None,
    idempotency_key: str,
    cast: Sequence[Any] | None = None,
    reference_images: Mapping[str, Sequence[bytes]] | None = None,
) -> RunResult:
    """Run the pipeline and publish every event to the run's log.

    The recorder is wired to `start_run`'s `on_event` hook, and photos are
    published from the outcome afterwards. Photos are published *before*
    `run_done` so a client that stops at the terminal event has already received
    every card it will ever get — the ordering the gate asserts.
    """
    existing = (
        deps.store.by_idempotency_key(idempotency_key) if idempotency_key else None
    )
    if existing is not None:
        # A replay must not re-render or re-publish; the log already has it.
        return RunResult(run=existing, reservation=deps.ledger.reservation_for_run(existing.run_id))

    run_id_hint = ""
    recorder: EventRecorder | None = None
    #: `run_done` is held back until photos are published, because the gate
    #: requires every `photo` before the terminal event. `start_run` emits it
    #: first (it does not know about photos), so we buffer and flush.
    deferred_done: dict[str, Any] | None = None

    def on_event(kind: str, payload: dict[str, Any]) -> None:
        nonlocal recorder, run_id_hint, deferred_done
        if recorder is None:
            run_id_hint = str(payload.get("run_id") or "")
            if not run_id_hint:
                return
            recorder = EventRecorder(deps.events, run_id_hint)
        if kind == "run_done":
            deferred_done = dict(payload)
            return
        recorder(kind, payload)

    providers = deps.providers
    result = await start_run(
        brief,
        account_id=account_id,
        ledger=deps.ledger,
        store=deps.store,
        llm=providers.llm,
        vision=providers.vision,
        web=providers.web,
        image=providers.image,
        blobs=providers.blobs,
        settings=deps.settings,
        cast=cast,
        reference_images=reference_images,
        shot_count=shot_count,
        idempotency_key=idempotency_key,
        budget=providers.budget,
        on_event=on_event,
    )

    outcome = result.outcome
    recorder = recorder or EventRecorder(deps.events, result.run.run_id)
    if outcome is not None:
        # Register every delivered photo so lineage/compare can find it. Done
        # before publishing so a client that reacts to the `photo` event and
        # immediately asks for lineage never races an empty store.
        for photo in outcome.photos:
            # Idempotent for an identical photo; `put` raises for a conflicting
            # one, which is deliberate and must not be swallowed.
            #
            # The owning account is stamped here, at the API boundary, not in
            # `render_plan`: rendering is a pure plan→bytes operation that has no
            # notion of who asked. Stamping at registration is also what makes
            # the account purge able to find these rows at all (SR-5).
            record = photo.as_dict()
            record["account_id"] = account_id
            record["run_id"] = result.run.run_id
            deps.photo_store.put(record)
        recorder.publish_photos(photo.as_dict() for photo in outcome.photos)
        recorder.publish_failures(outcome.failures)
    done = deferred_done if deferred_done is not None else result.run.as_dict()
    _publish_from_done(recorder, done)
    recorder("run_done", done)

    if deps.on_run_finished is not None:
        deps.on_run_finished(result)
    return result


def _publish_from_done(recorder: EventRecorder, payload: dict[str, Any]) -> None:
    """Publish failures carried on the terminal payload, before the terminal event."""
    failures = payload.get("failures") or []
    if failures:
        recorder.publish_failures(failures)


# --------------------------------------------------------------------------
# SSE
# --------------------------------------------------------------------------


async def stream_events(
    *,
    deps: ApiDependencies,
    run: GenerationRun,
    last_event_id: int,
    poll_seconds: float = SSE_POLL_SECONDS,
    keepalive_polls: int = SSE_KEEPALIVE_POLLS,
    max_idle_polls: int | None = None,
) -> Any:
    """Yield SSE frames for a run until it is terminal and fully drained.

    Termination is the subtle part: the stream ends when the run is terminal
    **and** the log has no unread events. Closing on `run_done` alone would drop
    any event appended in the same tick, and staying open forever would leak a
    connection per client.
    """
    cursor = last_event_id
    idle = 0
    polls = 0
    while True:
        pending = deps.events.events_after(run.run_id, cursor)
        for event in pending:
            cursor = event.seq
            if event.kind == "photo" and deps.flag_registry.blocked(
                str(event.data.get("photo_id") or "")
            ):
                # A flagged photo is private on the live stream too, not only in
                # the settled feed. The event is counted as read so the stream
                # advances rather than replaying it forever (SR-4).
                continue
            yield sse_frame(event=event.kind, data=event.data, event_id=event.seq)
        if pending:
            idle = 0
        else:
            idle += 1
            polls += 1
            if idle % keepalive_polls == 0:
                # A comment line keeps the socket warm through proxies.
                yield ": keepalive\n\n"

        current = deps.store.get(run.run_id) or run
        if current.terminal and not deps.events.events_after(run.run_id, cursor):
            return
        if max_idle_polls is not None and idle >= max_idle_polls:
            return
        if not pending:
            await asyncio.sleep(poll_seconds)


# --------------------------------------------------------------------------
# App factory
# --------------------------------------------------------------------------


def create_app(deps: ApiDependencies | None = None) -> FastAPI:
    """Build the runs API. `deps` is required in practice; the default raises."""
    app = FastAPI(title="PlusOne Runs API", version="v1")
    app.state.deps = deps

    def require_deps() -> ApiDependencies:
        current = getattr(app.state, "deps", None)
        if current is None:
            raise RuntimeError(
                "API dependencies are not configured; call create_app(build_dependencies(...))."
            )
        return current

    @app.post("/api/v1/shoots/{shoot_id}/generate")
    async def generate(
        shoot_id: str,
        request: Request,
        idempotency_key: str | None = Header(default=None, alias="Idempotency-Key"),
    ) -> Any:
        current = require_deps()
        try:
            body = await request.json()
        except (json.JSONDecodeError, UnicodeDecodeError) as exc:
            return _error(400, "invalid_json", f"request body is not valid JSON: {exc}")
        if not isinstance(body, dict):
            return _error(400, "invalid_body", "request body must be a JSON object")

        account_id = str(body.get("account_id") or "").strip()
        if not account_id:
            return _error(422, "missing_account", "account_id is required")

        brief_name = str(body.get("brief") or "").strip()
        try:
            brief = _brief_from_body(body, shoot_id, brief_name)
        except BriefNotFound as exc:
            return _error(404, "brief_not_found", str(exc))
        except ValueError as exc:
            return _error(422, "invalid_brief", str(exc))

        shot_count = body.get("shot_count")
        if shot_count is not None:
            try:
                shot_count = int(shot_count)
            except (TypeError, ValueError):
                return _error(422, "invalid_shot_count", "shot_count must be an integer")
            if shot_count < 1 or shot_count > current.settings.max_shots:
                return _error(
                    422,
                    "shot_count_out_of_range",
                    f"shot_count must be between 1 and {current.settings.max_shots}",
                )

        key = (idempotency_key or str(body.get("idempotency_key") or "")).strip()

        # Pre-flight affordability: a refusal here costs zero provider calls and
        # zero queued work, instead of failing inside the worker.
        wanted = shot_count if shot_count is not None else brief.shot_count
        from ..agent.run import CREDITS_PER_PHOTO

        needed = max(0, wanted) * CREDITS_PER_PHOTO
        if key:
            replay = current.store.by_idempotency_key(key)
            if replay is not None:
                return JSONResponse(status_code=200, content=run_payload(replay, current))
        balance = current.ledger.balance(account_id)
        if balance < needed:
            return _error(
                402,
                "insufficient_credits",
                f"this run needs {needed} credit(s); you have {balance}.",
                required=needed,
                available=balance,
            )

        try:
            result = await execute_run(
                deps=current,
                brief=brief,
                account_id=account_id,
                shot_count=shot_count,
                idempotency_key=key,
            )
        except RunInProgress as exc:
            return _error(409, "run_in_progress", str(exc))
        except InsufficientCredits as exc:
            return _error(
                402,
                "insufficient_credits",
                str(exc),
                required=exc.required,
                available=exc.available,
            )
        except (AgentError, ValueError) as exc:
            return _error(500, "run_failed", str(exc))

        return JSONResponse(status_code=202, content=run_payload(result.run, current))

    @app.get("/api/v1/runs/{run_id}")
    async def get_run(run_id: str) -> Any:
        current = require_deps()
        run = current.store.get(run_id)
        if run is None:
            return _error(404, "run_not_found", f"no run {run_id!r}")
        return JSONResponse(status_code=200, content=run_payload(run, current))

    @app.get("/api/v1/runs/{run_id}/events")
    async def run_events(
        run_id: str,
        request: Request,
        last_event_id: str | None = Header(default=None, alias="Last-Event-ID"),
        last_event_id_query: str | None = Query(default=None, alias="last_event_id"),
    ) -> Any:
        current = require_deps()
        run = current.store.get(run_id)
        if run is None:
            return _error(404, "run_not_found", f"no run {run_id!r}")

        # A browser EventSource cannot set headers on reconnect, so the query
        # parameter is the practical path; the header is the spec path.
        cursor = parse_last_event_id(
            last_event_id if last_event_id is not None else last_event_id_query
        )

        generator = stream_events(deps=current, run=run, last_event_id=cursor)
        return StreamingResponse(
            generator,
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache, no-transform",
                "Connection": "keep-alive",
                # Stops nginx buffering the stream into silence.
                "X-Accel-Buffering": "no",
            },
        )

    @app.get("/api/v1/health")
    async def health() -> Any:
        return {"status": "ok", "events_version": "events-v1"}

    @app.get("/api/v1/photos/{photo_id}/lineage")
    async def get_lineage(photo_id: str) -> Any:
        """Every take in this photo's family, as a tree plus a sibling list.

        FR-7.6: the user compares and deletes takes. A root photo's alternate
        takes are its edits, so both cases return the whole family.
        """
        current = require_deps()
        denied = _blocked(current, photo_id)
        if denied is not None:
            return denied
        try:
            tree = lineage_tree(current.photo_store, photo_id)
        except LineageError as exc:
            return _error(404, "photo_not_found", str(exc))
        return JSONResponse(status_code=200, content=tree)

    @app.get("/api/v1/photos/{photo_id}/compare")
    async def get_compare(photo_id: str) -> Any:
        """Parent vs child: what this edit changed and what it preserved."""
        current = require_deps()
        denied = _blocked(current, photo_id)
        if denied is not None:
            return denied
        try:
            payload = compare_with_parent(current.photo_store, photo_id)
        except LineageError as exc:
            status = 404 if "no photo" in str(exc) else 422
            return _error(status, "compare_unavailable", str(exc))
        return JSONResponse(status_code=200, content=payload)

    @app.get("/api/v1/feed")
    async def get_feed(
        cursor: str | None = None,
        limit: int = DEFAULT_PAGE_SIZE,
    ) -> Any:
        """The settled feed: every card from every finished run, newest first.

        Paginated with an `(created_at, id)` cursor, so a photo inserted by a
        live run cannot shift a page boundary (FR-5.6). The next cursor is in
        the body and in `X-Next-Cursor`, matching the shell's convention.
        """
        current = require_deps()
        limit = max(1, min(int(limit), MAX_PAGE_SIZE))

        all_cards: list[dict[str, Any]] = []
        for run in _all_runs(current):
            all_cards.extend(
                card
                for card in cached_cards(current, run.run_id)
                # A flagged photo is private: it must not reach the feed any
                # more than it reaches its own media URL (SR-4).
                if not current.flag_registry.blocked(str(card.get("id") or card.get("photo_id") or ""))
            )

        try:
            page, next_cursor = page_cards(all_cards, limit=limit, cursor=cursor)
        except FeedError as exc:
            return _error(422, "invalid_cursor", str(exc))

        headers = {"X-Next-Cursor": next_cursor} if next_cursor else {}
        return JSONResponse(
            status_code=200,
            content={
                "items": page,
                "next_cursor": next_cursor,
                "has_more": next_cursor is not None,
            },
            headers=headers,
        )

    install_admin_routes(app, require_deps)

    return app


def _blocked(deps: "ApiDependencies", photo_id: str) -> Any | None:
    """Refuse a photo the safety classifier flagged (SR-4).

    A 404, not a 403: the existence of a flagged photo is itself information
    about a person, so the route behaves exactly as if it were never there.
    """
    if not deps.flag_registry.blocked(photo_id):
        return None
    return _error(404, "photo_not_found", f"no photo {photo_id!r}")


def _all_runs(deps: "ApiDependencies") -> list[GenerationRun]:
    """Every run known to the store.

    The protocol only exposes point lookups, so discovery is delegated to the
    store when it can enumerate and otherwise derived from the event log (every
    run that ever published an event).
    """
    listed = getattr(deps.store, "list_runs", None)
    if callable(listed):
        return list(listed())
    run_ids = getattr(deps.events, "run_ids", None)
    if callable(run_ids):
        return [run for run in (deps.store.get(rid) for rid in run_ids()) if run]
    return []


def _brief_from_body(
    body: Mapping[str, Any], shoot_id: str, brief_name: str
) -> ShootBrief:
    """A golden brief by name, or an inline brief. Inline wins when present."""
    inline = body.get("brief_payload")
    if inline is not None:
        if not isinstance(inline, Mapping):
            raise ValueError("brief_payload must be an object")
        payload = dict(inline)
        payload.setdefault("shoot_id", shoot_id)
        payload["activities"] = tuple(payload.get("activities") or ())
        return ShootBrief(**payload)
    if not brief_name:
        raise ValueError("provide 'brief' (a brief name) or 'brief_payload'")
    brief = load_brief(brief_name)
    if brief.shoot_id != shoot_id:
        # The path is authoritative: a run is always a run *of this shoot*.
        brief = ShootBrief(
            shoot_id=shoot_id,
            title=brief.title,
            event=brief.event,
            venue_name=brief.venue_name,
            activities=brief.activities,
            story=brief.story,
            style_preset=brief.style_preset,
            shot_count=brief.shot_count,
        )
    return brief


def _canonical(payload: Any) -> str:
    return canonical_json(payload)


#: `uvicorn app.api:app` in development, once dependencies are injected.
app: FastAPI | None = None


__all__ = [
    "ApiDependencies",
    "ProviderBundle",
    "SSE_KEEPALIVE_POLLS",
    "SSE_POLL_SECONDS",
    "build_dependencies",
    "cached_cards",
    "create_app",
    "execute_run",
    "run_payload",
    "stream_events",
]
