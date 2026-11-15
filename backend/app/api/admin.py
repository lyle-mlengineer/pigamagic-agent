"""Admin surface: the safety review console and the account purge (SR-4, SR-5).

Two endpoints, both behind a shared secret:

- `GET  /api/v1/admin/flags`            — the review queue, with plan and prompt
- `POST /api/v1/admin/flags/{photo_id}` — clear or remove one flagged photo
- `GET  /api/v1/admin/flags/{photo_id}` — one flagged photo in full

**The admin routes are disabled when no token is configured.** An unset secret
must never mean "open to all" — that is how a review console leaks every flagged
photo to any client that guesses the path. A missing token is a 503 that says so,
not a 200 with the queue.

The review payload deliberately carries the **plan node and assembled prompt**,
not just the photo and a rule code. A reviewer deciding whether a photo is
acceptable needs the instruction that produced it; a bare image plus "sexual
content" is a rubber stamp, not a review.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Mapping

from fastapi import FastAPI, Header, Query, Request
from fastapi.responses import JSONResponse

from ..agent.purge import FlagRegistry, PurgeError, review_queue

if TYPE_CHECKING:  # pragma: no cover - typing only
    from .runs import ApiDependencies


ADMIN_TOKEN_HEADER = "X-Admin-Token"


def _error(status: int, code: str, message: str, **extra: Any) -> JSONResponse:
    payload: dict[str, Any] = {"error": {"code": code, "message": message}}
    payload["error"].update(extra)
    return JSONResponse(status_code=status, content=payload)


def install_admin_routes(app: FastAPI, require_deps: Any) -> None:
    """Attach the admin routes to the app.

    `require_deps` is passed in rather than imported, so `runs.py` owns the
    dependency wiring and this module owns only the admin policy.
    """

    def _authorise(token: str | None) -> JSONResponse | None:
        deps: ApiDependencies = require_deps()
        configured = deps.settings.admin_token.strip()
        if not configured:
            # Fail closed and say why, so a self-hoster sees the cause.
            return _error(
                503,
                "admin_disabled",
                "the admin console is disabled because AGENT_ADMIN_TOKEN is not set. "
                "Set it to a long random value to enable review.",
            )
        if not token or token.strip() != configured:
            return _error(401, "admin_unauthorised", "a valid admin token is required.")
        return None

    def _registry(deps: ApiDependencies) -> FlagRegistry:
        # Created lazily so an app without an admin console carries no state.
        registry = getattr(deps, "flags", None)
        if registry is None:
            registry = FlagRegistry()
            deps.flags = registry
        return registry

    @app.get("/api/v1/admin/flags")
    async def list_flags(
        request: Request,
        token: str | None = Header(default=None, alias=ADMIN_TOKEN_HEADER),
        account_id: str | None = Query(default=None),
    ) -> Any:
        denied = _authorise(token)
        if denied is not None:
            return denied
        deps = require_deps()
        body = review_queue(_registry(deps), account_id=account_id)
        body["shoot_url_template"] = "/api/v1/shoots/{shoot_id}"
        body["run_url_template"] = "/api/v1/runs/{run_id}"
        return JSONResponse(status_code=200, content=body)

    @app.get("/api/v1/admin/flags/{photo_id}")
    async def get_flag(
        photo_id: str,
        request: Request,
        token: str | None = Header(default=None, alias=ADMIN_TOKEN_HEADER),
    ) -> Any:
        denied = _authorise(token)
        if denied is not None:
            return denied
        deps = require_deps()
        entry = _registry(deps).get(photo_id)
        if entry is None:
            return _error(404, "flag_not_found", f"photo {photo_id!r} was never flagged.")
        from ..agent.purge import review_card

        return JSONResponse(status_code=200, content=review_card(entry))

    @app.post("/api/v1/admin/flags/{photo_id}")
    async def review_flag(
        photo_id: str,
        request: Request,
        token: str | None = Header(default=None, alias=ADMIN_TOKEN_HEADER),
    ) -> Any:
        """Clear or remove a flagged photo. Body: `{"action": "clear"|"remove"}`.

        Removal is irreversible on purpose: a reviewer who decides a photo must
        not exist should not be able to have it reinstated by a later request,
        and the registry refuses the transition with an explanation.
        """
        denied = _authorise(token)
        if denied is not None:
            return denied
        deps = require_deps()

        try:
            raw = await request.json()
        except Exception:  # noqa: BLE001 - any parse failure is a 400
            return _error(400, "invalid_json", "the request body must be JSON.")
        body: Mapping[str, Any] = raw if isinstance(raw, Mapping) else {}
        action = str(body.get("action") or "").strip().lower()
        if action not in ("clear", "remove"):
            return _error(
                422,
                "invalid_action",
                f"{action!r} is not a review action; use 'clear' or 'remove'.",
            )

        reviewer = str(body.get("reviewer") or "admin").strip() or "admin"
        note = str(body.get("note") or "").strip()

        registry = _registry(deps)
        try:
            if action == "clear":
                entry = registry.clear(photo_id, reviewer=reviewer, note=note)
            else:
                entry = registry.remove(photo_id, reviewer=reviewer, note=note)
        except PurgeError as exc:
            return _error(409, "review_refused", str(exc))

        from ..agent.purge import review_card

        return JSONResponse(status_code=200, content=review_card(entry))

    @app.post("/api/v1/admin/purge/{account_id}")
    async def purge_account_route(
        account_id: str,
        request: Request,
        token: str | None = Header(default=None, alias=ADMIN_TOKEN_HEADER),
    ) -> Any:
        """Delete every trace of an account (SR-5).

        Returns the `PurgeReport`, so the caller sees counts rather than a bare
        `{"ok": true}`. A deletion you cannot verify is a deletion you cannot
        claim.
        """
        denied = _authorise(token)
        if denied is not None:
            return denied
        deps = require_deps()

        report = await deps.purge(account_id)
        return JSONResponse(status_code=200, content=report.as_dict())


__all__ = ["ADMIN_TOKEN_HEADER", "install_admin_routes"]
