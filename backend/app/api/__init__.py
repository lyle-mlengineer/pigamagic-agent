"""HTTP surface for the agent layer.

`app.api.runs` is the only module here today: it exposes
`POST /shoots/{shoot_id}/generate`, `GET /runs/{run_id}` and
`GET /runs/{run_id}/events` (SSE).

Import `create_app()` to get a wired app; the module-level `app` exists so
`uvicorn app.api:app` works in development.
"""

from .runs import build_dependencies, create_app

__all__ = ["build_dependencies", "create_app"]
