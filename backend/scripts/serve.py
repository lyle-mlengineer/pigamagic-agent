"""Development server for the PlusOne runs API.

    cd backend && python scripts/serve.py            # http://127.0.0.1:8000
    cd backend && python scripts/serve.py --port 9000 --reload

`create_app()` is a factory: the API needs providers and a credit ledger that a
host is expected to supply (design.md §15). This script is that host for local
work, wired to fixture providers so it runs offline with no credentials and no
cost. Swap the provider bundle here to exercise a real image vendor.

The seed balance exists because a generation run reserves credits *before* it
renders; with a zero balance every call correctly answers 402.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Running `python scripts/serve.py` puts scripts/ on sys.path, not backend/.
BACKEND_ROOT = Path(__file__).resolve().parents[1]
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

DEFAULT_ACCOUNT = "demo"
DEFAULT_CREDITS = 50


def build_app(account_id: str = DEFAULT_ACCOUNT, credits: int = DEFAULT_CREDITS):
    """Return the FastAPI app wired to fixtures, storage and a seeded ledger."""
    from pigamagic_agent.agent.config import get_settings
    from pigamagic_agent.agent.ledger import CreditLedger
    from pigamagic_agent.agent.providers.llm_fixture import FixtureLLMProvider
    from pigamagic_agent.agent.providers.registry import (
        build_image_provider,
        build_web_provider,
    )
    from pigamagic_agent.agent.providers.vision_fixture import FixtureVisionProvider
    from pigamagic_agent.agent.storage import build_blob_store
    from pigamagic_agent.api import build_dependencies, create_app
    from pigamagic_agent.api.runs import ProviderBundle

    settings = get_settings()
    providers = ProviderBundle(
        llm=FixtureLLMProvider(),
        vision=FixtureVisionProvider(),
        web=build_web_provider(settings),
        image=build_image_provider(settings),
        blobs=build_blob_store(settings),
    )
    deps = build_dependencies(
        settings=settings,
        providers=providers,
        ledger=CreditLedger(balances={account_id: credits}),
    )
    return create_app(deps)


def app_factory():
    """`uvicorn --factory scripts.serve:app_factory`."""
    return build_app()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="PlusOne runs API (development)")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--reload", action="store_true", help="restart on file changes")
    parser.add_argument("--account", default=DEFAULT_ACCOUNT, help="seeded account id")
    parser.add_argument("--credits", type=int, default=DEFAULT_CREDITS, help="seed balance")
    args = parser.parse_args(argv)

    import uvicorn

    if args.reload:
        # --reload needs an import string, so it rebuilds the app per process.
        uvicorn.run(
            "scripts.serve:app_factory",
            factory=True,
            host=args.host,
            port=args.port,
            reload=True,
            reload_dirs=[str(BACKEND_ROOT.parent)],
        )
        return 0

    uvicorn.run(build_app(args.account, args.credits), host=args.host, port=args.port)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
