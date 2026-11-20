"""The Colab entry point must work both as a script and inside a notebook.

`!python scripts/colab_smoke.py` runs in its own process and `asyncio.run()` is
fine there. `%run scripts/colab_smoke.py` executes inside the Colab kernel, which
already has a running asyncio loop, and a bare `asyncio.run()` raises
"asyncio.run() cannot be called from a running event loop". That is the reported
2026 bug these tests pin down.
"""

from __future__ import annotations

import asyncio
import importlib.util
import sys
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[2] / "scripts" / "colab_smoke.py"


def _load_script():
    """Import colab_smoke.py as a module without executing __main__."""
    spec = importlib.util.spec_from_file_location("colab_smoke_under_test", SCRIPT)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture()
def script(monkeypatch):
    module = _load_script()
    calls: list[str] = []

    async def _main() -> int:
        calls.append("main")
        return 0

    monkeypatch.setattr(module, "main", _main)
    monkeypatch.setattr(module, "check_environment", lambda: calls.append("check"))
    module.calls = calls
    return module


def test_pick_model_honours_plusone_model(script, monkeypatch) -> None:
    monkeypatch.setenv("PLUSONE_MODEL", "stabilityai/sdxl-turbo")
    assert script.pick_model() == "stabilityai/sdxl-turbo"


def test_pick_model_defaults_to_the_ungated_checkpoint(script, monkeypatch) -> None:
    monkeypatch.delenv("PLUSONE_MODEL", raising=False)
    from pigamagic_agent.agent.providers.image_huggingface import DEFAULT_MODEL, is_gated

    assert script.pick_model() == DEFAULT_MODEL
    assert is_gated(DEFAULT_MODEL) is False


def test_run_works_with_no_event_loop(script) -> None:
    assert script.run() == 0
    assert script.calls == ["main"]


def test_run_works_inside_a_running_event_loop(script) -> None:
    """The `%run` path: Jupyter owns the loop, so we must not start another."""
    pytest.importorskip("nest_asyncio")

    async def from_inside_a_loop() -> int:
        return script.run()

    assert asyncio.run(from_inside_a_loop()) == 0
    assert script.calls == ["main"]


def test_run_falls_back_to_a_subprocess_without_nest_asyncio(
    script, monkeypatch
) -> None:
    """A bare kernel without nest_asyncio re-executes instead of crashing."""
    import subprocess

    monkeypatch.setitem(sys.modules, "nest_asyncio", None)  # makes the import fail
    seen: dict[str, object] = {}

    def _call(argv, **kwargs):
        seen["argv"] = argv
        return 0

    monkeypatch.setattr(subprocess, "call", _call)

    async def from_inside_a_loop() -> int:
        return script.run()

    assert asyncio.run(from_inside_a_loop()) == 0
    assert str(SCRIPT) in seen["argv"]
    assert script.calls == []  # the child process runs main, not this one
