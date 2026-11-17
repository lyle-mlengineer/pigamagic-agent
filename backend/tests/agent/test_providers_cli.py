"""The `providers` command: what the agent will actually use, and what it cannot.

These run the real CLI as a subprocess, because the command's whole job is to
describe the environment the process was started with. An in-process call would
read a cached settings object and could report a stack nobody would get.

Exit codes are the contract:

- 0: every selected surface could run
- 3: at least one selected surface is missing the credential it needs (the
     pipeline would degrade with a warning, which is easy to miss in a log)
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

BACKEND_DIR = Path(__file__).resolve().parents[2]

REQUESTED_STACK = {
    "AGENT_LLM_PROVIDER": "groq",
    "AGENT_VISION_PROVIDER": "nemotron",
    "AGENT_IMAGE_PROVIDER": "nvidia",
    "AGENT_WEB_PROVIDER": "duckduckgo",
}

CREDENTIAL_VARS = (
    "AGENT_LLM_API_KEY",
    "AGENT_VISION_API_KEY",
    "AGENT_IMAGE_API_KEY",
    "AGENT_WEB_API_KEY",
    "AGENT_NVIDIA_API_KEY",
    "AGENT_GROQ_API_KEY",
    "AGENT_SERPER_API_KEY",
)


def _run(*args: str, **overrides: str) -> subprocess.CompletedProcess[str]:
    env = {key: value for key, value in os.environ.items() if not key.startswith("AGENT_")}
    for name in CREDENTIAL_VARS:
        env[name] = ""
    env.update(overrides)
    return subprocess.run(
        [sys.executable, "-m", "pigamagic_agent.agent.cli", "providers", *args],
        cwd=BACKEND_DIR,
        capture_output=True,
        text=True,
        timeout=180,
        env=env,
    )


def test_the_default_stack_is_fixtures_everywhere() -> None:
    result = _run()
    assert result.returncode == 0, result.stderr
    rows = {row["surface"]: row for row in json.loads(_run("--json").stdout)["stack"]}
    assert set(rows) == {"llm", "vision", "image", "web"}
    assert all(row["provider"] == "fixture" for row in rows.values())
    assert all(row["usable"] for row in rows.values())
    assert "surface" in result.stdout


def test_the_requested_stack_is_reported_without_a_key_but_fails_loudly() -> None:
    result = _run(**REQUESTED_STACK)
    assert result.returncode == 3, result.stdout
    assert "flux-3-image-klein" in result.stdout
    assert "openai/gpt-oss-120b" in result.stdout
    assert "nvidia/nemotron-nano-12b-v2-vl" in result.stdout
    assert "duckduckgo" in result.stdout
    # The missing setting is named, so the operator does not have to guess.
    assert "AGENT_LLM_API_KEY" in result.stderr
    assert "AGENT_VISION_API_KEY" in result.stderr
    assert "AGENT_IMAGE_API_KEY" in result.stderr
    # Keyless search is not a fault.
    assert "duckduckgo cannot run" not in result.stderr


def test_a_configured_key_makes_the_stack_usable_and_is_never_printed() -> None:
    result = _run(
        **REQUESTED_STACK,
        AGENT_LLM_API_KEY="gsk-do-not-print-me",
        AGENT_NVIDIA_API_KEY="nvapi-do-not-print-me",
    )
    assert result.returncode == 0, result.stderr
    payload = json.loads(_run("--json", **REQUESTED_STACK, AGENT_LLM_API_KEY="k", AGENT_NVIDIA_API_KEY="k").stdout)
    assert payload["usable"] is True
    assert "do-not-print-me" not in result.stdout
    assert "do-not-print-me" not in result.stderr


def test_serper_is_selectable_and_needs_its_own_key_name() -> None:
    without = _run(AGENT_WEB_PROVIDER="serper")
    assert without.returncode == 3
    assert "AGENT_SERPER_API_KEY" in without.stderr

    with_key = _run(
        "--json", AGENT_WEB_PROVIDER="serper", AGENT_SERPER_API_KEY="serper-key"
    )
    assert with_key.returncode == 0, with_key.stderr
    rows = {row["surface"]: row for row in json.loads(with_key.stdout)["stack"]}
    assert rows["web"]["provider"] == "serper"
    assert rows["web"]["usable"] is True
