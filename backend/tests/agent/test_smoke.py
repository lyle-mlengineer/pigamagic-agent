"""Step 0.1 smoke tests: the agent package imports and is versioned."""

import subprocess
import sys
from pathlib import Path

import pigamagic_agent.agent as agent


def test_agent_package_imports() -> None:
    assert agent.__version__ == "0.1.0"
    assert agent.PROMPT_VERSION == "v1"


def test_agent_imports_in_a_fresh_process() -> None:
    """Guards against import-time side effects (credentials, network, DB)."""
    backend_dir = Path(__file__).resolve().parents[2]
    result = subprocess.run(
        [sys.executable, "-c", "import pigamagic_agent.agent; print(pigamagic_agent.agent.PROMPT_VERSION)"],
        cwd=backend_dir,
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "v1"
