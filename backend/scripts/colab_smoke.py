# PlusOne — Colab smoke test for local Hugging Face image models
#
# Paste this into a Colab notebook (GPU runtime) to generate REAL photos from a
# golden brief using open-weight models. No API key and no licence click: the
# default checkpoint is ungated. ~32 GB of weights, so allow for the download.
#
# Runtime > Change runtime type > T4 GPU
#
# Cell 1: install (Qwen-Image-2.1 needs diffusers main and transformers>=5.17)
#   !pip install -q git+https://github.com/huggingface/diffusers.git "transformers>=5.17" accelerate safetensors
#
# Cell 2: clone/upload this repo, then run this file.
#
# ── Which model to use ──────────────────────────────────────────────────────
#   Qwen-Image-2.1-Turbo  ungated, 8 steps, text-to-image AND editing from
#                         reference photos. The default.
#   Qwen-Image-Edit-2509  ungated, documented facial-identity retention.
#   sdxl-turbo            ungated, ~2.6 GB, 2 steps: the 2-minute plumbing check.
#   FLUX.1-schnell        4 steps, fastest text-to-image — but gated since 2026:
#                         accept the licence on its HF page, set HF_TOKEN, and
#                         pass PLUSONE_MODEL=black-forest-labs/FLUX.1-schnell.
#   FLUX.1-dev / Kontext  gated too: licence click plus HF_TOKEN.
# ────────────────────────────────────────────────────────────────────────────

from __future__ import annotations

import asyncio
import os
import sys
import time

# Make the package importable from any working directory: running a file inside
# scripts/ otherwise puts *that* folder on sys.path, not backend/, and the
# support for PLUSONE_MODEL/PATH then fails with ModuleNotFoundError.
_BACKEND = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _BACKEND not in sys.path:
    sys.path.insert(0, _BACKEND)

# --- configuration ---------------------------------------------------------

# Gated models need a Hugging Face token: https://huggingface.co/settings/tokens
HF_TOKEN = os.environ.get("HF_TOKEN", "")


def pick_model() -> str:
    """The checkpoint to load, honouring PLUSONE_MODEL.

    The default is ungated, so a fresh Colab needs no account at all. Gated
    checkpoints still work, but only with a licence click and `HF_TOKEN` — the
    provider's preflight explains that before anything downloads.
    """
    chosen = os.environ.get("PLUSONE_MODEL", "").strip()
    if chosen:
        return chosen

    from pigamagic_agent.agent.providers.image_huggingface import DEFAULT_MODEL

    return DEFAULT_MODEL

MODEL = pick_model()

BRIEF = "wedding"
OUTPUT_DIR = "out"
MAX_SHOTS = 4          # keep the first run short; a full plan is 10+ shots
WIDTH = HEIGHT = 1024  # 512x512 is much faster for a first check

# Optional: real reference photos of the people. Put files next to the notebook
# or mount Drive. Leave empty to run text-to-image only.
REFERENCE_PATHS: list[str] = []


def check_environment() -> None:
    """Fail early and clearly, instead of mid-generation."""
    try:
        import torch
    except ImportError:
        sys.exit(
            "torch is missing. Run:\n"
            "  !pip install -q git+https://github.com/huggingface/diffusers.git "
            "'transformers>=5.17' accelerate safetensors"
        )

    if not torch.cuda.is_available():
        print("WARNING: no GPU detected. Generation will be very slow on CPU.")
    else:
        name = torch.cuda.get_device_name(0)
        total = torch.cuda.get_device_properties(0).total_memory / 1024**3
        print(f"GPU: {name} ({total:.1f} GB)")

    try:
        import diffusers  # noqa: F401
    except ImportError:
        sys.exit(
            "diffusers is missing. Run:\n"
            "  !pip install -q git+https://github.com/huggingface/diffusers.git "
            "'transformers>=5.17' accelerate safetensors"
        )

    # Qwen-Image-2.1 ships a QwenImage21Pipeline class that only exists in
    # diffusers main (PR #14950), so an older release build loads nothing.
    if MODEL.startswith("Qwen/Qwen-Image-2.1") and not hasattr(
        diffusers, "QwenImage21Pipeline"
    ):
        sys.exit(
            f"{MODEL} needs diffusers with QwenImage21Pipeline. Run:\n"
            "  !pip install -q git+https://github.com/huggingface/diffusers.git "
            "'transformers>=5.17'\n"
            "then restart the runtime."
        )

    # Gating is the one failure that costs a download first, so check it here.
    from pigamagic_agent.agent.providers.image_huggingface import (
        is_gated,
        ungated_alternatives,
    )

    if is_gated(MODEL) and not HF_TOKEN:
        print(
            f"\n{MODEL} is gated: accept its licence at "
            f"https://huggingface.co/{MODEL} and set HF_TOKEN, or re-run with\n"
            f"  PLUSONE_MODEL={ungated_alternatives(MODEL)[0]}  (ungated, no token)\n"
        )
        sys.exit(1)


async def main() -> int:
    from pigamagic_agent.agent.cli import _demo_venue, demo_cast
    from pigamagic_agent.agent.config import AgentSettings
    from pigamagic_agent.agent.pipeline import load_brief, run_pipeline
    from pigamagic_agent.agent.providers.image_huggingface import HuggingFaceImageProvider
    from pigamagic_agent.agent.providers.llm_fixture import FixtureLLMProvider
    from pigamagic_agent.agent.providers.vision_fixture import FixtureVisionProvider
    from pigamagic_agent.agent.providers.web_fixture import FixtureWebProvider
    from pigamagic_agent.agent.providers.registry import describe_image_providers

    settings = AgentSettings(
        _env_file=None,
        image_provider="huggingface",
        image_model=MODEL,
        image_api_key=HF_TOKEN,
        offline=False,
    )

    print("\n=== available image providers ===")
    for row in describe_image_providers(settings):
        flag = "identity" if row["identity_refs"] else "text-only"
        print(f"  {row['name']:<12} {flag:<10} refs={row['max_refs']:<3} {row['note'][:52]}")

    provider = HuggingFaceImageProvider(
        model=MODEL, api_key=HF_TOKEN, settings=settings
    )
    print(
        f"\nmodel      : {MODEL}\n"
        f"identity   : {provider.capabilities.supports_identity_refs} "
        f"(max {provider.capabilities.max_identity_refs} refs)\n"
        f"cost       : $0.00 (your own GPU)\n"
    )

    # 1. Plan the shoot using the offline fixtures for the *thinking* half.
    brief = load_brief(BRIEF)
    print(f"planning '{brief.title}' ({brief.event}, {brief.shot_count} shots)...")
    artifact = await run_pipeline(
        brief,
        llm=FixtureLLMProvider(),
        vision=FixtureVisionProvider(),
        web=FixtureWebProvider(),
        settings=settings,
        cast_override=demo_cast(brief.event),
        venue_override=_demo_venue(brief),
    )
    print(f"plan: {len(artifact.plan.shots)} shots, confidence {artifact.plan.confidence:.2f}")

    # 2. Attach reference photos if provided.
    identity_refs: dict[str, list[bytes]] = {}
    if REFERENCE_PATHS:
        keys = [entry.person_key for entry in artifact.plan.cast]
        for index, path in enumerate(REFERENCE_PATHS):
            if not os.path.exists(path):
                print(f"  ! missing reference: {path}")
                continue
            with open(path, "rb") as handle:
                identity_refs.setdefault(keys[index % len(keys)], []).append(handle.read())
        print(f"references: {sum(len(v) for v in identity_refs.values())} image(s)")
    elif provider.capabilities.supports_identity_refs:
        print(
            "note: this model CAN use reference photos, but REFERENCE_PATHS is "
            "empty, so this run is text-to-image."
        )

    # 3. Render.
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    shots = artifact.plan.shots[:MAX_SHOTS]
    print(f"\nrendering {len(shots)} shot(s) at {WIDTH}x{HEIGHT}...\n")

    started = time.time()
    for index, shot in enumerate(shots, start=1):
        assembled = artifact.prompts[shot.node_id]
        t0 = time.time()
        print(f"[{index}/{len(shots)}] {shot.node_id} {shot.moment} ({shot.framing})", end="", flush=True)

        image = await provider.generate(
            prompt=assembled.prompt,
            negative_prompt=assembled.negative_prompt,
            identity_refs=identity_refs or None,
            width=WIDTH,
            height=HEIGHT,
            seed=1000 + index,
        )

        path = os.path.join(OUTPUT_DIR, f"{shot.node_id}_{shot.moment[:24].replace(' ', '_')}.png")
        with open(path, "wb") as handle:
            handle.write(image.content)
        print(f"  -> {path}  ({len(image.content) // 1024} KB, {time.time() - t0:.1f}s)")

    print(f"\ndone in {time.time() - started:.1f}s. Files in {os.path.abspath(OUTPUT_DIR)}/")

    # 4. Free VRAM so a re-run does not OOM.
    provider.unload()
    print("\nTo view inline:\n  from IPython.display import Image, display\n"
          f"  display(Image('{OUTPUT_DIR}/" + (shots[0].node_id if shots else "s00") + ".png'))")
    return 0


def run() -> int:
    """Run `main()` whether or not a loop is already running.

    `%run scripts/colab_smoke.py` executes this file inside the Colab kernel,
    which already has a running asyncio loop, and `asyncio.run()` refuses to
    start a second one. `!python scripts/colab_smoke.py` is a separate process,
    so it takes the plain path. Both must work: the notebook is where this
    script is actually used.
    """
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(main())  # normal script: no loop yet

    # A loop is already running (Jupyter/IPython). nest_asyncio ships with Colab
    # and makes run_until_complete reentrant; if it is missing, re-exec ourselves
    # in a subprocess rather than failing with a confusing RuntimeError.
    try:
        import nest_asyncio  # noqa: PLC0415

        nest_asyncio.apply()
        return asyncio.get_event_loop().run_until_complete(main())
    except ImportError:
        import subprocess  # noqa: PLC0415

        print("nest_asyncio is not installed; re-running in a subprocess.")
        return subprocess.call(
            [sys.executable, os.path.abspath(__file__)], env=dict(os.environ)
        )


if __name__ == "__main__":
    check_environment()
    raise SystemExit(run())
