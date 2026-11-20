# PlusOne — Colab smoke test for local Hugging Face image models
#
# Paste this into a Colab notebook (GPU runtime) to generate REAL photos from a
# golden brief using open-weight models. No API key needed: with no HF_TOKEN the
# script runs the ungated sdxl-turbo; set HF_TOKEN to get FLUX.1-schnell.
#
# Runtime > Change runtime type > T4 GPU
#
# Cell 1: install
#   !pip install -q diffusers transformers accelerate safetensors
#
# Cell 2: clone/upload this repo, then run this file.
#
# ── Which model to use ──────────────────────────────────────────────────────
#   sdxl-turbo            ungated, ~2.6 GB, 2 steps. Zero-setup default.
#   Qwen-Image-Edit-2509  ungated, PRESERVES FACIAL IDENTITY from references.
#                         This is the one that answers "is it really me?".
#   FLUX.1-schnell        4 steps, fastest — but gated since 2026: accept the
#                         licence on its HF page and set HF_TOKEN, then pass
#                         PLUSONE_MODEL=black-forest-labs/FLUX.1-schnell.
#   FLUX.1-dev / Kontext  gated too: licence click plus HF_TOKEN.
# ────────────────────────────────────────────────────────────────────────────

from __future__ import annotations

import asyncio
import os
import sys
import time

# --- configuration ---------------------------------------------------------

# Gated models need a Hugging Face token: https://huggingface.co/settings/tokens
HF_TOKEN = os.environ.get("HF_TOKEN", "")


def pick_model() -> str:
    """Choose a checkpoint that can actually download in this runtime.

    `FLUX.1-schnell` is the fastest thing here but Hugging Face gated it, so it
    needs both a licence click and `HF_TOKEN`. With no token the script starts
    on an ungated checkpoint instead of dying on a 403 after a long download.
    """
    chosen = os.environ.get("PLUSONE_MODEL", "").strip()
    if chosen:
        return chosen
    if HF_TOKEN:
        return "black-forest-labs/FLUX.1-schnell"

    from pigamagic_agent.agent.providers.image_huggingface import QUICK_TEST_MODEL

    return QUICK_TEST_MODEL


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
        sys.exit("torch is missing. Run: !pip install -q diffusers transformers accelerate safetensors")

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
            "  !pip install -q diffusers transformers accelerate safetensors"
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
    for row in describe_image_providers():
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


if __name__ == "__main__":
    check_environment()
    raise SystemExit(asyncio.run(main()))
