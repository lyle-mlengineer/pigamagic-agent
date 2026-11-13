# PlusOne — Colab smoke test for local Hugging Face image models
#
# Paste this into a Colab notebook (GPU runtime) to generate REAL photos from a
# golden brief using open-weight models. No API key needed for the default model.
#
# Runtime > Change runtime type > T4 GPU
#
# Cell 1: install
#   !pip install -q diffusers transformers accelerate safetensors
#
# Cell 2: clone/upload this repo, then run this file.
#
# ── Which model to use ──────────────────────────────────────────────────────
#   FLUX.1-schnell        ungated, 4 steps, fastest. No licence click. Default.
#   Qwen-Image-Edit-2509  PRESERVES FACIAL IDENTITY from reference photos.
#                         This is the one that answers "is it really me?".
#   FLUX.1-dev / Kontext  gated: accept the licence on the HF model page and
#                         set HF_TOKEN.
# ────────────────────────────────────────────────────────────────────────────

from __future__ import annotations

import asyncio
import os
import sys
import time

# --- configuration ---------------------------------------------------------

# Start here: no licence, no token, ~1 minute on a free T4.
MODEL = os.environ.get("PLUSONE_MODEL", "black-forest-labs/FLUX.1-schnell")

# For identity-preserving runs, use:
#   MODEL = "Qwen/Qwen-Image-Edit-2509"
# and provide reference photos (see REFERENCE_PATHS below).

# Gated models need a Hugging Face token: https://huggingface.co/settings/tokens
HF_TOKEN = os.environ.get("HF_TOKEN", "")

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


async def main() -> int:
    from app.agent.cli import _demo_venue, demo_cast
    from app.agent.config import AgentSettings
    from app.agent.pipeline import load_brief, run_pipeline
    from app.agent.providers.image_huggingface import HuggingFaceImageProvider
    from app.agent.providers.llm_fixture import FixtureLLMProvider
    from app.agent.providers.vision_fixture import FixtureVisionProvider
    from app.agent.providers.web_fixture import FixtureWebProvider
    from app.agent.providers.registry import describe_image_providers

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
