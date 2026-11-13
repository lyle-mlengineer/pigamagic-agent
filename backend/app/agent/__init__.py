"""PlusOne agent — the AI photographer pipeline.

This package turns an event brief plus reference photos into a validated
photoshoot plan and, eventually, a stream of generated photographs.

Module map (see design.md §15):

    profiles.py     structured Person/Wardrobe/Venue profiles
    plan.py         ShootPlan, ShotSpec, EditInstruction
    analyze.py      reference hashing + vision pass + deterministic fusion
    research.py     venue research with cache and provenance
    conventions.py  curated event-type packs
    rag.py          corpus ingest and retrieval
    compile.py      plan compiler + PlanValidator
    prompts.py      deterministic prompt assembly
    render.py       fan-out, prompt-hash cache, retries
    qa.py           integrity, identity, plausibility, safety gates
    run.py          run orchestration + credit settlement
    cli.py          offline compile CLI

Import order is deliberate: nothing here imports a provider at module load,
so the package stays importable with no credentials configured.
"""

__version__ = "0.1.0"

# The prompt version participates in every prompt hash (see agent/hashing.py).
# Bumping it deliberately invalidates cached generations.
PROMPT_VERSION = "v1"

__all__ = ["__version__", "PROMPT_VERSION"]
