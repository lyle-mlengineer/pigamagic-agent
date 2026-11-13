# PlusOne — Build Steps

A step-by-step build order. Each step is small enough to finish in one sitting and ends with an explicit **Done when** check, so progress is verifiable rather than assumed.

**How we work:** one step at a time. I implement, we run the check, the check passes, `progress.md` is updated, then the next step starts.

**Ground rule:** we do **not** wire a real image provider until Step 8. Steps 1–7 prove the plan compiler is good using fixtures and text output. Image generation is the most expensive and least reversible part of the build; the plan must be right before we spend money on it.

**Conventions:**
- **[you]** marks a step needing your input (accounts, keys, provider choice, brand decisions).
- Every step lists the requirement IDs it satisfies from `prd.md`.
- Every step lists the files it creates or changes.

**Reference documents:** `prd.md` (what and why), `design.md` (how, in detail), `progress.md` (where we are).

---

## Phase 0 — Foundations

### Step 0.1 — Environment and repository skeleton
Create the project layout, a virtual environment, pinned dependencies, `.gitignore`, and a runnable test harness.

```
pigamagic-agent/
  backend/app/agent/          # this build lives here
  backend/tests/agent/
  backend/requirements.txt
  .env.example
  pyproject.toml              # pytest config only
```

**Done when:** `python -m pytest` runs and exits 0 with zero tests collected, and `python -c "import app.agent"` succeeds from `backend/`.

### Step 0.2 — Configuration and adapter protocols
Define `AgentSettings` (provider names, budgets, model IDs, thresholds) loaded from env with defaults, plus `LLMProvider`, `VisionProvider`, `ImageProvider` protocols in `agent/providers/base.py`. No implementations yet.

**Done when:** settings load with defaults when no env vars are set, and constructing a provider protocol with a dummy class satisfies the type checker in a test.

### Step 0.3 — Deterministic IDs, hashing and prompt versioning
Implement content hashing for reference images (`sha256`), prompt hashing that includes `prompt_version` and all resolved parameters, and stable plan node IDs.

**Done when:** the same inputs produce identical hashes across processes (assert with a subprocess check), and changing `prompt_version` changes `prompt_hash` while everything else stays equal.

---

## Phase 1 — Plan without pixels (proves the thesis)

### Step 1 — Profile and plan schemas (FR-4.1, FR-4.4)
Implement the typed domain models from `design.md` §4.1 and §7.1: `PersonProfile`, `FaceFeatures`, `HairFeatures`, `WardrobeProfile`, `Garment`, `VenueProfile`, `SceneHints`, `ShotSpec`, `CameraSpec`, `CastEntry`, `ShootPlan`, `EditInstruction`. Strict Pydantic v2 models, `extra="forbid"`, documented field constraints.

**Files:** `agent/profiles.py`, `agent/plan.py`, `tests/agent/test_schemas.py`

**Done when:** a fully populated `ShootPlan` round-trips through JSON byte-stably, each model rejects an unknown field with a readable error, and a field-level test covers every `Literal` value.

### Step 2 — Reference analysis with deterministic fusion (FR-4.1)
Implement `analyze_references()`: hash → cache lookup → vision pass (fixture-backed) → fusion into a `ShootCast`. Fusion must prefer the **sharpest, most frontal** image for facial fields and merge non-conflicting fields across images.

**Files:** `agent/analyze.py`, `agent/providers/vision_fixture.py`, `tests/agent/test_analyze.py`

**Done when:** three reference images of the same two people fuse into 2 cast entries (not 6), the sharper image's face fields win, a blurry image's fields do not overwrite good ones, and a second run with identical bytes makes zero vision calls.

### Step 3 — Event convention packs (FR-4.3)
Ship curated JSON packs for the event types in scope, a loader, and validation. Include `canonical_moments`, `phases`, `lighting_notes`, `composition_taboos`, `wardrobe_expectations`, `sensitive_handling`.

**Files:** `agent/conventions.py`, `agent/data/conventions/*.json`, `tests/agent/test_conventions.py`

**Done when:** every pack loads and validates, an unknown event type falls back to a documented `generic` pack with a warning, and every pack's `canonical_moments` is non-empty and deduplicated.

### Step 4 — Venue research with cache and provenance (FR-4.2)
Implement the venue pipeline: normalize key → cache lookup → budgeted search/extract/fetch (fixture-backed offline) → `VenueProfile` with `confidence` and `sources`. Include the low-confidence fallback path.

**Files:** `agent/research.py`, `agent/providers/web_fixture.py`, `tests/agent/test_research.py`

**Done when:** the same venue researched twice performs one fetch round; a no-result venue yields `confidence: low` plus a warning and no fabricated `signature_details`; every profile carries non-empty `sources` or an explicit `no-sources` marker.

### Step 5 — Plan compiler and validator (FR-4.4 → FR-4.6)
Implement `compile_plan()`: prompt construction, LLM call (fixture-backed), then `PlanValidator` with the repair loop. Validator enforces cast coverage ≥2, activity coverage ≥1, subject existence, shot count, duplicate-framing rejection, and policy blocks.

**Files:** `agent/compile.py`, `agent/prompts.py`, `agent/providers/llm_fixture.py`, `tests/agent/test_compile.py`

**Done when:** every golden brief in `tests/agent/golden/` compiles with `validation.repaired == False`; a deliberately cast-incomplete plan is repaired by injection; a policy-violating node is dropped with a warning; two identical compiles produce byte-identical prompts.

### Step 6 — Prompt assembly determinism (FR-4.7)
Harden `assemble_prompt()` so output is a pure function of the plan node plus cast plus reference notes, with the reference-notes block token-capped and delimited as data (SR-6).

**Files:** `agent/prompts.py`, `tests/agent/test_prompts.py`

**Done when:** prompt length is capped under the configured limit for a worst-case node, a prompt-injection string placed in `story` appears only inside the delimited reference block, and a golden-file test locks the exact output for three nodes.

### Step 7 — Golden briefs and the offline compile CLI
Build a CLI that compiles a brief to a `ShootPlan` and prints it, plus a golden-brief suite covering 5 event types × with/without references.

**Files:** `agent/cli.py`, `tests/agent/golden/*.json`, `tests/agent/test_golden.py`

**Done when:** `python -m app.agent.cli plan tests/agent/golden/wedding_with_refs.json` prints a complete valid plan with no provider credentials, and all golden briefs pass in CI.

> **Milestone 1 gate — the product thesis.** At this point the plan is proven without spending on image generation. Do not proceed until every golden brief produces a plan you would be happy to photograph.

---

## Phase 2 — One real photo

### Step 8 — Real LLM and vision adapters **[you: provider + key]**
Replace the fixture adapters with real implementations behind the same protocols. Structured-output enforcement, retries with backoff, timeouts, and cost accounting per call.

**Done when:** a real brief compiles using the real provider, the recorded cost is logged, and disabling the network produces a specific, readable error rather than a hang.

### Step 9 — Image provider adapter and single-node render (FR-4.7)
Implement the `ImageProvider` adapter, `PhotoJobSpec`, `prompt_hash` cache, and a single-node render path that persists a `Media` + `GeneratedPhoto` row.

**Files:** `agent/render.py`, `agent/providers/images.py`, `tests/agent/test_render.py`

**Done when:** one plan node renders to a stored photo, re-running the identical node is a cache hit with zero provider calls and zero cost, and a forced provider error returns a typed failure after exactly 2 retries.

### Step 10 — QA gate (FR-4.8, SR-4)
Implement integrity, identity, plausibility and safety checks. Identity embeddings are computed, scored and discarded (NFR-9).

**Files:** `agent/qa.py`, `tests/agent/test_qa.py`

**Done when:** a corrupt buffer fails integrity, a deliberately mismatched face is marked `identity_weak` and retried once, an unsafe fixture is flagged and never published, and no embedding is written to any store (assert by inspecting all DB models).

---

## Phase 3 — Runs, streaming, credits

### Step 11 — Run orchestration and the credit ledger (FR-4.10, FR-8.2, FR-8.3)
Implement `generation_runs`, reserve-then-settle, per-node refunds, and the run state machine (`queued → researching → compiling → rendering → done | partial | failed`).

**Files:** `agent/run.py`, `agent/ledger.py`, `tests/agent/test_run.py`

**Done when:** a 12-node run with 1 forced failure ends `partial`, charges 11, and shows one refund line; an insufficient-credit request is rejected before any provider call; re-posting the same idempotency key returns the same run.

### Step 12 — Worker process and queue
A standalone worker consuming the queue with visibility timeout, retry, and crash recovery.

**Files:** `agent/worker.py`, `tests/agent/test_worker.py`

**Done when:** killing the worker mid-job returns the job to `queued` with `attempt += 1`, and no job is processed twice concurrently.

### Step 13 — Runs API and SSE (FR-4.9, FR-5.7)
`POST /shoots/{id}/generate`, `GET /runs/{id}`, `GET /runs/{id}/events` (SSE with `Last-Event-ID` replay).

**Files:** `app/api/runs.py`, `agent/events.py`, `tests/agent/test_runs_api.py`

**Done when:** an SSE client receives `run`, `photo`, and `run_done` events in order; reconnecting with `Last-Event-ID` replays only missed events; no photo is delivered twice.

### Step 14 — Feed integration
Wire generated photos into the existing feed serialization so streamed and settled photos use one card contract.

**Done when:** a settled photo and a streamed photo produce identical card payload shapes, and cursor pagination stays stable while photos are inserted.

---

## Phase 4 — Research quality

### Step 15 — Real web research **[you: search API key]**
Swap the web fixture for a real search provider with the budget in `design.md` §6.1.

**Done when:** a known real venue produces a profile with ≥2 correct signature details verified by eye, and the budget is never exceeded on a pathological venue name.

### Step 16 — RAG corpus (FR-4.4)
pgvector ingest of the four corpora, retrieval with a similarity floor, token-capped context block.

**Done when:** retrieval returns ≤3 chunks under the token cap, a nonsense query returns nothing rather than noise, and retrieved text appears only in the data-delimited block.

---

## Phase 5 — Feedback loop

### Step 17 — Comment modes and edit compilation (FR-7.1 → FR-7.4)
Add `mode` to comments, compile `EditInstruction` deltas, rebuild from the parent spec while preserving identity and venue.

**Done when:** "make it brighter" changes only lighting and leaves `preserves` intact; an edit of an edit chains correctly; the parent photo is byte-identical before and after.

### Step 18 — Lineage, compare and per-edit credits (FR-7.5, FR-7.6)
`GET /photos/{id}/lineage`, compare payloads, independent credit reservation per edit.

**Done when:** a photo has ≥2 siblings retrievable, and each edit appears as its own ledger line.

---

## Phase 6 — Hardening

### Step 19 — Eval harness in CI
Automated identity, coverage, prompt-stability and cost assertions on golden briefs.

**Done when:** the harness fails loudly when a prompt change degrades identity score below threshold.

### Step 20 — Safety review console and purge paths (SR-1 → SR-7)
Admin flag review with plan and prompt visible; account deletion purge verified end to end.

**Done when:** deleting an account removes media, analyses, photos, runs, and ledger rows, and a flagged photo is unreachable via any public route.

---

## Step index

| Phase | Steps | Milestone |
| --- | --- | --- |
| 0 | 0.1–0.3 | Scaffold runs |
| 1 | 1–7 | **Plan proven without pixels** (gate) |
| 2 | 8–10 | One real, QA-passed photo |
| 3 | 11–14 | Streaming feed with credits |
| 4 | 15–16 | Real research quality |
| 5 | 17–18 | Comment-to-edit loop |
| 6 | 19–20 | Hardened and safe |
