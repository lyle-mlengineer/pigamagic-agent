# PlusOne — the photoshoot agent

**A personal AI photographer.** Give it a short event brief and a handful of reference photos, and it plans, writes and renders a full photo story of a day that was badly photographed — with the same people, the same venue and the same clothes in every frame.

This repository is the **agent layer**: the whole reasoning and rendering backend (plan → prompts → photos → feed → credits). It is designed to run offline against fixtures with zero credentials, to switch to a real image vendor by configuration alone, and to be vendored into the PlusOne product shell.

- Product requirements: [`prd.md`](prd.md)
- Technical design: [`design.md`](design.md)
- Build order: [`steps.md`](steps.md)
- Build state / resume file: [`progress.md`](progress.md)

---

## Table of contents

1. [What this is](#what-this-is)
2. [Status at a glance](#status-at-a-glance)
3. [Repository layout](#repository-layout)
4. [How it works](#how-it-works)
5. [Requirements](#requirements)
6. [Quick start (offline, no keys)](#quick-start-offline-no-keys)
7. [Configuration: the `.env` file](#configuration-the-env-file)
8. [Running it locally](#running-it-locally)
9. [Generating real images](#generating-real-images)
10. [Running it on Google Colab](#running-it-on-google-colab)
11. [Tests and the eval gate](#tests-and-the-eval-gate)
12. [Troubleshooting](#troubleshooting)
13. [Safety, privacy and consent](#safety-privacy-and-consent)
14. [License](#license)

---

## What this is

Most people cannot get good photographs of themselves at the moments that matter: phone cameras fail in low light, camera-shy people avoid being photographed, and amateur framing produces 200 unusable shots. Professional coverage is expensive and cannot be arranged retroactively.

PlusOne takes the facts the user already knows — who was there, what the place looked like, what happened — plus reference photos, and produces a convincing, beautiful, *consistent* set of photographs.

It is explicitly **not** a face-swap or deepfake tool, not an image editor, and not a social network. It generates new photographs from an attested cast, keeps identity consistent across a shoot by construction, and never depicts anyone the user has not vouched for.

### What "generating a photoshoot" means here

One call to the pipeline turns:

```
brief (title, event type, venue, activities, story, style)
+ reference photos with captions
        │
        ▼
ShootPlan   — cast, wardrobe, venue profile, shot list, prompts, provenance
        │
        ▼
photographs — one independent, retryable, cacheable job per shot
```

The plan is the single artifact that makes the whole system reproducible: same plan + same seed = same photo.

---

## Status at a glance

| Area | State |
| --- | --- |
| Agent pipeline (analyse → research → compile → prompts) | **Done**, offline on fixtures |
| 13 golden briefs (wedding, memorial, graduation, …) | **Done** |
| Plan compiler + deterministic validator | **Done** |
| Prompt assembly + prompt cache keyed by content hash | **Done** |
| Image providers: fixture, NVIDIA NIM, Hugging Face/Qwen, Gemini, OpenAI, fal | **Done** |
| Blob storage: local, S3-compatible (Backblaze B2 / R2 / MinIO / Wasabi) | **Done** |
| Safety gate + admin review console + account purge | **Done** |
| Run orchestration, credit ledger, worker queue with leases | **Done** |
| HTTP API: generate, poll, SSE stream, feed, lineage | **Done** |
| Comment → edit (new take, preserved identity, lineage) | **Done** |
| Test suite | **829 tests, green** (`backend/`) |
| LLM / vision / web adapters | **Fixtures only** — real vendors are the next step (`design.md` Step 15) |

The evaluation gate (`python -m pigamagic_agent.agent.cli eval`) runs every golden brief and exits non-zero if any of them stops compiling valid plans. That is the CI entry point.

---

## Repository layout

```
pigamagic-agent/
├── .env.example                 # every setting, no secrets
├── prd.md                       # product requirements
├── design.md                    # technical design (the long one)
├── steps.md                     # build order
├── progress.md                  # build state and resume instructions
└── backend/
    ├── requirements.txt         # runtime + test dependencies
    ├── pytest.ini               # testpaths=tests, asyncio_mode=auto
    ├── pigamagic_agent/
    │   ├── agent/
    │   │   ├── cli.py            # offline CLI: briefs | plan | render | eval
    │   │   ├── pipeline.py       # brief + refs → ShootArtifact (plan + prompts)
    │   │   ├── analyze.py        # reference photos → Person/Wardrobe profiles
    │   │   ├── research.py       # venue research with provenance + confidence
    │   │   ├── conventions.py    # curated event-convention packs
    │   │   ├── compile.py        # LLM proposes a plan, Python validates it
    │   │   ├── plan.py           # ShootPlan schema
    │   │   ├── prompts.py        # deterministic prompt assembly
    │   │   ├── hashing.py        # canonical JSON + content hashes (cache keys)
    │   │   ├── safety.py         # pre-flight refusal gate
    │   │   ├── render.py         # fan-out, retry, per-shot QA
    │   │   ├── run.py            # run state machine + credit reservation
    │   │   ├── worker.py         # job queue with leases and fencing
    │   │   ├── ledger.py         # credit ledger (reserve → settle → refund)
    │   │   ├── feed.py           # cursor-paginated feed cards
    │   │   ├── comments.py       # comment → edit instruction
    │   │   ├── lineage.py        # takes, parent/child compare
    │   │   ├── storage.py        # blob store: local / S3 / B2
    │   │   ├── eval.py           # offline eval harness
    │   │   ├── data/briefs/      # 13 golden briefs
    │   │   ├── data/conventions/ # 14 convention packs
    │   │   └── providers/        # one module per vendor + fixtures
    │   └── api/
    │       ├── runs.py           # generate / poll / SSE / feed / lineage
    │       └── admin.py          # safety review console + purge
    ├── scripts/
    │   ├── serve.py              # local dev server (fixtures + seeded ledger)
    │   └── colab_smoke.py        # real open-weight models on a Colab GPU
    └── tests/agent/              # 829 tests, no network required
```

---

## How it works

### The pipeline

```
1. Ingest & analyse      reference photos → hash → vision pass
                         → PersonProfile / WardrobeProfile → ShootCast

2. Research              venue web search + images → VenueProfile (provenance, confidence)
                         event type → ConventionPack
                         (budgeted: AGENT_MAX_SEARCH_CALLS)

3. Compile plan          LLM proposes shots → Python validates the plan
                         → PromptSpec per shot (activity present, cast covered)

4. Render                each shot is an independent job:
                         prompt → image provider → integrity/QA → blob → feed event

5. Stream                photos appear one at a time as they pass QA (SSE)
```

Hard invariants enforced by the validator, not by hope:

- every activity the user listed appears in at least one shot;
- every person in the cast appears in at least two shots;
- sensitive events (memorial, religious/cultural) get a conservative treatment;
- a prompt is never emitted for a plan that failed validation.

### The plan is the artifact

The plan is stored, versioned and diffable. That is what makes regeneration, single-photo edits, golden-file tests, cost debugging and admin review possible. An edit of one photo creates a **sibling take** with its own seed and lineage — the original is never mutated.

### Providers are swappable

Every external capability sits behind a narrow typed protocol:

| Seam | Adapters today |
| --- | --- |
| `LLMProvider` | fixture |
| `VisionProvider` | fixture |
| `WebProvider` | fixture |
| `ImageProvider` | fixture, NVIDIA NIM, Hugging Face (FLUX/Qwen/SD), Gemini, OpenAI, fal |
| `BlobStore` | local filesystem, S3-compatible (B2, R2, MinIO, Wasabi) |

Fixtures are deterministic and offline, so tests and the CLI cost nothing and never flake. Real vendors produce images; the planning half is still fixture-based until Step 15 lands a real LLM/vision adapter.

### Credits and run states

```
queued → researching → compiling → rendering → done | partial | failed
```

Ordering is deliberate:

1. **safety gate** — a refusal costs zero credits and zero provider calls;
2. **reserve** — an unaffordable run is rejected before any work starts;
3. research + compile;
4. render;
5. **settle** — charge what was delivered, refund what failed, refund cache hits.

States are validated transitions; a run cannot jump `queued → done`. Every credit movement is a ledger line with a reason, so support can explain any balance.

### Safety

- consent attestation is required before a run;
- minors are never placed in unsafe, romantic or physically risky contexts;
- web text and user text are treated as data, never as instructions;
- a flagged photo is private everywhere — feed, media URL, and live stream;
- reference deletion removes derived analysis; account deletion purges media, analyses, photos and runs.

---

## Requirements

| Item | Version |
| --- | --- |
| Python | **3.11+** (verified on 3.15.2) |
| pip packages | `backend/requirements.txt` |
| Optional (GPU image generation) | `torch`, `diffusers`, `transformers`, `accelerate`, `safetensors` |
| Disk | a media folder you can grow — the repo's own `.env.example` points at `D:/pigamagic-media/agent` |
| OS | Windows, macOS or Linux (the CLI, tests and API are pure Python) |

Nothing else is required: no database, no Redis, no cloud account. The persistence layers used by the API are in-memory by design at this stage; Postgres/Redis are the planned swap, not a prerequisite to run.

---

## Quick start (offline, no keys)

```bash
# 1. get the code
git clone https://github.com/lyle-mlengineer/pigamagic-agent.git
cd pigamagic-agent

# 2. create and activate a virtual environment
python -m venv .venv
# Windows (Git Bash):
source .venv/Scripts/activate
# macOS / Linux:
# source .venv/bin/activate

# 3. install dependencies
pip install -r backend/requirements.txt

# 4. create your environment file
cp .env.example .env      # then edit it — see the next section

# 5. prove the install
cd backend
python -m pytest -q                     # expect: 829 passed
python -m pigamagic_agent.agent.cli briefs
```

Expected output of the last command:

```
Available briefs:
  anniversary  anniversary 5 activities
  baby-shower  baby-shower 5 activities
  ...
  wedding      wedding     4 activities
```

Compile one photoshoot plan end to end, with no network and no cost:

```bash
python -m pigamagic_agent.agent.cli plan --brief wedding --show-prompt
python -m pigamagic_agent.agent.cli plan --brief wedding --out artifact.json
```

Render a couple of photos with the fixture provider (real PNGs, synthetic content):

```bash
python -m pigamagic_agent.agent.cli render --brief wedding --store local --provider fixture --max-shots 2
```

The files land under `AGENT_MEDIA_DIR` (default `D:/pigamagic-media/agent` in `.env.example`, `./var/media` in code).

> **Note on the CLI.** `render` uses the built-in demonstration cast and venue (`cli.demo_cast`, `cli._demo_venue`), because the CLI has no upload path. Real casts come from the vision pass over the user's reference photos via the API or the product shell. The CLI's job is to prove the agent layer end to end, cheaply.

---

## Configuration: the `.env` file

Copy `.env.example` to `.env` in the **repository root** and edit it. `.env` is gitignored; `.env.example` must never contain a real secret.

Settings are loaded by `pydantic-settings` with the `AGENT_` prefix. The loader reads `.env` from the current directory **and** `../.env`, so running commands from `backend/` picks up the root file — no duplication needed.

### Provider selection

| Variable | Values | Default |
| --- | --- | --- |
| `AGENT_LLM_PROVIDER` | `fixture` | `fixture` |
| `AGENT_VISION_PROVIDER` | `fixture` | `fixture` |
| `AGENT_WEB_PROVIDER` | `fixture` | `fixture` |
| `AGENT_IMAGE_PROVIDER` | `fixture`, `nvidia`, `huggingface`, `qwen`, `gemini`, `openai`, `fal` | `fixture` |

Aliases accepted by the image provider switch: `google`/`nano-banana` → Gemini, `flux` → fal, `hf`/`local`/`colab`/`transformers`/`diffusers`/`flux-local`/`sd` → Hugging Face, `qwen` → Hugging Face with the identity-preserving Qwen default, `nim`/`bfl`/`black-forest-labs` → NVIDIA.

`fixture` needs **no** credentials and never touches the network. Everything below it can be left blank while you are on fixtures.

### Credentials

| Variable | Used by | Notes |
| --- | --- | --- |
| `AGENT_LLM_API_KEY` | LLM provider | blank while on fixtures |
| `AGENT_VISION_API_KEY` | vision provider | blank while on fixtures |
| `AGENT_IMAGE_API_KEY` | Gemini / OpenAI / fal / Hugging Face | for Hugging Face this is an **HF token**, only needed for gated models |
| `AGENT_WEB_API_KEY` | web research | blank while on fixtures |
| `AGENT_NVIDIA_API_KEY` | NVIDIA NIM | get one at <https://build.nvidia.com> → profile → API Keys; used only when `AGENT_IMAGE_PROVIDER=nvidia` |
| `AGENT_NVIDIA_IMAGE_BASE_URL` | NVIDIA NIM | default `https://ai.api.nvidia.com/v1/genai/black-forest-labs` |
| `AGENT_NVIDIA_NIM_SELF_HOSTED` | NVIDIA NIM | `false` on the hosted preview; set `true` only for your own NIM container (it is the only mode that can use real reference faces) |

### Models

Leave blank to use the provider's pinned, tested default.

| Variable | Blank default |
| --- | --- |
| `AGENT_LLM_MODEL` | provider default |
| `AGENT_VISION_MODEL` | provider default |
| `AGENT_IMAGE_MODEL` | `gemini` → `gemini-3.1-flash-image`; `openai` → `gpt-image-2.5-flare`; `fal` → `blackforestlabs/flux-3/text-to-image`; `nvidia` → `flux.3-image-klein`; `huggingface` → `black-forest-labs/FLUX.1-schnell`; `qwen` → `Qwen/Qwen-Image-Edit-2509` |

**Identity-preserving local models:** `Qwen/Qwen-Image-Edit-2509` (up to 3 reference faces, the right choice for this app) and `black-forest-labs/FLUX.1-Kontext-dev` (gated, 1 reference). Text-only local models: `FLUX.1-schnell` (ungated, 4 steps, fastest), `FLUX.1-dev` (gated), `Qwen/Qwen-Image`, `stabilityai/stable-diffusion-3.5-large`.

### Budgets, limits and determinism

| Variable | Default | Meaning |
| --- | --- | --- |
| `AGENT_PROMPT_VERSION` | `v1` | **hash-relevant.** Bumping it deliberately invalidates every cached prompt/generation |
| `AGENT_MAX_SHOTS` | `16` | upper bound on shots per shoot and on `shot_count` in the API |
| `AGENT_MAX_SEARCH_CALLS` | `6` | venue-research budget |
| `AGENT_MAX_REFERENCE_NOTES_TOKENS` | `300` | reference-note budget inside a prompt |
| `AGENT_IDENTITY_SIMILARITY_FLOOR` | `0.45` | below this, a shot is retried |
| `AGENT_IDENTITY_SIMILARITY_TARGET` | `0.62` | the quality target; must be ≥ the floor |
| `AGENT_IMAGE_BUDGET_USD` | `0.0` | **`0` means unlimited, not blocked.** Set a real ceiling before spending money |
| `AGENT_PROVIDER_MAX_ATTEMPTS` | `3` | retries for outbound provider calls |
| `AGENT_PROVIDER_TIMEOUT_S` | `120` | per-call timeout |
| `AGENT_OFFLINE` | `false` | `true` forbids **all** outbound HTTP; fixtures keep working. Tests and CI stay offline |

### Storage

| Variable | Default | Meaning |
| --- | --- | --- |
| `AGENT_STORAGE_PROVIDER` | `local` | `local`, `s3`, `backblaze` (`b2`), `r2`, `minio`, `wasabi` |
| `AGENT_MEDIA_DIR` | `./var/media` | local store root. Use an absolute path on a roomy drive |
| `AGENT_MEDIA_BASE_URL` | *(blank)* | prefix for local media URLs, e.g. `http://localhost:8000/media` |
| `AGENT_STORAGE_BUCKET` | *(blank)* | bucket name for the S3-compatible stores |
| `AGENT_STORAGE_ENDPOINT` | *(blank)* | leave blank for Backblaze (derived as `https://s3.<region>.backblazeb2.com`); required for R2/MinIO/Wasabi |
| `AGENT_STORAGE_REGION` | *(blank)* | e.g. `us-west-004` |
| `AGENT_STORAGE_ACCESS_KEY_ID` | *(blank)* | B2: the **application key id** (the master key does not work with the S3 API) |
| `AGENT_STORAGE_SECRET_ACCESS_KEY` | *(blank)* | B2: the **application key** |
| `AGENT_STORAGE_PREFIX` | `plusone` | key prefix, so several apps can share a bucket |
| `AGENT_STORAGE_PUBLIC_BASE_URL` | *(blank)* | CDN/public read base; blank serves through the app |

### Admin

| Variable | Default | Meaning |
| --- | --- | --- |
| `AGENT_ADMIN_TOKEN` | *(blank)* | shared secret for `/api/v1/admin/*`, sent as the `X-Admin-Token` header. **Blank disables the admin routes** (503) — an unset secret must never mean "open to everyone". Generate one with `python -c "import secrets;print(secrets.token_urlsafe(32))"` |

### Minimal files to copy

**Offline / tests / CLI — everything is a fixture:**

```dotenv
AGENT_LLM_PROVIDER=fixture
AGENT_VISION_PROVIDER=fixture
AGENT_IMAGE_PROVIDER=fixture
AGENT_WEB_PROVIDER=fixture
AGENT_STORAGE_PROVIDER=local
AGENT_MEDIA_DIR=D:/pigamagic-media/agent
AGENT_OFFLINE=true
```

**Real images on a hosted vendor (example: Gemini):**

```dotenv
AGENT_LLM_PROVIDER=fixture
AGENT_VISION_PROVIDER=fixture
AGENT_WEB_PROVIDER=fixture
AGENT_IMAGE_PROVIDER=gemini
AGENT_IMAGE_API_KEY=your-key-here
AGENT_IMAGE_BUDGET_USD=5
AGENT_STORAGE_PROVIDER=local
AGENT_MEDIA_DIR=D:/pigamagic-media/agent
```

**Free local weights on a GPU (Colab or your own machine):**

```dotenv
AGENT_IMAGE_PROVIDER=qwen
AGENT_IMAGE_MODEL=Qwen/Qwen-Image-Edit-2509
AGENT_IMAGE_API_KEY=
AGENT_STORAGE_PROVIDER=local
AGENT_MEDIA_DIR=D:/pigamagic-media/agent
```

---

## Running it locally

Two ways to drive the agent: the offline CLI (fastest, no server) and the HTTP API (what the product shell talks to).

### 1. The CLI

All commands are run from `backend/`.

```bash
cd backend

# list the golden briefs
python -m pigamagic_agent.agent.cli briefs

# compile a plan and print the summary
python -m pigamagic_agent.agent.cli plan --brief wedding

# print the assembled prompt for the first shot
python -m pigamagic_agent.agent.cli plan --brief wedding --show-prompt

# write the full artifact (plan + prompts + provenance) for diffing/storage
python -m pigamagic_agent.agent.cli plan --brief wedding --out artifact.json

# exercise the degraded path (simulated planner outage → deterministic fallback)
python -m pigamagic_agent.agent.cli plan --brief wedding --degraded

# enforce a per-prompt token budget (reference notes shrink first)
python -m pigamagic_agent.agent.cli plan --brief memorial --max-prompt-tokens 120

# render photos: provider and store can be overridden per call
python -m pigamagic_agent.agent.cli render --brief wedding --store local --provider fixture --max-shots 3 --out manifest.json

# run the CI gate over every golden brief (exit 0 = healthy)
python -m pigamagic_agent.agent.cli eval
```

`plan` and `render` accept a brief **name** (`wedding`) or a **path** to your own brief JSON (`./my-brief.json`). The brief schema is the one used by `data/briefs/*.json`: `title`, `event`, `venue_name`, `activities[]`, `story`, `style_preset`, `shot_count`.

Exit codes: `0` success, `1` eval/render failure, `2` unknown brief, `3` provider/storage setup failure, `4` safety refusal.

### 2. The local API server

`create_app()` is a factory: the API needs providers and a ledger injected, so `uvicorn pigamagic_agent.api:app` alone will not work. `backend/scripts/serve.py` is the ready-made local host — it wires fixture providers, the configured blob store, and a seeded credit balance:

```bash
cd backend
python scripts/serve.py                     # http://127.0.0.1:8000, account "demo", 50 credits
python scripts/serve.py --port 9000         # different port
python scripts/serve.py --credits 200       # bigger seed balance
python scripts/serve.py --reload            # auto-restart on file changes
```

Open <http://127.0.0.1:8000/docs> for the interactive OpenAPI docs.

State is in-memory: restarting the server clears runs, feed cards and the ledger. Photo bytes persist, because they live in the blob store (`AGENT_MEDIA_DIR`).

#### Endpoints

| Method | Path | Purpose |
| --- | --- | --- |
| `POST` | `/api/v1/shoots/{shoot_id}/generate` | reserve credits, run the pipeline, return `202` with the run |
| `GET` | `/api/v1/runs/{run_id}` | persisted run state (survives a page reload) |
| `GET` | `/api/v1/runs/{run_id}/events` | `text/event-stream` of run events, with `Last-Event-ID` replay |
| `GET` | `/api/v1/feed?limit=24&cursor=…` | settled feed, newest first; `X-Next-Cursor` header |
| `GET` | `/api/v1/photos/{photo_id}/lineage` | every take in a photo's family, as a tree + siblings |
| `GET` | `/api/v1/photos/{photo_id}/compare` | parent vs child: what an edit changed and kept |
| `GET` | `/api/v1/health` | liveness + stream version |
| `GET` | `/api/v1/admin/flags` | safety review queue (**needs `X-Admin-Token`**) |
| `GET` | `/api/v1/admin/flags/{photo_id}` | one flagged photo with its plan and prompt |
| `POST` | `/api/v1/admin/flags/{photo_id}` | clear or remove a flagged photo |
| `POST` | `/api/v1/admin/purge/{account_id}` | purge an account (media, analyses, photos, runs) |

#### Walkthrough with curl

```bash
# generate two photos from the wedding brief for the seeded "demo" account
curl -s -X POST http://127.0.0.1:8000/api/v1/shoots/demo/generate \
  -H "Content-Type: application/json" \
  -H "Idempotency-Key: demo-wedding-1" \
  -d '{"account_id":"demo","brief":"wedding","shot_count":2}'

# poll the run
curl -s http://127.0.0.1:8000/api/v1/runs/<run_id>

# watch it live (SSE; -N disables buffering)
curl -N http://127.0.0.1:8000/api/v1/runs/<run_id>/events

# read the feed
curl -s "http://127.0.0.1:8000/api/v1/feed?limit=2"
```

Request body fields for `generate`:

| Field | Required | Notes |
| --- | --- | --- |
| `account_id` | yes | who pays; must have credits |
| `brief` | yes | golden brief name or a path to a brief JSON |
| `shot_count` | no | must be within `1 … AGENT_MAX_SHOTS` (`422` otherwise); default is the brief's own count |
| `idempotency_key` | no | same key → the *same* run, not a second charge (also accepted as the `Idempotency-Key` header) |

Error contract: `402` insufficient credits (with `required`/`available`), `404` unknown brief or run, `409` a live run already exists for that shoot, `422` invalid brief or shot count, `503` admin console disabled, `500` run failure. A safety refusal is not an HTTP error here: `POST …/generate` answers `202` with a run whose status is `failed`, and `error` holds the reason — cost is zero, because the gate runs before any provider call.

The API runs the pipeline inside the request (there is no separate worker process in this repo yet); the SSE endpoint exists so a client never has to poll, and the stream is a notification channel over persisted state — never the transport for pixels.

---

## Generating real images

Pick one provider in `.env` and put its key in `AGENT_IMAGE_API_KEY` (NVIDIA also accepts `AGENT_NVIDIA_API_KEY`). Nothing else changes: `render_plan()` treats every adapter identically.

| Provider | Needs | Identity refs | Real cost |
| --- | --- | --- | --- |
| `fixture` | nothing | yes (simulated) | $0 |
| `nvidia` | NVIDIA API key | **no** on the hosted preview | trial credits |
| `huggingface` | a GPU | model-dependent | $0 |
| `qwen` | a GPU | **yes, up to 3** | $0 |
| `gemini` | Google key | vendor-dependent | per image |
| `openai` | OpenAI key | vendor-dependent | per image |
| `fal` | fal key | vendor-dependent | per image |

Two things worth knowing before you spend anything:

1. **The hosted NVIDIA preview cannot preserve a real face.** It accepts only predefined example images, so the adapter declares `supports_identity_refs=False` and refuses instead of quietly producing a stranger. Self-hosted NIM (`AGENT_NVIDIA_NIM_SELF_HOSTED=true`) declares 8 refs. It also caps prompts at 2000 characters, which the adapter enforces by trimming reference notes first — subject and action always survive.
2. **Set `AGENT_IMAGE_BUDGET_USD` before the first paid run.** `0.0` means *unlimited*, and the budget ledger only refuses calls once a positive ceiling is set. A single runaway loop can otherwise drain an account.

Render through a real provider from the CLI:

```bash
cd backend
AGENT_IMAGE_PROVIDER=qwen AGENT_IMAGE_MODEL=Qwen/Qwen-Image-Edit-2509 \
  python -m pigamagic_agent.agent.cli render --brief wedding --max-shots 1
```

Generated images are content-addressed: identical plan + prompt version + settings + seed hits the cache and is not re-billed. Changing `AGENT_PROMPT_VERSION` is the deliberate way to invalidate that cache.

---

## Running it on Google Colab

Colab is the free way to run **real open-weight models** with no API key: `backend/scripts/colab_smoke.py` plans a golden brief with the offline fixtures and then renders the shots on the GPU with `diffusers`. Nothing in this path touches a paid vendor.

### 0. What you need

- a Google account and a browser;
- a Colab runtime with a GPU (free T4 is enough for FLUX.1-schnell);
- optionally 1–3 reference photos per person, if you want an identity-preserving run.

### 1. Start a GPU runtime

In Colab: **Runtime → Change runtime type → Hardware accelerator: T4 GPU → Save**.

Confirm it:

```python
!nvidia-smi
```

You should see a Tesla T4 and ~15 GB of memory.

### 2. Get the code

```python
!git clone https://github.com/lyle-mlengineer/pigamagic-agent.git
%cd pigamagic-agent/backend
```

(Alternatively upload the folder as a zip and `%cd` into its `backend/`.)

### 3. Install the dependencies

```python
!pip install -q -r requirements.txt
!pip install -q diffusers transformers accelerate safetensors
```

Colab already ships a CUDA build of `torch`; these commands only add the pipeline libraries. Expect 1–2 minutes.

> `scripts/colab_smoke.py` builds its own `AgentSettings(_env_file=None, …)`, so **no `.env` file and no API key are needed** on Colab. Model selection comes from the `PLUSONE_MODEL` environment variable.

### 4. First run: FLUX.1-schnell (no token, no licence)

```python
!PYTHONPATH=. python scripts/colab_smoke.py
```

What happens:

1. prints the provider capability table;
2. plans the `wedding` brief with the fixture LLM/vision/web adapters → ~10 shots;
3. renders the first `MAX_SHOTS = 4` shots at 1024×1024;
4. writes PNGs to `out/` and unloads the model to free VRAM.

The first run downloads several gigabytes of weights (roughly 12–20 GB depending on the checkpoint), so budget a few minutes for the download and a few tens of seconds per image on a T4.

> The `PYTHONPATH=.` matters: running a file inside `scripts/` puts *that* folder on `sys.path`, not `backend/`, so the import of `pigamagic_agent` fails otherwise.
>
> In a notebook cell you can instead use `%run`, which inherits the notebook path:
>
> ```python
> %run scripts/colab_smoke.py
> ```

### 5. Identity-preserving run: Qwen

`Qwen/Qwen-Image-Edit-2509` is the model that answers "is it really me?" — it keeps facial identity across poses and accepts up to 3 reference images per person. It is also much heavier than FLUX.

```python
%env PLUSONE_MODEL=Qwen/Qwen-Image-Edit-2509
!PYTHONPATH=. python scripts/colab_smoke.py
```

Notes:

- the adapter puts the pipeline in fp16 and enables **CPU offload** automatically on a T4, because the weights exceed VRAM;
- expect several minutes per image on a free T4, and possible out-of-memory on a small runtime. If it OOMs, switch to a high-RAM GPU runtime, or set `WIDTH = HEIGHT = 512` at the top of the script while testing;
- the model is ungated but large; the download alone takes a while.

### 6. Reference photos (the whole point of the identity path)

Upload the photos into `/content/`:

```python
from google.colab import files
uploaded = files.upload()   # pick 1–3 photos per person
```

`REFERENCE_PATHS` is a module constant, so patch it in the notebook before running:

```python
import pathlib
p = pathlib.Path("scripts/colab_smoke.py")
text = p.read_text()
text = text.replace(
    "REFERENCE_PATHS: list[str] = []",
    'REFERENCE_PATHS: list[str] = ["/content/alice.jpg", "/content/bob.jpg"]',
)
p.write_text(text)
```

Then re-run:

```python
!PYTHONPATH=. python scripts/colab_smoke.py
```

How references are wired: the script takes the cast in plan order and assigns each file to a person round-robin (`keys[index % len(keys)]`), so the first file goes to the first cast member. Qwen accepts up to 3 per person; the adapter caps anything beyond that. FLUX.1-schnell is text-to-image only — the script prints a note and ignores the files.

### 7. View and keep the output

```python
import glob
from IPython.display import Image, display

for path in sorted(glob.glob("out/*.png")):
    print(path)
    display(Image(path))
```

Colab deletes `/content` when the runtime is recycled, so copy results to Drive if you want to keep them:

```python
from google.colab import drive
drive.mount("/content/drive")
!mkdir -p /content/drive/MyDrive/plusone && cp out/*.png /content/drive/MyDrive/plusone/
```

### 8. Useful knobs on Colab

Set these in the same cell as the run, or edit the constants at the top of `scripts/colab_smoke.py`:

| Knob | Default | Effect |
| --- | --- | --- |
| `PLUSONE_MODEL` (env) | `black-forest-labs/FLUX.1-schnell` | which checkpoint to load |
| `BRIEF` | `wedding` | any golden brief name, or a path to your own JSON |
| `MAX_SHOTS` | `4` | keep the first run short; full plans are 10+ shots |
| `WIDTH = HEIGHT` | `1024` | `512` is much faster for a first check |
| `REFERENCE_PATHS` | `[]` | reference photos for the identity path |

To render a different brief without editing the file, patch it the same way as `REFERENCE_PATHS`:

```python
text = p.read_text().replace('BRIEF = "wedding"', 'BRIEF = "graduation"')
p.write_text(text)
```

### 9. Colab-specific gotchas

| Symptom | Cause / fix |
| --- | --- |
| `ModuleNotFoundError: pigamagic_agent` | run from `backend/` **with** `PYTHONPATH=.`, or use `%run` |
| `torch is missing` | runtime reverted to CPU or the session reset — reinstall `diffusers transformers accelerate safetensors` |
| No GPU detected | **Runtime → Change runtime type → T4 GPU**, then reconnect |
| Out-of-memory or the kernel dies on Qwen | a free T4 has 16 GB; use a high-RAM runtime, lower `WIDTH`/`HEIGHT`, or render fewer shots |
| `401` / `403` / "gated" on a FLUX dev or Kontext model | accept the licence on the model page and supply an HF token as `AGENT_IMAGE_API_KEY`; or use the ungated `FLUX.1-schnell` |
| Generated faces do not match the references | you are on a text-to-image model, or `REFERENCE_PATHS` is empty; use `Qwen/Qwen-Image-Edit-2509` |
| Session disconnects mid-run | free runtimes expire; render fewer shots per run and save to Drive |

---

## Tests and the eval gate

```bash
cd backend
python -m pytest -q                          # 829 passed, ~100 s, fully offline
python -m pytest -q tests/agent/test_render.py
python -m pigamagic_agent.agent.cli eval     # exit 0 = every golden brief still compiles
```

- Tests never touch the network: fixtures are deterministic and `AGENT_OFFLINE` can stay `true`.
- `tests/agent/golden/prompts/*.json` are golden prompt files, so a change in prompt assembly is a visible diff instead of silent drift.
- The `live` pytest marker is reserved for tests that need real credentials and is skipped by default (`python -m pytest -m live`).
- CI's whole gate is `pytest` plus `cli eval`; a non-zero exit from either fails the build.

---

## Troubleshooting

| Symptom | Cause / fix |
| --- | --- |
| `pydantic_settings` import error | install `backend/requirements.txt` (`pydantic-settings>=2.5`) |
| Settings are ignored | `.env` belongs in the repo root (or `backend/`); `AgentSettings` reads `.env` and `../.env` from the **current directory** |
| `ProviderNotConfigured: Unknown image provider` | the provider name is validated against a fixed vocabulary — see [Provider selection](#provider-selection) |
| `ValueError: identity_similarity_target must be >= …` | the target must not be below the floor |
| `402 insufficient_credits` from the dev server | raise `--credits`, or use the seeded `demo` account |
| `503 admin_disabled` | set `AGENT_ADMIN_TOKEN` and send it as `X-Admin-Token` |
| Photo files missing | check `AGENT_MEDIA_DIR`; relative paths resolve against the working directory, so prefer an absolute path |
| `AGENT_IMAGE_BUDGET_USD=0` did not stop spending | `0` means unlimited by design — set a positive ceiling |
| `python -m pigamagic_agent.agent.cli` → `No module named` | run from `backend/`, which is the package root |
| Colab: identity refs ignored | the selected model is text-to-image; use `qwen` / `Qwen/Qwen-Image-Edit-2509` |

---

## Safety, privacy and consent

- **Consent first.** A run requires attestation that the user may depict everyone referenced. Refusals cost nothing and happen before any provider call.
- **No public figures, no non-consenting third parties, no minors in unsafe, romantic or physically risky contexts.** Sensitive events (memorial, religious/cultural) get a conservative treatment automatically.
- **Flagged media is private everywhere.** The feed, the media URL and the live event stream behave as if it does not exist; only the admin console (with the token) sees it, together with the plan and prompt that produced it.
- **No biometric retention.** Face comparisons are scored for QA and are not persisted.
- **Deletion means deletion.** Removing a reference removes its derived analysis; deleting an account purges media, analyses, photos and run records.
- **Prompt hygiene.** Web pages and user text are data, never instructions.
- **Secrets.** `.env` is gitignored, keys are never logged, and the admin surface fails closed when unconfigured.

---

## Documentation map

| File | Read it for |
| --- | --- |
| [`prd.md`](prd.md) | requirements, personas, success metrics, scope |
| [`design.md`](design.md) | pipeline stages, schemas, run lifecycle, failure handling, module layout |
| [`steps.md`](steps.md) | build order and per-step acceptance checks |
| [`progress.md`](progress.md) | what is done, known open items, resume protocol |
| `backend/tests/agent/` | executable specification — each module's contract, in tests |

---

## License

See [`LICENSE`](LICENSE).
