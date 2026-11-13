# PlusOne — Agent Design: The AI Photographer

> **Scope of this document.** This is the *agent* design for PlusOne: the pipeline that turns an event brief plus reference photos into a structured photoshoot plan and a stream of generated photographs, plus the feedback loop that lets a comment re-direct a single photo. It assumes the product shell already exists (accounts, shoots, studio, feed, albums, likes, comments, credits, admin) and specifies precisely how the AI layer plugs into it.
>
> If no shell exists yet, §2 states what the shell must provide; everything after §2 is the new work.

---

## 0. Build path at a glance

1. **Treat the shoot as a compiled artifact, not a prompt.** A shoot compiles into a versioned `ShootPlan` (JSON) built from *evidence*: reference-photo analysis, venue research, event-type knowledge, and user intent. Every photo is generated from a plan node, never from a raw form. See §3.
2. **Separate the four kinds of knowledge into four different stores.** People and wardrobe live in per-user structured profiles; venue appearance lives in a research cache; event-type knowledge lives in a curated RAG corpus; style lives in a global preset library. Do not mash them into one blob. See §4.
3. **Analyse reference photos once, reuse forever.** A vision pass produces a `PersonProfile` and `WardrobeProfile` keyed by media hash. Shoots reference profiles, not pixels. See §5.
4. **Research the venue as a real retrieval task** with a strict budget: web search → page extraction → image candidates → ranking → a compact `VenueProfile`. Cache by normalized venue key so a second user at the same hotel costs zero. See §6.
5. **Let the LLM write the plan, and only the plan.** Deterministic Python validates the plan against schemas and policy, then schedules generation. LLM output never becomes executable code. See §7.
6. **Generate as an idempotent job DAG per photo,** with content-addressed caching on the resolved prompt, so retries and edits re-run only what changed. See §8.
7. **Stream results as a job, not as a request.** `POST /shoots/{id}/generate` returns a `run_id`; the browser follows Server-Sent Events for per-photo completion and appends to the infinite-scroll feed. See §9, §10.
8. **Make comments a first-class edit job.** A comment compiles into an `EditInstruction` delta against the plan node, producing a *new sibling photo* that remembers its lineage. See §11.
9. **Charge credits on reservation, settle on outcome.** Reserve before enqueue, refund on failure, charge for retries per policy. See §12.
10. **Instrument quality and cost from day one.** Every job records provider, model, latency, token cost, image cost, and a face-similarity score. See §13.

### Decisions worth making up front

| Decision | Recommendation | Why |
| --- | --- | --- |
| Where the AI runs | Background worker consuming a queue; API only enqueues | Image generation takes 10–60 s; HTTP requests must not hold it |
| Plan representation | Typed `ShootPlan` Pydantic model persisted as JSON + revision | Auditable, diffable, replayable, testable without a provider |
| Who decides composition | LLM proposes, Python validates + enforces policy | Safety and cost must not depend on model obedience |
| Identity consistency | Structured `PersonProfile` text + fixed reference image set per person | Text-only identity drifts; reference-image pinning is what holds faces |
| Venue knowledge | Cached research with confidence + provenance | Venue facts are shared, expensive, and often wrong if guessed |
| RAG store | Postgres + pgvector, one table, namespaced by corpus | No extra datastore; already running Postgres |
| Feedback loop | Edit jobs with lineage, never in-place mutation | Users must be able to compare and revert |
| Streaming to UI | SSE over a job-scoped channel backed by Redis | Cheap, resumable, works with plain `EventSource` |
| Failure mode | Jobs fail loudly and visibly, credits refunded | Silent or partially-failed runs destroy trust in the product |

---

## 1. Purpose and boundary

### 1.1 What the agent does

Given a completed brief, the agent must produce photographs that a reasonable person would accept as *photos of these specific people at this specific event*. That means four independent things must be right:

| Axis | Failure looks like |
| --- | --- |
| **Identity** | Faces drift, age/gender/hairstyle change, wrong number of people |
| **Place** | The "hotel ballroom" looks like a generic stock ballroom; the "church" has the wrong architecture |
| **Event** | A graduation with no gown, a wedding with no ceremony, activities that don't happen at that event type |
| **Craft** | Bad light, malformed hands, impossible perspective, repeated identical framings |

Most amateur attempts at this product fail on **place** and **event** because they only feed the model the user's text. PlusOne's differentiator is that it *researches* the venue and the event type and *derives* a shot list, so the photos look like they were taken at the real place.

### 1.2 What the agent deliberately does not do

- **It does not run code produced by an LLM.** No generated Python, shell, SQL, or provider API payloads beyond typed, validated fields.
- **It does not invent people.** Nobody appears in a photo who is not in the shoot's cast, which is built from the user's reference photos and explicit descriptions.
- **It does not identify real people.** Vision analysis describes appearance for synthesis. It never asserts a name from a face, never matches faces against an external database, and never retains biometric embeddings beyond the user's own shooting session (see §14).
- **It does not publish anything.** Photos are private until the owner marks them public; safety flags block publication (§14).
- **It does not guarantee factual accuracy about the venue.** It maximizes plausibility and records confidence; when research is thin it falls back to event-type conventions and says so in the plan.

### 1.3 Success criteria

| Metric | Target | Measured by |
| --- | --- | --- |
| Face similarity to reference (same person) | cosine ≥ 0.62 median on a 50-person eval set | Automated eval harness (§13) |
| Venue plausibility, human rating | ≥ 80% "looks like this kind of place" | Blind rating of 100 outputs |
| Plan validity, first attempt | ≥ 95% pass schema + policy validation | Worker metrics |
| Cost per photo (all-in) | ≤ 1 credit, tracked in KES | Ledger reconciliation |
| Time to first photo in feed | ≤ 25 s p50, ≤ 60 s p95 | Job telemetry |
| Rejected-photo rate (user deletes) | ≤ 20% | Delete events joined to jobs |

---

## 2. Assumed shell (integration contract)

The agent hangs off these existing surfaces. Names follow the current PlusOne backend; if a shell is built fresh, provide equivalents.

```
POST /api/v1/shoots/{shoot_id}/schedule     -> validate + compile plan + create run      (§7, §9)
POST /api/v1/shoots/{shoot_id}/generate     -> enqueue run (credits reserved)            (§9, §12)
GET  /api/v1/runs/{run_id}/events           -> SSE stream of job + photo events           (§10)
GET  /api/v1/runs/{run_id}                  -> run status, counts, per-photo state        (§10)
POST /api/v1/photos/{photo_id}/comments     -> create comment + optional edit run         (§11)
GET  /api/v1/photos/{photo_id}/lineage      -> siblings / parent for compare & revert     (§11)
```

Required tables already present in the shell: `users`, `media`, `studio_items`, `photo_shoots`, `shoot_references`, `generation_jobs`, `generated_photos`, `photo_likes`, `photo_comments`, `albums`, `album_photos`, `credit_entries`, `payments`, `activity_events`.

**The one gap.** Today `schedule_shoot` only logs its parameters and `generate_shoot_feed` returns bundled demo images. Those two functions are the entire integration point. Everything in this document is what replaces them.

### 2.1 New tables introduced by the agent

```sql
-- Structured understanding of a reference photo (cached by content hash)
reference_analysis      (id, media_id, media_sha256, kind, profile_json,
                         model, prompt_version, created_at)

-- One row per person appearing in a shoot
shoot_cast              (id, shoot_id, profile_json, source_analysis_ids jsonb,
                         role, position)

-- Researched venue, cached by normalized key
venue_research          (id, venue_key, venue_name, profile_json, confidence,
                         sources_json, fetched_at, expires_at)

-- Event-type convention pack (curated, seeded, versioned)
event_conventions       (id, event_key, pack_json, version)

-- RAG corpus for venue/event/activity/style knowledge
knowledge_chunks        (id, corpus, source_url, title, content, embedding vector(1536),
                         metadata jsonb, created_at)

-- A compiled plan revision
shoot_plans             (id, shoot_id, run_id, revision, plan_json, status,
                         compile_cost, created_at)

-- One generation run of a shoot
generation_runs         (id, shoot_id, owner_id, plan_id, status, requested_count,
                         reserved_credits, spent_credits, error, created_at, finished_at)

-- Per-photo job with prompt caching and lineage
photo_jobs              (id, run_id, plan_node_id, prompt_hash, resolved_prompt,
                         provider, model, params jsonb, status, attempt,
                         parent_photo_id, edit_instruction_json,
                         cost_credits, cost_kes, latency_ms, error,
                         created_at, started_at, finished_at)
```

Indexes that matter: `photo_jobs(run_id, status)`, `photo_jobs(prompt_hash)` (cache lookup), `shoot_cast(shoot_id)`, `venue_research(venue_key)` unique, `knowledge_chunks` ivfflat/HNSW on `embedding`, `reference_analysis(media_sha256)` unique.

---

## 3. The pipeline in detail

### 3.1 Stages

```
                       ┌──────────────────────────────────────────────┐
   brief + refs ──────►│ 1. Ingest & analyse                          │
                       │    hash refs → vision → Person/Wardrobe      │
                       │    → ShootCast, SceneHints                   │
                       └───────────────────┬──────────────────────────┘
                                           ▼
                       ┌──────────────────────────────────────────────┐
                       │ 2. Research (parallel, budgeted)             │
                       │    venue web + images → VenueProfile         │
                       │    event + activities → ConventionPack       │
                       │    RAG over corpus → style & detail hints    │
                       └───────────────────┬──────────────────────────┘
                                           ▼
                       ┌──────────────────────────────────────────────┐
                       │ 3. Compile plan (LLM proposes, Python checks)│
                       │    ShotList[] → PromptSpec[] → validated     │
                       └───────────────────┬──────────────────────────┘
                                           ▼
                       ┌──────────────────────────────────────────────┐
                       │ 4. Render (fan-out, idempotent, cached)      │
                       │    each PromptSpec → image job → QA → Media  │
                       └───────────────────┬──────────────────────────┘
                                           ▼
                       ┌──────────────────────────────────────────────┐
                       │ 5. Publish to feed as it lands (SSE)         │
                       └──────────────────────────────────────────────┘
```

Stages 1 and 2 run in parallel because they are independent. Stage 3 needs both. Stage 4 fans out. Stage 5 interleaves with 4.

### 3.2 Why the plan is a first-class artifact

The single most important design choice: **the plan is stored, versioned, diffable data**, not a transient list of prompts.

Consequences that justify it:

- **Reproducibility.** Given a `plan_id` you can regenerate the identical shoot, which is required for support, for "regenerate this one photo", and for cost debugging.
- **Cheap edits.** A comment changes one node's `EditInstruction`; every other photo is untouched and does not re-run.
- **Testability.** The plan compiler can be tested against golden briefs with no image provider at all.
- **Auditability.** Admin can see exactly what was asked for, and safety review has structured fields rather than a wall of text.
- **Partial failure recovery.** If 2 of 12 nodes fail, the plan already knows what was requested.

### 3.3 Stage contracts

Each stage is a pure-ish function with typed input and output, so each can be tested and swapped:

```python
class AnalyseRefs(In):  shoot: PhotoShoot, references: list[ShootReference]
class AnalyseRefs(Out): cast: ShootCast, scene_hints: SceneHints, warnings: list[str]

class ResearchVenue(In):  venue: str, event: str, hints: SceneHints
class ResearchVenue(Out): profile: VenueProfile, confidence: float, sources: list[Source]

class CompilePlan(In):   shoot, cast, venue, conventions, style_brief
class CompilePlan(Out):  plan: ShootPlan, validation: ValidationReport

class RenderPlan(In):    plan: ShootPlan, cast: ShootCast
class RenderPlan(Out):   photo_job ids (streamed)
```

---

## 4. Knowledge architecture: four stores, four jobs

Do not let one retrieval store pretend to hold all knowledge. The four kinds differ in **ownership, volatility, and trust**:

| Store | Contents | Owner | Volatility | Trust | Retrieval |
| --- | --- | --- | --- | --- | --- |
| **User profiles** | faces, hair, build, wardrobe, relationships | One user | Low | High (user-supplied) | Direct FK lookup |
| **Venue cache** | what this specific place looks like | Shared | Low (TTL 180 d) | Medium (web) | Exact key → else research |
| **Event conventions** | what happens at a graduation | Curated | Very low | High (human-curated) | Exact key → else LLM prior |
| **RAG corpus** | style, composition, detail vocabulary | Shared | Medium | Medium | Vector + filters |

**Rule:** retrieval is used only where the answer is *open-ended* (style, detail vocabulary, photographic technique). Whenever the answer is *specific and checkable* (a person's hair, a venue's architecture), it comes from a structured record, not from semantic search. Semantic search over your own database to answer "what colour is her hair" is how products ship wrong facts.

### 4.1 Profile schemas

These are the typed payloads that make the pipeline legible.

```python
class PersonProfile(BaseModel):
    person_key: str                 # stable within a shoot, e.g. "p1"
    label: str                      # "bride", "the groom's brother"
    age_band: Literal["child","teen","adult","senior"]
    gender_presentation: Literal["masculine","feminine","androgynous","unspecified"]
    face: FaceFeatures
    hair: HairFeatures
    build: Literal["slim","average","athletic","curvy","plus","unspecified"]
    skin_tone: str                  # descriptive, not a label
    distinguishing: list[str]       # beard, glasses, freckles, scar, tattoo
    reference_media_ids: list[str]  # the images pinned at generation time
    confidence: float

class FaceFeatures(BaseModel):
    shape: str; eyes: str; brows: str; nose: str; mouth: str; jaw: str

class HairFeatures(BaseModel):
    length: str; style: str; colour: str; texture: str

class WardrobeProfile(BaseModel):
    person_key: str
    garments: list[Garment]         # name, colour, fabric, fit
    accessories: list[str]          # veil, boutonnière, watch, earrings
    overall_style: Literal["formal","smart-casual","casual","traditional","athletic","festive"]
    must_keep: list[str]            # e.g. "the red dress"
    avoid: list[str]                # e.g. "no hats", inferred from "never wears them"

class VenueProfile(BaseModel):
    venue_key: str
    name: str; kind: str            # "hotel ballroom", "Catholic church", "hiking trail"
    indoor_outdoor: Literal["indoor","outdoor","mixed"]
    architecture: list[str]         # "Gothic arches", "floodlit terrace"
    materials: list[str]            # "travertine", "pale oak"
    palette: list[str]              # dominant colours
    lighting: list[str]             # "warm uplighting", "windows on north wall"
    signature_details: list[str]    # "the long reflecting pool", "arched colonnade"
    layout: str                     # how spaces connect
    time_of_day_evidence: list[str]
    confidence: float
    sources: list[str]

class SceneHints(BaseModel):
    """What the reference images reveal about place and group, beyond the brief."""
    group_size: int | None
    group_dynamics: list[str]       # "affectionate", "formal and posed"
    clothing_style: str
    glimpsed_venue_features: list[str]
    mood: str
```

**Design rule:** every profile field is *descriptive and visual*, never an identity claim. `label` comes from the user's own words. `age_band` exists only to keep rendering plausible; it is not a protected-attribute decision surface, it is a description of the photo the user supplied.

### 4.2 Style presets

A small curated library, stored as `event_conventions` rows with `event_key` prefixed `style.`:

| Preset | Look |
| --- | --- |
| `editorial` | 85 mm, directional light, deep falloff, magazine finish |
| `candid-documentary` | 35 mm, available light, motion, grain |
| `warm-celebration` | golden bounce, vibrant but not oversaturated |
| `cinematic-lowlight` | teal shadows, practical lights, shallow depth |
| `bright-outdoor` | open shade, high key, clean skies |

A preset is a *rendering contract*, resolved into concrete provider parameters in stage 4. Users pick a preset, never raw model settings.

---

## 5. Reference photo analysis

### 5.1 One pass, hashed, cached

Analysis is expensive and deterministic per image, so hash first:

```
sha256(bytes) → reference_analysis(media_sha256) hit? → reuse
                                       miss? → vision pass → persist
```

A user who reused the same portrait across four shoots pays for it once.

### 5.2 The vision pass

One structured-output call per image, not per person per image (cheaper, and it keeps group composition intact):

```python
class ReferenceAnalysis(BaseModel):
    people: list[PersonProfile]
    wardrobe: list[WardrobeProfile]
    group: GroupObservation            # who stands with whom, relative heights
    venue_cues: list[str]
    scene_quality: Literal["usable","too-dark","blurry","portrait-closeup"]
    notes: str
```

Then a **deterministic fusion step** merges per-image people into shoot-level identities using the user's ordering plus the image descriptions, never face matching:

```
for each image, for each person:
    if image.description names a person ("my sister Ada"), attach the label
    else assign to the nearest unlabelled identity by ordinal position
    merge profiles field-by-field, preferring the sharpest image for face fields
```

Fusion is where quality is won: preferring the **sharpest, most frontal, well-lit** crop for facial fields gives far better identity anchoring than averaging everything.

### 5.3 Prompt hygiene for the analysis call

The vision model is instructed to:

- Describe appearance in photographic terms (shape, colour, texture), not identity.
- Report a confidence per field and **omit fields it cannot see** rather than guess.
- Never guess ethnicity, religion, or nationality.
- Flag `scene_quality` honestly — a blurry reference should degrade to "use the description text, don't anchor on pixels".

### 5.4 What happens with no usable reference photos

Users who are camera-shy often have bad photos, or none. Fallback ladder:

1. **Text-only cast.** Build a `PersonProfile` from the reference *descriptions* and the shoot story. Mark `confidence: low`.
2. **Studio cast.** The existing `studio_items` tables already hold `people` and `wardrobe` categories — those are reusable references across shoots and should be offered as a first resort.
3. **Render one cheap "cast portrait" per person** as an additional identity anchor before the main run (cost: one photo per person). This is the only place the agent generates something the user did not ask for; make it opt-in and visible.

---

## 6. Venue and event research

### 6.1 Venue research pipeline

```
normalize(venue, event) → key              # "sarova stanley nairobi|wedding"
  ↓ cache hit (and not expired) → return
  ↓ miss
1. Query planning (LLM, cheap model, structured output)
     → 3–5 search queries: interior, exterior, ceremony space, grounds, "photos"
2. Web search (budgeted: ≤ 6 calls)
3. Extract top 4–6 pages → clean text → per-page summary
4. Image candidates from pages (≤ 8) → dedupe by perceptual hash → rank
5. Vision pass on top 2–3 images → architectural vocabulary
6. Synthesize VenueProfile from text + images, with confidence
7. Persist; queue image embedding into the RAG corpus if high confidence
```

**Budget and failure.** If searches return nothing, or the venue is a generic description ("my aunt's garden"), synthesising a fake profile is worse than admitting ignorance. Emit `confidence: low` with `signature_details` from the *event conventions* pack instead, and record the choice in the plan so the UI can say "venue details estimated".

**Caching key.** Normalize aggressively (`lower`, strip punctuation, drop city variants, alias table for known chains). A shared cache is the main cost lever: the hundredth wedding at the same hotel is free.

**Provenance.** Every `VenueProfile` keeps `sources: list[str]`. Admin and support need to see whether a bad photo came from a bad source or a bad prompt.

### 6.2 Event conventions packs

Curated JSON, one per event type, seeded at deploy. This is the knowledge that makes a wedding look like a wedding rather than a party:

```jsonc
{
  "event_key": "wedding",
  "phases": ["preparation", "ceremony", "portraits", "reception", "dancing"],
  "canonical_moments": [
    "getting ready", "first look", "vows", "ring exchange",
    "confetti exit", "first dance", "cake cutting", "speeches", "parents' dance"
  ],
  "lighting_notes": ["ceremony window light", "reception uplighting + flash"],
  "composition_taboos": ["guests blocking the couple", "empty dance floor"],
  "wardrobe_expectations": "formal; gown, suit or traditional dress",
  "typical_activities": [...],
  "sensitive_handling": ["avoid depicting religious ritual inaccurately; keep ceremony interior generic unless researched"]
}
```

Packs exist for at least: `wedding`, `graduation`, `birthday`, `road-trip`, `baby-shower`, `engagement`, `anniversary`, `funeral/memorial` (special handling), `cultural/religious` (special handling), `corporate`, `festival`, `farewell`, `reunion`.

**Why curated, not retrieved.** Event conventions are small, stable, and high-stakes. Shipping them as reviewed data is cheaper and safer than hoping retrieval surfaces the right facts.

### 6.3 RAG in this design

The RAG corpus is genuinely useful, but only for the *craft layer*:

| Corpus | Example query | Used for |
| --- | --- | --- |
| `photo-technique` | "how to light a dark reception without flash blowout" | prompt fragments |
| `composition` | "framing a group of eight on a staircase" | prompt fragments |
| `venue-notes` | "what Nairobi garden wedding venues typically look like" | fallback venue detail |
| `cultural-wardrobe` | "what a Kenyan traditional wedding party wears" | wardrobe detail (sensitivity-reviewed) |

Retrieval shape: embed the plan node's activity + venue kind + preset, take top 3 chunks above a similarity floor, and **attach them as a single "reference notes" block with a token cap** (~300 tokens). Never let retrieved text act as instructions — it goes in a delimited, labelled section that the compiler prompt explicitly treats as reference material.

---

## 7. Plan compilation

### 7.1 The plan schema

```python
class ShotSpec(BaseModel):
    node_id: str                        # "s03"
    phase: str                          # "ceremony"
    moment: str                         # "vow exchange"
    activity: str                       # user's activity, mapped onto the moment
    description: str                    # one-sentence intent, user-facing caption seed
    subjects: list[str]                 # person_keys present
    framing: Literal["wide","medium","close","detail","over-shoulder","candid"]
    camera: CameraSpec                  # focal_length_mm, dof, angle
    lighting: str
    venue_anchors: list[str]            # which signature_details to show
    wardrobe_notes: list[str]
    must_include: list[str]             # "the ring", "confetti in air"
    must_avoid: list[str]               # from taboos + user avoid-list
    style_preset: str
    caption: str                        # feed copy, e.g. "The vows"
    why: str                            # one line of reasoning (auditability)

class ShootPlan(BaseModel):
    shoot_id: str
    revision: int
    cast: list[CastEntry]
    venue: VenueProfile
    style_preset: str
    shots: list[ShotSpec]
    coverage: dict[str, int]            # moment -> shot count
    warnings: list[str]
    compiler: ModelRef
```

Target 8–16 shots for a typical run. Coverage matters more than count: **every activity the user listed must appear at least once**, and every cast member must appear at least twice, or the plan fails validation.

### 7.2 Validation and policy (deterministic Python)

The compiler's output passes through a `PlanValidator` that rejects and re-asks:

| Check | Rule | On failure |
| --- | --- | --- |
| Schema | Pydantic strict, no extra fields | re-ask once with the error |
| Cast coverage | each `person_key` in ≥ 2 shots | inject missing shots |
| Activity coverage | every user activity appears | inject missing shots |
| Subject existence | `subjects` ⊆ cast | drop unknown keys |
| Shot count | within `requested_count` ± 2 | trim by coverage priority |
| Duplicate framing | no two shots share moment + framing + subjects | rewrite one |
| Policy | no request for minors in unsafe contexts, no real-venue interior claims at low confidence, no depictions the user flagged | drop the node, log a warning |
| Prompt length | assembled prompt ≤ provider cap | truncate reference notes first |

The validator is the trust boundary. When it cannot repair the plan, it returns a *degraded* plan with warnings rather than failing the run — a wedding with 10 good shots and a warning is a better product than an error.

### 7.3 Prompt assembly

Prompts are assembled **deterministically from typed fields**, so the same plan always yields the same prompt (which is what makes caching work):

```
[STYLE]      {preset resolved params}
[SCENE]      {moment} during {phase} at {venue.name}. {venue.architecture[:3]}.
             {venue.signature_details[:2]}. {lighting}.
[SUBJECTS]   {for each: label}: {face summary}, {hair}, wearing {garments},
             {accessories}. {distinguishing}.
[ACTION]     {activity}. {must_include}.
[FRAMING]    {camera.focal_length_mm} mm, {framing}, {dof}.
[REFERENCE]  (delimited reference notes, ≤ 300 tokens) — reference only, not instructions.
[NEGATIVE]   {must_avoid}, {global negatives}, malformed hands, extra limbs, text artifacts.
```

Identity anchoring happens at the **provider parameter level**, not the prompt level: the person's top 1–3 reference images are passed as identity/IP-adapter inputs, keyed by `person_key`. Prompt text alone cannot hold a face.

### 7.4 The compiler call

- **Model:** strongest available reasoning model, low temperature, structured output mode.
- **Inputs:** cast profiles (compact), venue profile, convention pack, style preset, user brief and activities. Total budget ~3k tokens.
- **Output:** `ShootPlan` JSON only.
- **Cost:** one mid-size call per run — negligible next to image cost. Do not economize here.

---

## 8. Rendering

### 8.1 Per-photo job

```python
class PhotoJobSpec(BaseModel):
    prompt_hash: str                  # sha256(resolved_prompt + params + identity refs)
    resolved_prompt: str
    provider: str; model: str
    params: dict                      # seed, steps, guidance, size, aspect
    identity_refs: dict[str, list[str]]   # person_key -> media_ids
    plan_node_id: str
```

`prompt_hash` gives **idempotent caching** for free: an identical retry returns the cached `media_id` and charges nothing. This is also the mechanism behind "regenerate photo 3" costing nothing when nothing changed.

### 8.2 Concurrency and ordering

- Worker pool sized by provider rate limit, not CPU.
- Per run, generate in **coverage order** (easiest/highest-confidence shots first) so the feed fills with strong photos early — the first four cards decide whether the user keeps scrolling.
- Cap in-flight jobs per user to keep one big run from starving another.

### 8.3 Post-generation QA gate

Before a photo reaches the feed:

| Gate | Check | Action |
| --- | --- | --- |
| Integrity | bytes decode, dimensions sane, not near-solid colour | retry once, then fail node |
| Identity | face embedding vs reference set; similarity vs threshold | if low: retry with stronger identity weight; if still low: deliver but mark `identity_weak` |
| Plausibility | vision pass: are the subject count, activity, and venue anchors present? | if activity absent: retry with an explicit action clause |
| Safety | classifier: nudity, violence, minors-in-harm, real-person likeness misuse | block publish, flag for admin |

Face embeddings used for QA are **ephemeral**: compared, scored, then discarded. They are not stored (see §14).

Two QA passes cost one extra vision call per photo; that is cheap relative to a wasted generation and a lost user.

### 8.4 Provider abstraction

```python
class ImageProvider(Protocol):
    async def generate(self, spec: PhotoJobSpec) -> GeneratedImage: ...
    async def price(self, spec: PhotoJobSpec) -> CostEstimate: ...
```

One adapter per provider; the plan never names a provider, the *config* does (so the same plan renders on a cheaper model for previews). Keep adapters thin and typed: no provider SDK object escapes its adapter.

---

## 9. Runs, endpoints, and the credit ledger

### 9.1 Run lifecycle

```
POST /shoots/{id}/schedule   →  compile plan (sync check, cheap)
POST /shoots/{id}/generate   →  reserve credits, create run, enqueue, return run_id
        run: queued → researching → compiling → rendering → done | partial | failed
```

`partial` is a legitimate terminal state: some nodes succeeded, some failed after retries. The feed shows what exists; the run row explains the shortfall.

### 9.2 Credit reservation

| Step | Ledger action |
| --- | --- |
| Generate requested | `CreditEntry(-N, "reservation", job/run)` and `users.credits -= N` atomically |
| Node succeeds | no action (reservation stands) |
| Node fails after retries | `CreditEntry(+1, "refund: render_failed")` |
| Edit job | its own reservation |
| Cached hit (no render) | refund the node's reservation |

Reserve-then-settle prevents the classic failure: user has 5 credits, requests 12 photos, half the run fails, and they cannot retry. It also blocks double-submit abuse (`SELECT ... FOR UPDATE` on the user row, or a unique partial index on active runs per shoot).

### 9.3 Idempotency and abuse

- `Idempotency-Key` header on `POST /generate`; duplicate keys return the same `run_id`.
- One active run per shoot (`unique (shoot_id) where status in ('queued','researching','compiling','rendering')`).
- Rate limits: generate 5/hour/user, edits 20/hour/user, analysis 30/hour/user.
- Global budget guard: if provider spend exceeds the daily ceiling, new runs stay `queued` and the UI explains why.

---

## 10. Streaming to the feed

### 10.1 Why SSE, not a request that streams images

The user must be able to leave the page, come back, and find the same run. That means the run must outlive the connection, so the stream is a **notification channel over persisted state**, not the transport for image bytes.

### 10.2 Events

```
event: run
data: {"run_id":"...","status":"compiling","plan_id":"...","total":12}

event: photo
data: {"photo_id":"...","node_id":"s03","caption":"The vows",
       "image_url":"/api/v1/media/...","activity":"vows","phase":"ceremony"}

event: photo_failed
data: {"node_id":"s07","reason":"render_failed","refunded":true}

event: run_done
data: {"status":"done","generated":12,"failed":0,"spent_credits":11}
```

Backed by Redis pub/sub per run channel, with the `photo_jobs` and `generated_photos` rows as the source of truth. On reconnect, the client sends `Last-Event-ID`; the server replays from the DB, so a dropped connection never loses photos.

### 10.3 Frontend behaviour

1. Post the shoot, get `run_id`, route to the feed with a "developing your photos" strip at the top.
2. Open `EventSource`; append each `photo` event as a real feed card (the same card component used for settled photos — no separate "loading" markup to maintain).
3. The existing `IntersectionObserver` sentinel still paginates the settled feed; streaming cards are prepended, so scroll position stays put.
4. Each card shows a subtle `developing` state until QA passes, then becomes interactive (like/download/save/comment). Optimistic like/save as today.
5. On `run_done`, replace the strip with a summary of count and any failures the user can retry in one click.

---

## 11. The feedback loop: comments as edit jobs

### 11.1 Comment → instruction

A comment on a photo is *simultaneously* social feedback and a generation instruction. Distinguish them at the API boundary:

```python
class CommentCreate(BaseModel):
    body: str
    mode: Literal["feedback", "edit"] = "feedback"     # default: it's a comment
    reference_media_ids: list[str] = []                # optional uploaded images
```

- `feedback` — stored, shown on the card, no job.
- `edit` — compiles into an `EditInstruction` and enqueues a child job. Explicit choice, because generating is expensive and sometimes the user is just saying "love this one".

An `EditInstruction` is a **delta against the plan node**, produced by a small LLM call constrained to a fixed vocabulary:

```python
class EditInstruction(BaseModel):
    base_node_id: str
    change: Literal["subject","activity","framing","lighting","venue_emphasis",
                    "wardrobe","style","remove_person","add_person","regenerate"]
    detail: str                        # the user's intent, normalized
    preserves: list[str]               # fields held constant (identity, venue, wardrobe)
    reference_media_ids: list[str]
```

Compiling to a delta rather than free text keeps identity stable: `preserves` explicitly carries forward the person keys and venue anchors, so "make me smile more" cannot silently change her dress.

### 11.2 Lineage and compare

Child photos store `parent_photo_id` and `edit_instruction_json`. The UI offers a compare view (parent vs child), and the feed shows edits as new cards rather than replacing originals. `GET /photos/{id}/lineage` returns siblings so a user can see all takes and delete the ones they dislike.

**Never mutate a delivered photo.** Replacing a photo the user already liked or saved is a trust violation and breaks album references.

### 11.3 The edit prompt

Rebuild from the parent's `PhotoJobSpec`, then apply the delta:

```
base resolved prompt (parent)
  + [EDIT]  {change}: {detail}; preserve {preserves}
  + image-conditioned: parent image as structural reference (img2img / edit endpoint)
  + any new reference images as identity/wardrobe anchors
```

Seed handling: use a *new* seed by default (the user wants a different result) but keep the seed fixed across identical retries so retries are reproducible.

---

## 12. Cost model

Approximate per-photo unit costs, used to set credits and guard rails:

| Stage | Cost driver | Notes |
| --- | --- | --- |
| Reference analysis | vision tokens × images | cached by hash; near-zero on reuse |
| Venue research | 3–6 web calls + 1–3 vision calls + LLM synthesis | cached by venue key; the biggest per-venue cost |
| Plan compile | one reasoning call | negligible |
| Render | image model × count | dominant cost |
| QA | 1 vision call + 1 embedding per photo | ~10–15% of render cost |
| Edit | render + QA + small LLM | same as render |

**Policy:** a photo costs 1 credit. A run of 12 with 1 failure and 1 cache hit charges 10. Research and compilation are not separately charged on first run, but a *re-run with a changed venue* is (the venue cache is new work). Refunds are automatic and visible in the dashboard ledger.

---

## 13. Observability and evaluation

### 13.1 Telemetry per job

`photo_jobs` already carries `provider, model, prompt_hash, attempt, cost_kes, latency_ms, status, error`. Add counters for `qa_identity_score`, `qa_activity_present`, `cache_hit`. Ship these into the admin dashboard as: cost/photo, p50/p95 latency, failure reason breakdown, retry rate, cache hit rate.

### 13.2 Offline eval harness

A fixed set of briefs (10 venues × 5 event types, with and without references) scored automatically:

| Test | Assertion |
| --- | --- |
| Plan validity | 100% of golden briefs compile without repair |
| Coverage | every activity and every person appears in ≥ 1 / ≥ 2 shots |
| Prompt stability | same plan → byte-identical prompt |
| Identity (with refs) | median face similarity ≥ 0.62; no member below 0.45 |
| Venue fidelity | vision check confirms ≥ 2 signature details present |
| Cost | median cost/photo within 15% of budget |

This runs on prompt-template or model changes. Without it, every "small" prompt tweak ships a silent regression in face fidelity — the most expensive bug in this product class.

### 13.3 Prompt versioning

Every stored analysis, plan, and job records `prompt_version`. Bumping a template invalidates caches deliberately (`prompt_version` is part of `prompt_hash`), which is what lets you tell whether a quality drop was the model or the prompt.

---

## 14. Safety, privacy, and consent

This product synthesises photorealistic images of real people. The following are requirements, not options.

| Concern | Control |
| --- | --- |
| **Consent of depicted people** | The brief includes an explicit attestation that the user has permission to depict everyone referenced. Reject references where the user has no relationship; surface this in the UI in plain language. |
| **Minors** | Detected minor age bands are rendered only in benign, family contexts; never in swimwear-altering, romantic, or physically risky scenes. Hard policy block in `PlanValidator`. |
| **Real-person impersonation** | Refuse to depict public figures or anyone the user does not attest to knowing; admin-reviewable flag on suspicious briefs. |
| **Biometric storage** | Face embeddings for QA are computed in-memory, scored, discarded. No embedding table. `reference_analysis` stores visual descriptions, not vectors. |
| **Reference photo retention** | References are the user's media; deleting a reference deletes the derived analysis. Deleting the shoot cascades. Account deletion purges media and analyses. |
| **Output safety** | Pre-publish classifier; `safety_flagged` photos are private, unshareable, and admin-visible. Public sharing requires an explicit opt-in per photo (already modelled by `is_public`). |
| **Provenance** | Every `generated_photos` row links to its `photo_job` and thus its plan, model, and prompt. An admin can answer "why does this exist" for any photo. |
| **Sensitive events** | Memorials, religious and cultural ceremonies get a conservative mode: no invented ritual detail, no fabricated attendees' emotional states, generic venue interiors when confidence is low. |
| **Prompt injection** | Web pages and user text are delimited and labelled as data; retrieved text is never placed in an instruction position. Ref uses are sanitized to vocabulary lists where possible. |

---

## 15. Module layout

Mirrors the existing backend structure (`app/api/*`, `app/models.py`, `app/schemas.py`):

```
backend/app/agent/
  __init__.py
  profiles.py        # PersonProfile, WardrobeProfile, VenueProfile, SceneHints schemas
  plan.py            # ShotSpec, ShootPlan, EditInstruction schemas
  analyze.py         # reference hashing, vision pass, deterministic fusion -> ShootCast
  research.py        # venue research pipeline, cache, provenance, budget
  conventions.py     # event pack loading + validation
  rag.py             # corpus ingest, embed, retrieve, token-capped context block
  compile.py         # plan compiler call + PlanValidator + repair loop
  prompts.py         # deterministic prompt assembly + prompt_version
  providers/
    base.py          # ImageProvider / VisionProvider / LLMProvider protocols
    images.py        # adapter(s)
    vision.py
    llm.py
  render.py          # fan-out, prompt_hash cache, retries, provider adapter use
  qa.py              # integrity, identity, plausibility, safety gates
  edits.py           # comment -> EditInstruction -> child job
  run.py             # run orchestration, credits reservation/settlement, SSE publish
  events.py          # Redis pub/sub channel helpers
app/api/runs.py      # /runs/{id}, /runs/{id}/events (SSE)
```

Wiring in existing files:

- `app/api/photos.py::schedule_shoot` — call `compile.compile_plan` and persist a `shoot_plans` row.
- `app/api/photos.py::generate_shoot_feed` — delete; replaced by `run.start_run`.
- `app/api/photos.py::add_comment` — branch on `mode == "edit"` into `edits.start_edit`.
- `app/api/__init__.py` — include `runs_router`.
- `app/models.py` — add the §2.1 tables.
- `app/schemas.py` — add `CommentCreate.mode`, `run` and `plan` response models.
- `worker/` — a separate process (`python -m app.worker`) consuming the queue; Redis is already a dependency.

---

## 16. Failure handling

| Failure | Behaviour |
| --- | --- |
| Vision provider down during analysis | proceed text-only, mark cast `confidence: low`, warn in plan |
| Venue research returns nothing | use conventions pack, `confidence: low`, plan warning, no extra charge |
| Planner returns invalid JSON twice | fall back to a **procedural plan** derived from the conventions pack (phases × activities), warn |
| One render fails after 2 retries | refund node, emit `photo_failed`, run continues as `partial` |
| Whole provider outage | pause the run, keep it resumable, notify user; no credits consumed for unrendered nodes |
| QA safety block | photo stays private and flagged; user sees "this one needs a review", admins see why |
| SSE disconnect | client reconnects with `Last-Event-ID`; state replayed from DB |
| Worker crash mid-job | queue visibility timeout → job returns to `queued`, `attempt += 1` |

**Rule:** never let a failed generation look like a slow one. Every terminal failure surfaces in the UI with a retry affordance and a refund line in the ledger.

---

## 17. Roadmap

**Milestone 1 — Plan without pixels.**
Schemas, profile fusion, conventions packs, plan compiler, `PlanValidator`, golden-brief tests. Deliverable: a CLI that prints a complete `ShootPlan` for a brief. No provider account required. This is also where the product's quality thesis is proven or falsified.

**Milestone 2 — One real photo.**
Provider adapter, prompt assembly, single-node render, QA gate, media persistence. Deliverable: one brief → one validated photo saved as a `generated_photo`. Establish the cost baseline here.

**Milestone 3 — Runs and streaming.**
Queue, run orchestration, credit reservation/settlement, `runs` API, SSE, feed cards that grow live. Deliverable: brief → 12 photos streaming into the existing infinite-scroll UI.

**Milestone 4 — Research.**
Venue web + image pipeline with cache and provenance, RAG corpus seeding, low-confidence fallbacks. Deliverable: photos that visibly look like the chosen venue.

**Milestone 5 — Edits.**
Comment modes, `EditInstruction` compiler, lineage, compare UI, per-edit credits. Deliverable: "make it brighter and put us by the pool" produces a new sibling photo that keeps their faces.

**Milestone 6 — Hardening.**
Eval harness in CI, cost dashboards, safety review console, abuse limits, account-deletion purge.

---

## 18. Acceptance checklist (milestone 1)

- [ ] `ShootPlan` compiles from a brief with no provider credentials configured
- [ ] `PlanValidator` rejects missing cast/activity coverage and repairs it
- [ ] Every golden brief passes validation on the first attempt
- [ ] Same brief + same version → byte-identical prompts
- [ ] Reference analysis reuses cache on identical bytes (no second vision call)
- [ ] Venue research returns `confidence` and `sources` for every profile
- [ ] Low-confidence venue path produces a plan with explicit warnings
- [ ] Prompt injection string in `story` cannot alter instruction position
- [ ] Every plan node is traceable to a moment in the conventions pack
- [ ] Unit tests cover fusion preference (sharpest image wins face fields)
