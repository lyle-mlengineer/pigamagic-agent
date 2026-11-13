# PlusOne — Product Requirements Document (PRD)

**Product:** PlusOne — a personal AI photographer
**Document status:** Requirements baseline for the agent build
**Related documents:** `design.md` (technical design), `steps.md` (build order), `progress.md` (resume state)

---

## 1. Problem

Most people cannot get good photographs of themselves at the moments that matter.

Three specific failures drive this:

| Who | Why they fail | What they lose |
| --- | --- | --- |
| **People with weak phone cameras** | Low-light reception halls, cheap sensors, heavy compression | The one photo of the first dance looks like noise |
| **Camera-shy people** | They avoid being photographed, or freeze when a phone is raised | They are missing from their own family history |
| **People who take bad photos** | Bad framing, bad timing, no understanding of light | 200 blurry photos and no usable one |

The shared outcome: the event happens, everyone is there, and there is no beautiful record of it. Professional photography solves this but costs money and cannot be arranged retroactively.

**PlusOne's premise:** the user already knows what happened. Give the system enough structured truth about the people, the place, the event and the activities, and it can produce a convincing, beautiful photo story of a day that was poorly photographed.

---

## 2. Goals and non-goals

### 2.1 Goals

| ID | Goal |
| --- | --- |
| G1 | Turn a short event brief plus reference photos into a set of photographs the user is happy to share |
| G2 | Keep the same people recognisable across every photo in a shoot |
| G3 | Make the venue and event type look real, not generic |
| G4 | Let the user correct any single photo by describing the change, without losing the rest |
| G5 | Keep cost predictable and legible through a credit ledger |
| G6 | Never publish, never impersonate, never retain biometric data |

### 2.2 Non-goals (explicit)

- **Not** a face-swap or deepfake tool. No depiction of anyone the user has not attested to knowing.
- **Not** an image editor. It generates new photographs; it does not retouch existing ones beyond guided regeneration.
- **Not** a social network. Sharing is opt-in per photo and per user; the feed is personal-first.
- **Not** a video product, a print product, or a marketplace in this phase.
- **Not** a real-time camera app.

---

## 3. Users

| Persona | Context | Primary need |
| --- | --- | --- |
| **The under-photographed celebrant** | Graduation, birthday, engagement | "I want photos that look like they were taken by a professional" |
| **The family historian** | Reunion, wedding, memorial | "Everyone who was there should be in the record" |
| **The budget-conscious planner** | Community or low-cost wedding | "Professional coverage is out of reach" |
| **The camera-shy creator** | Any event | "I want to appear in photos without the awkwardness" |

**Primary journey.** Register → create a photoshoot (title, event type, venue, activities, story) → upload reference photos with descriptions → generate → scroll the feed → like/save/download the good ones → comment to fix the bad ones.

---

## 4. Functional requirements

Requirement IDs are stable and referenced by `steps.md` and `design.md`.

### 4.1 Accounts and identity (FR-1)

| ID | Requirement |
| --- | --- |
| FR-1.1 | Register with email, password, full name, handle |
| FR-1.2 | Log in, log out; sessions are server-side and revocable |
| FR-1.3 | View and edit profile: name, handle, bio, avatar |
| FR-1.4 | Delete account, purging media, derived analyses, photos and run records |
| FR-1.5 | Admins can ban accounts and view activity |

### 4.2 Reference studio (FR-2)

| ID | Requirement |
| --- | --- |
| FR-2.1 | Upload reference images (JPEG/PNG/WebP, ≤15 MB) |
| FR-2.2 | Categorise references: events, locations, activities, wardrobe, people |
| FR-2.3 | Caption each reference; captions are inputs to analysis |
| FR-2.4 | Search and filter own references |
| FR-2.5 | References are private; no cross-user visibility |

### 4.3 Photoshoot brief (FR-3)

| ID | Requirement |
| --- | --- |
| FR-3.1 | Create a shoot with title, event type, venue, one or more activities, story |
| FR-3.2 | Attach up to 8 reference images, each with a description |
| FR-3.3 | Multi-step creation form, resumable as a draft |
| FR-3.4 | Edit and delete a shoot |
| FR-3.5 | Attest that the user has consent to depict everyone referenced |

### 4.4 Generation pipeline (FR-4) — *the core of this build*

| ID | Requirement |
| --- | --- |
| FR-4.1 | Analyse reference photos into structured person and wardrobe profiles |
| FR-4.2 | Research the venue from the web, including images, with provenance and confidence |
| FR-4.3 | Resolve the event type against a curated conventions pack |
| FR-4.4 | Compile a validated photoshoot plan from cast, venue, conventions and style preset |
| FR-4.5 | Every user-listed activity appears in at least one planned shot |
| FR-4.6 | Every person in the cast appears in at least two planned shots |
| FR-4.7 | Render each planned shot as an independent, retryable, cacheable job |
| FR-4.8 | QA each photo for integrity, identity fidelity, activity presence and safety before it is shown |
| FR-4.9 | Stream photos into the feed as each one passes QA |
| FR-4.10 | Degrade gracefully: partial runs, low-confidence research, and provider failures never look like silence |

### 4.5 Feed and interaction (FR-5)

| ID | Requirement |
| --- | --- |
| FR-5.1 | Infinite-scroll feed, Instagram-like, one photo per card |
| FR-5.2 | Each card shows the shoot context and a short activity summary |
| FR-5.3 | Like / unlike a photo |
| FR-5.4 | Download a photo |
| FR-5.5 | Save a photo to an album; create an album inline |
| FR-5.6 | Cursor-based pagination, stable under inserts |
| FR-5.7 | Newly generated photos appear without a page reload |

### 4.6 Albums (FR-6)

| ID | Requirement |
| --- | --- |
| FR-6.1 | Create, view, edit, delete albums with title and description |
| FR-6.2 | Add and remove photos |
| FR-6.3 | Search and filter albums |
| FR-6.4 | Album cover derived from contents unless set |

### 4.7 Feedback loop — comment-to-edit (FR-7)

| ID | Requirement |
| --- | --- |
| FR-7.1 | Comment on any photo the user can see |
| FR-7.2 | A comment may be marked as an *edit request*, which generates a new sibling photo |
| FR-7.3 | Edit requests may include uploaded reference images |
| FR-7.4 | An edit preserves identity, wardrobe and venue unless the user explicitly changes them |
| FR-7.5 | Edited photos retain lineage to their parent; originals are never mutated |
| FR-7.6 | The user can compare and delete takes |

### 4.8 Credits and payments (FR-8)

| ID | Requirement |
| --- | --- |
| FR-8.1 | Credit packages purchased; M-Pesa primary, card backup |
| FR-8.2 | Credits are reserved before a run and settled after |
| FR-8.3 | Failed nodes are refunded automatically and visibly |
| FR-8.4 | Ledger of every credit movement, readable by the user |
| FR-8.5 | Payment callbacks are signed and idempotent |

### 4.9 Admin (FR-9)

| ID | Requirement |
| --- | --- |
| FR-9.1 | List, search, edit, ban users |
| FR-9.2 | View runs and per-node failures |
| FR-9.3 | Review safety-flagged photos with their generating prompt and plan |
| FR-9.4 | View payment success/failure and revenue |
| FR-9.5 | View cost per photo and failure-rate dashboards |

---

## 5. Non-functional requirements

| ID | Requirement |
| --- | --- |
| NFR-1 | Python + FastAPI backend, vanilla HTML/CSS/JS frontend, no build step |
| NFR-2 | PostgreSQL for data, Redis for queue and pub/sub, blob storage behind an interface |
| NFR-3 | Time to first photo in feed ≤ 25 s p50, ≤ 60 s p95 |
| NFR-4 | All generation runs in background workers; no HTTP request holds an image job |
| NFR-5 | Every generation is reproducible from a stored plan revision |
| NFR-6 | Rate limits on upload, generate, edit and analysis routes |
| NFR-7 | Responsive UI; the feed is usable on a mid-range phone |
| NFR-8 | Provider-agnostic adapters for LLM, vision and image models |
| NFR-9 | No face embeddings persisted after QA scoring |
| NFR-10 | All money figures in KES; all credits integers |

---

## 6. Safety and consent requirements

| ID | Requirement |
| --- | --- |
| SR-1 | Explicit consent attestation before a run |
| SR-2 | Minors never depicted in unsafe, romantic, or physically risky contexts |
| SR-3 | No depiction of public figures or non-consenting third parties |
| SR-4 | Safety classifier gates publication; flagged photos are private and admin-visible |
| SR-5 | Reference photo deletion removes derived analysis; account deletion purges everything |
| SR-6 | Web and user text are treated as data, never as instructions |
| SR-7 | Sensitive events (memorials, religious/cultural ceremonies) use a conservative mode |

---

## 7. Success metrics

| Metric | Target |
| --- | --- |
| Face similarity to reference (median, cosine) | ≥ 0.62 |
| Venue plausibility (human rating) | ≥ 80% |
| Plan validity first attempt | ≥ 95% |
| Time to first photo | ≤ 25 s p50 |
| User-deleted photo rate | ≤ 20% |
| Cost per photo | ≤ 1 credit |
| Runs reaching `done` or `partial` | ≥ 98% |

---

## 8. Assumptions and dependencies

- An existing product shell provides accounts, media, studio, shoots, feed, albums, likes, comments, credits and admin. Where it does not exist, `steps.md` Step 0 provides it.
- Image, vision and LLM providers are available behind adapters; no single vendor is hard-coded.
- Postgres with the `vector` extension is available for RAG.
- Users supply honest briefs and have consent to depict the people referenced.

---

## 9. Out of scope for this phase

- Video generation, print ordering, physical delivery.
- Multi-user collaboration on one shoot.
- Real-time face capture or live camera guidance.
- Marketplace for presets or photographers.
- Non-English UI (English only in this phase; content may be non-English).
