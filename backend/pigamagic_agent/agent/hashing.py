"""Content hashing and cache keys.

Every cache in the agent is keyed by a hash produced here. Three separate
caches depend on this module being *stable across processes and restarts*:

1. **Reference analysis cache**  keyed by `sha256_bytes(image)`
2. **Venue research cache**      keyed by `normalize_venue_key(venue, event)`
3. **Generation cache**          keyed by `prompt_hash(...)`

Rules that make the caches trustworthy:

- **Deterministic.** No `hash()`, no `id()`, no set iteration order, no floats
  rendered with locale-dependent formatting. Same inputs must produce the same
  digest in a fresh interpreter, otherwise a worker restart re-bills the user.
- **Canonical JSON.** Keys sorted, separators fixed, non-ASCII preserved.
- **Versioned.** `prompt_hash` folds in `settings.hash_relevant_settings()`,
  so bumping `prompt_version` or changing a model deliberately invalidates
  every cached generation instead of silently serving stale photos.
"""

from __future__ import annotations

import hashlib
import json
import re
import unicodedata
from pathlib import Path
from typing import Any, Mapping, Sequence

from pigamagic_agent.agent.config import AgentSettings, get_settings

# Bump when the canonicalisation itself changes shape (not the prompt content).
# This is separate from settings.prompt_version: that says "the prompt text
# changed", this says "how we serialise the prompt changed".
HASH_SCHEME_VERSION = "1"

_READ_CHUNK = 1 << 20  # 1 MiB, so a large upload never lands fully in memory

# Characters stripped when normalising a venue key.
# Apostrophes are DELETED (not replaced with a space) so possessives stay one
# token: "my aunt's garden" -> "my aunts garden", not "my aunt s garden".
_APOSTROPHE_RE = re.compile(r"['\u2019\u02bc`]")
_PUNCT_RE = re.compile(r"[^\w\s]", flags=re.UNICODE)
_SPACE_RE = re.compile(r"\s+")

# Filler words that carry no identity for a venue lookup.
#
# Deliberately conservative: only leading/trailing articles and prepositions are
# noise. Words like "of" are KEPT because they appear inside real proper names
# ("Hall of Fame") and dropping them would merge distinct venues.
_VENUE_FILLER = frozenset(
    {
        "the",
        "a",
        "an",
        "at",
        "in",
        "on",
    }
)


# --------------------------------------------------------------------------
# Content hashing
# --------------------------------------------------------------------------


def sha256_bytes(data: bytes) -> str:
    """Hex SHA-256 of raw bytes. Used as the reference-image content key."""
    if not isinstance(data, (bytes, bytearray, memoryview)):
        raise TypeError("sha256_bytes expects bytes-like input")
    return hashlib.sha256(bytes(data)).hexdigest()


def sha256_file(path: str | Path) -> str:
    """Hex SHA-256 of a file, streamed so large uploads stay off the heap."""
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(_READ_CHUNK), b""):
            digest.update(chunk)
    return digest.hexdigest()


def short_hash(digest: str, length: int = 12) -> str:
    """A human-friendly prefix for logs and IDs. Never use as a cache key."""
    if not 4 <= length <= len(digest):
        raise ValueError(f"length must be between 4 and {len(digest)}")
    return digest[:length]


# --------------------------------------------------------------------------
# Canonical JSON
# --------------------------------------------------------------------------


def canonical_json(value: Any) -> str:
    """Serialise a value so equal data always yields the identical string.

    Sorted keys and fixed separators remove dict-ordering differences. Floats
    are rendered by `repr`, which round-trips exactly, so 0.1 is never
    "0.10000000000000001".
    """
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
        default=_json_default,
    )


def _json_default(value: Any) -> Any:
    """Last-resort serialisation for types canonical_json otherwise rejects."""
    if isinstance(value, (set, frozenset)):
        # Sets are unordered, so sort to stay deterministic.
        return sorted(value, key=canonical_json)
    if isinstance(value, (bytes, bytearray, memoryview)):
        # Hash the payload rather than embedding it: prompts can carry tens of
        # kilobytes of reference image bytes.
        return {"__bytes_sha256__": sha256_bytes(bytes(value))}
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"canonical_json cannot serialise {type(value).__name__}")


def stable_digest(*parts: Any) -> str:
    """Digest any number of JSON-serialisable parts under one scheme version."""
    payload = {"scheme": HASH_SCHEME_VERSION, "parts": list(parts)}
    return sha256_bytes(canonical_json(payload).encode("utf-8"))


# --------------------------------------------------------------------------
# Venue keys
# --------------------------------------------------------------------------


def normalize_venue_key(venue: str, event: str = "") -> str:
    """Normalise a venue name into a shared cache key.

    `"The Sarova Stanley Hotel, Nairobi!"` + `"Wedding"` becomes
    `"sarova stanley hotel nairobi|wedding"`, so two users describing the same
    place two ways hit one cache entry (design.md §6.1).

    Returns `"unknown"` when nothing usable remains, so the caller gets an
    explicit low-information key rather than an empty string.
    """
    venue_part = _normalize_text(venue, drop_filler=True)
    event_part = _normalize_text(event, drop_filler=False)
    if not venue_part:
        return "unknown"
    return f"{venue_part}|{event_part}" if event_part else venue_part


def _normalize_text(text: str, *, drop_filler: bool) -> str:
    if not text:
        return ""
    # Decompose accents so "Café" and "Cafe" collide, then drop the marks.
    decomposed = unicodedata.normalize("NFKD", text)
    stripped = "".join(char for char in decomposed if not unicodedata.combining(char))
    lowered = stripped.casefold()
    # Remove apostrophes before punctuation, so "aunt's" -> "aunts" (one token).
    deapostrophed = _APOSTROPHE_RE.sub("", lowered)
    depunctuated = _PUNCT_RE.sub(" ", deapostrophed)
    tokens = [token for token in _SPACE_RE.sub(" ", depunctuated).strip().split(" ") if token]

    if drop_filler:
        # Strip filler only from the edges, and only while real content remains.
        # This keeps proper names intact ("The Venue" -> "venue", not "unknown")
        # and never deletes a word the user chose from the middle of a name.
        while len(tokens) > 1 and tokens[0] in _VENUE_FILLER:
            tokens.pop(0)
        while len(tokens) > 1 and tokens[-1] in _VENUE_FILLER:
            tokens.pop()

    return " ".join(tokens)


# --------------------------------------------------------------------------
# Plan node IDs
# --------------------------------------------------------------------------


def node_id(index: int) -> str:
    """Stable plan node ID, e.g. 3 -> 's03'.

    Zero-padded so lexical order matches numeric order when a plan is sorted,
    diffed or displayed. Index is 0-based.
    """
    if index < 0:
        raise ValueError("node index must be >= 0")
    if index > 999:
        raise ValueError("node index exceeds 999; a shoot cannot have that many shots")
    return f"s{index:02d}"


# --------------------------------------------------------------------------
# Prompt hashing (the generation cache key)
# --------------------------------------------------------------------------


def prompt_hash(
    *,
    resolved_prompt: str,
    params: Mapping[str, Any] | None = None,
    identity_refs: Mapping[str, Sequence[bytes]] | None = None,
    negative_prompt: str = "",
    plan_node_id: str = "",
    settings: AgentSettings | None = None,
) -> str:
    """Content-addressed key for one generated image.

    Two calls with identical inputs must return the identical key, including
    across processes and restarts, so a retry is a free cache hit
    (design.md §8.1).

    Identity references are folded in **by content hash**, not by media ID, so
    re-uploading the same photo does not orphan the cache and swapping in a
    different face always does invalidate it.
    """
    active = settings or get_settings()

    normalised_refs: dict[str, list[str]] = {}
    for person_key, blobs in (identity_refs or {}).items():
        normalised_refs[str(person_key)] = [sha256_bytes(bytes(blob)) for blob in blobs]

    payload = {
        "prompt": resolved_prompt,
        "negative_prompt": negative_prompt,
        "params": dict(params or {}),
        "identity_refs": normalised_refs,
        "plan_node_id": plan_node_id,
        # Changing a model or prompt_version must invalidate the cache, while
        # credentials and budgets must not (see config.hash_relevant_settings).
        "settings": active.hash_relevant_settings(),
    }
    return stable_digest("prompt", payload)


def reference_analysis_key(image_bytes: bytes) -> str:
    """Cache key for a vision analysis of a reference photo."""
    return stable_digest("reference-analysis", {"image": sha256_bytes(image_bytes)})


def venue_research_key(venue: str, event: str = "") -> str:
    """Cache key for researched venue knowledge."""
    return stable_digest("venue-research", {"venue_key": normalize_venue_key(venue, event)})


__all__ = [
    "HASH_SCHEME_VERSION",
    "canonical_json",
    "node_id",
    "normalize_venue_key",
    "prompt_hash",
    "reference_analysis_key",
    "sha256_bytes",
    "sha256_file",
    "short_hash",
    "stable_digest",
    "venue_research_key",
]
