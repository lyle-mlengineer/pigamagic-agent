"""Curated event convention packs (design.md §6.2).

Event conventions are small, stable and high-stakes, so they ship as reviewed
JSON rather than being retrieved at runtime. They are the knowledge that makes
a wedding look like a wedding instead of a party:

    phases              the arc of the day, in order
    canonical_moments    the moments a photographer would actually capture
    lighting_notes      how this event is normally lit
    composition_taboos  shots that immediately look wrong
    wardrobe_expectations
    typical_activities  used when the user supplies fewer than they need
    emotion_notes
    sensitive_handling  conservative-mode rules (memorial, religious/cultural)

Loading is strict: a malformed pack fails loudly at import/test time, because a
silently-degraded pack would produce subtly wrong photos that are hard to debug.
Resolution never raises: an unknown event type falls back to `generic` with a
warning, since a celebration with slightly generic framing is a far better
product outcome than an error page.
"""

from __future__ import annotations

import json
import re
from functools import lru_cache
from pathlib import Path
from typing import Annotated

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

CONVENTIONS_DIR = Path(__file__).resolve().parent / "data" / "conventions"

# Packs that must exist for the product to be usable.
REQUIRED_PACKS = ("generic", "wedding", "graduation", "birthday", "road-trip")

# Event types that must run in conservative mode regardless of user input.
SENSITIVE_EVENTS = frozenset({"memorial", "religious-cultural"})

ShortText = Annotated[str, Field(max_length=160)]
Note = Annotated[str, Field(max_length=400)]


class ConventionPack(BaseModel):
    """One reviewed event-type knowledge pack."""

    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    event_key: Annotated[str, Field(pattern=r"^[a-z][a-z0-9-]{1,40}$")]
    display_name: ShortText
    phases: list[ShortText] = Field(min_length=1, max_length=12)
    canonical_moments: list[ShortText] = Field(min_length=1, max_length=60)
    lighting_notes: list[Note] = Field(default_factory=list, max_length=20)
    composition_taboos: list[Note] = Field(default_factory=list, max_length=20)
    wardrobe_expectations: Note = ""
    typical_activities: list[ShortText] = Field(default_factory=list, max_length=30)
    emotion_notes: list[ShortText] = Field(default_factory=list, max_length=20)
    sensitive_handling: list[Note] = Field(default_factory=list, max_length=20)

    @field_validator(
        "phases",
        "canonical_moments",
        "lighting_notes",
        "composition_taboos",
        "typical_activities",
        "emotion_notes",
        "sensitive_handling",
    )
    @classmethod
    def _clean_and_dedupe(cls, value: list[str]) -> list[str]:
        seen: set[str] = set()
        cleaned: list[str] = []
        for item in value:
            text = item.strip()
            if not text:
                continue
            key = text.casefold()
            if key in seen:
                continue
            seen.add(key)
            cleaned.append(text)
        return cleaned

    @model_validator(mode="after")
    def _moments_and_phases_are_coherent(self) -> "ConventionPack":
        if not self.canonical_moments:
            raise ValueError(f"pack {self.event_key!r} has no canonical moments")
        if not self.phases:
            raise ValueError(f"pack {self.event_key!r} has no phases")
        return self

    @property
    def is_sensitive(self) -> bool:
        """True only for events that require conservative mode.

        This is deliberately NOT keyed off `sensitive_handling`: most packs
        carry some care note (a wedding must not invent ritual detail), but
        conservative mode is much stricter and would make ordinary shoots
        bland if applied everywhere. Only the events listed in
        SENSITIVE_EVENTS qualify.
        """
        return self.event_key in SENSITIVE_EVENTS

    @property
    def has_care_notes(self) -> bool:
        """True when this pack carries any extra care instructions."""
        return bool(self.sensitive_handling)

    def moment_for_activity(self, activity: str) -> str | None:
        """Best canonical moment for a user-supplied activity, if any.

        Matching is deliberately conservative: an exact (case-insensitive)
        activity wins, then substring in either direction. Guessing further
        would silently mislabel a shot.
        """
        needle = activity.strip().casefold()
        if not needle:
            return None
        for activity_option in self.typical_activities:
            if activity_option.casefold() == needle:
                return self._moment_mentioning(activity_option) or activity_option
        for moment in self.canonical_moments:
            if needle in moment.casefold() or moment.casefold() in needle:
                return moment
        for activity_option in self.typical_activities:
            option = activity_option.casefold()
            if needle in option or option in needle:
                return self._moment_mentioning(activity_option) or activity_option
        return None

    def _moment_mentioning(self, activity: str) -> str | None:
        needle = activity.casefold()
        for moment in self.canonical_moments:
            lowered = moment.casefold()
            if needle in lowered or lowered in needle:
                return moment
        return None


class ConventionResolution(BaseModel):
    """The outcome of resolving a user's event string."""

    model_config = ConfigDict(extra="forbid")

    pack: ConventionPack
    matched_alias: str | None = None
    used_fallback: bool = False
    warnings: list[str] = Field(default_factory=list)

    @property
    def event_key(self) -> str:
        return self.pack.event_key


# --------------------------------------------------------------------------
# Aliases
# --------------------------------------------------------------------------

# Maps user language onto a pack key. Matching is exact-token based after
# normalisation, then substring, so "my wedding ceremony" finds "wedding".
ALIASES: dict[str, tuple[str, ...]] = {
    "wedding": (
        "wedding",
        "weddings",
        "matrimony",
        "marriage",
        "marriage ceremony",
        "wedding ceremony",
        "wedding reception",
        "nikkah",
        "white wedding",
        "traditional wedding",
        "bridal",
    ),
    "graduation": (
        "graduation",
        "graduations",
        "convocation",
        "commencement",
        "graduating",
        "graduate",
        "graduation ceremony",
    ),
    "birthday": (
        "birthday",
        "birthdays",
        "bday",
        "birthday party",
        "sweet sixteen",
        "milestone birthday",
        "turns 18",
        "turns 21",
        "turns 30",
        "turns 40",
        "turns 50",
        "turns 60",
        "turning 30",
        "turn up",
    ),
    "road-trip": (
        "road trip",
        "roadtrip",
        "road-trip",
        "drive",
        "driving trip",
        "road journey",
    ),
    "baby-shower": ("baby shower", "baby-shower", "shower", "baby sprinkle"),
    "engagement": ("engagement", "engagement party", "proposal", "introduction"),
    "anniversary": ("anniversary", "anniversaries", "jubilee"),
    "corporate": (
        "corporate",
        "corporate event",
        "conference",
        "company event",
        "retreat",
        "team building",
        "product launch",
        "gala",
    ),
    "festival": (
        "festival",
        "concert",
        "gig",
        "music festival",
        "carnival",
        "show",
    ),
    "reunion": ("reunion", "class reunion", "family reunion", "get-together", "get together"),
    "farewell": ("farewell", "send-off", "send off", "sendoff", "going away", "leaving party"),
    "memorial": (
        "memorial",
        "funeral",
        "memorial service",
        "burial",
        "wake",
        "remembrance",
        "celebration of life",
    ),
    "religious-cultural": (
        "religious",
        "cultural",
        "religious ceremony",
        "cultural ceremony",
        "church",
        "mosque",
        "temple",
        "traditional ceremony",
        "rites",
        "ruracio",
        "dowry ceremony",
    ),
    "generic": ("generic", "celebration", "gathering", "party", "event"),
}


def _normalize(text: str) -> str:
    """Lowercase, collapse punctuation and whitespace, for alias matching."""
    cleaned = []
    for char in text.casefold():
        cleaned.append(char if char.isalnum() or char.isspace() else " ")
    return " ".join("".join(cleaned).split())


# "turns 30", "turning 50", "turned 21" are unambiguously birthdays and are far
# too numerous to list as aliases, so they are matched as a pattern.
_TURNS_AGE_RE = re.compile(r"\bturn(?:s|ed|ing)?\s+\d{1,3}\b")
_PATTERN_EVENTS: tuple[tuple[re.Pattern[str], str], ...] = (
    (_TURNS_AGE_RE, "birthday"),
    (re.compile(r"\b\d{1,3}(?:st|nd|rd|th)\s+birthday\b"), "birthday"),
)


def _alias_index() -> dict[str, str]:
    index: dict[str, str] = {}
    for key, aliases in ALIASES.items():
        for alias in aliases:
            index[_normalize(alias)] = key
    return index


# --------------------------------------------------------------------------
# Loading
# --------------------------------------------------------------------------


@lru_cache
def load_conventions(directory: Path | None = None) -> dict[str, ConventionPack]:
    """Load and validate every pack in the conventions directory.

    Raises if a pack is malformed or a required pack is missing: this runs once
    at import/startup so a broken pack is a test failure, not a runtime surprise.
    """
    root = directory or CONVENTIONS_DIR
    if not root.is_dir():
        raise FileNotFoundError(f"Conventions directory not found: {root}")

    packs: dict[str, ConventionPack] = {}
    for path in sorted(root.glob("*.json")):
        raw = json.loads(path.read_text(encoding="utf-8"))
        try:
            pack = ConventionPack.model_validate(raw)
        except Exception as exc:  # re-raise with the file named
            raise ValueError(f"Invalid convention pack {path.name}: {exc}") from exc
        if pack.event_key in packs:
            raise ValueError(f"Duplicate convention pack event_key {pack.event_key!r} in {path.name}")
        packs[pack.event_key] = pack

    missing = [key for key in REQUIRED_PACKS if key not in packs]
    if missing:
        raise ValueError(f"Missing required convention pack(s): {missing}")

    if packs["generic"].event_key not in packs:
        raise ValueError("A 'generic' convention pack is required as the fallback")

    return packs


def get_pack(event_key: str) -> ConventionPack:
    """Fetch one pack by exact key. Raises KeyError for unknown keys."""
    packs = load_conventions()
    if event_key not in packs:
        raise KeyError(f"no convention pack for {event_key!r}")
    return packs[event_key]


def resolve_event_key(event: str) -> str | None:
    """Map a user's event string to a pack key, or None if nothing matches."""
    needle = _normalize(event)
    if not needle:
        return None

    packs = load_conventions()
    # 1. Exact pack key.
    if needle in packs:
        return needle

    index = _alias_index()

    # 2. Exact alias.
    if needle in index:
        return index[needle]

    # 3. Pattern-based events, e.g. "turns 30", "my 40th birthday".
    for pattern, key in _PATTERN_EVENTS:
        if pattern.search(needle):
            return key

    # 4. Longest alias contained in the text ("my cousin's wedding ceremony").
    candidates = sorted(index, key=len, reverse=True)
    for alias in candidates:
        if alias and alias in needle:
            return index[alias]

    # 5. Pack key mentioned directly.
    for key in sorted(packs, key=len, reverse=True):
        if key != "generic" and key in needle:
            return key

    return None


def resolve_event(event: str) -> ConventionResolution:
    """Resolve an event string to a pack, falling back to `generic`.

    Never raises for unknown input: a fallback with a warning is always more
    useful to the user than an error.
    """
    packs = load_conventions()
    key = resolve_event_key(event)
    if key is not None and key in packs:
        matched = None if key == _normalize(event) else event.strip()
        return ConventionResolution(pack=packs[key], matched_alias=matched, used_fallback=False)

    warnings = [
        f"No convention pack matched the event {event!r}; using the generic pack. "
        "Photos will be plausible but less specific to this event type."
    ]
    return ConventionResolution(pack=packs["generic"], used_fallback=True, warnings=warnings)


def pack_keys() -> list[str]:
    """All loaded pack keys, sorted. Used by the API and admin."""
    return sorted(load_conventions())


__all__ = [
    "ALIASES",
    "CONVENTIONS_DIR",
    "ConventionPack",
    "ConventionResolution",
    "REQUIRED_PACKS",
    "SENSITIVE_EVENTS",
    "get_pack",
    "load_conventions",
    "pack_keys",
    "resolve_event",
    "resolve_event_key",
]
