"""Step 3: curated event convention packs.

The properties that matter:
- every shipped pack is valid and internally coherent
- resolution never raises: unknown events fall back to `generic` + warning
- aliases resolve user language onto the right pack
- sensitive events are flagged for conservative handling
"""

import json
from pathlib import Path

import pytest

from app.agent.conventions import (
    ALIASES,
    CONVENTIONS_DIR,
    SENSITIVE_EVENTS,
    ConventionPack,
    get_pack,
    load_conventions,
    pack_keys,
    resolve_event,
    resolve_event_key,
)

# --------------------------------------------------------------------------
# Loading
# --------------------------------------------------------------------------


@pytest.fixture(scope="module")
def packs() -> dict[str, ConventionPack]:
    return load_conventions()


def test_required_packs_are_present(packs: dict[str, ConventionPack]) -> None:
    for required in ("generic", "wedding", "graduation", "birthday", "road-trip"):
        assert required in packs, f"missing required pack {required!r}"


def test_all_expected_packs_load(packs: dict[str, ConventionPack]) -> None:
    expected = {
        "generic",
        "wedding",
        "graduation",
        "birthday",
        "road-trip",
        "baby-shower",
        "engagement",
        "anniversary",
        "corporate",
        "festival",
        "reunion",
        "farewell",
        "memorial",
        "religious-cultural",
    }
    assert expected <= set(packs)


def test_every_pack_has_moments_and_phases(packs: dict[str, ConventionPack]) -> None:
    for key, pack in packs.items():
        assert pack.canonical_moments, f"{key} has no canonical moments"
        assert pack.phases, f"{key} has no phases"


def test_every_pack_key_matches_its_filename(packs: dict[str, ConventionPack]) -> None:
    for path in CONVENTIONS_DIR.glob("*.json"):
        raw = json.loads(path.read_text(encoding="utf-8"))
        assert raw["event_key"] == path.stem, f"{path.name} has event_key {raw['event_key']!r}"


def test_moments_are_deduplicated_case_insensitively() -> None:
    pack = ConventionPack(
        event_key="test-pack",
        display_name="x",
        phases=["a"],
        canonical_moments=["The cake", "the cake", "  The CAKE  ", "dancing"],
    )
    assert pack.canonical_moments == ["The cake", "dancing"]


def test_pack_requires_at_least_one_moment() -> None:
    with pytest.raises(ValueError):
        ConventionPack(
            event_key="test-pack", display_name="x", phases=["a"], canonical_moments=[]
        )


def test_pack_rejects_unknown_fields() -> None:
    with pytest.raises(ValueError):
        ConventionPack(
            event_key="test-pack",
            display_name="x",
            phases=["a"],
            canonical_moments=["m"],
            mystery=1,
        )


def test_malformed_pack_directory_raises(tmp_path: Path) -> None:
    (tmp_path / "bad.json").write_text('{"event_key": "bad"}', encoding="utf-8")
    with pytest.raises(ValueError) as excinfo:
        load_conventions(tmp_path)
    assert "Invalid convention pack" in str(excinfo.value)


def test_missing_required_packs_raise(tmp_path: Path) -> None:
    pack = {
        "event_key": "generic",
        "display_name": "g",
        "phases": ["a"],
        "canonical_moments": ["m"],
    }
    (tmp_path / "generic.json").write_text(json.dumps(pack), encoding="utf-8")
    with pytest.raises(ValueError) as excinfo:
        load_conventions(tmp_path)
    assert "Missing required" in str(excinfo.value)


def test_get_pack_by_key_and_unknown_key() -> None:
    assert get_pack("wedding").event_key == "wedding"
    with pytest.raises(KeyError):
        get_pack("teleportation")


def test_pack_keys_are_sorted(packs: dict[str, ConventionPack]) -> None:
    assert pack_keys() == sorted(packs)


# --------------------------------------------------------------------------
# Resolution: aliases
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("event", "expected"),
    [
        ("wedding", "wedding"),
        ("Wedding", "wedding"),
        ("wedding ceremony", "wedding"),
        ("my cousin's wedding reception", "wedding"),
        ("nikkah", "wedding"),
        ("graduation", "graduation"),
        ("Convocation ceremony", "graduation"),
        ("bday", "birthday"),
        ("Amina turns 30", "birthday"),
        ("my dad is turning 50", "birthday"),
        ("she turned 21 last week", "birthday"),
        ("Jomo's 40th birthday", "birthday"),
        ("birthday party", "birthday"),
        ("road trip", "road-trip"),
        ("Roadtrip up the coast", "road-trip"),
        ("baby shower", "baby-shower"),
        ("engagement party", "engagement"),
        ("25th anniversary", "anniversary"),
        ("company retreat", "corporate"),
        ("music festival", "festival"),
        ("class reunion", "reunion"),
        ("send-off", "farewell"),
        ("funeral", "memorial"),
        ("traditional ceremony", "religious-cultural"),
    ],
)
def test_aliases_resolve_to_the_right_pack(event: str, expected: str) -> None:
    assert resolve_event_key(event) == expected


def test_pack_key_is_resolvable_directly() -> None:
    for key in pack_keys():
        assert resolve_event_key(key) == key


def test_empty_event_does_not_resolve() -> None:
    assert resolve_event_key("") is None
    assert resolve_event_key("   ") is None
    assert resolve_event_key("!!!") is None


def test_every_alias_points_at_a_real_pack(packs: dict[str, ConventionPack]) -> None:
    for key in ALIASES:
        assert key in packs, f"alias table references missing pack {key!r}"


# --------------------------------------------------------------------------
# Resolution: fallback never raises
# --------------------------------------------------------------------------


def test_unknown_event_falls_back_to_generic_with_warning() -> None:
    resolution = resolve_event("underground cheese-rolling contest")

    assert resolution.used_fallback is True
    assert resolution.pack.event_key == "generic"
    assert resolution.warnings
    assert "generic" in resolution.warnings[0]


def test_known_event_uses_no_fallback() -> None:
    resolution = resolve_event("wedding")
    assert resolution.used_fallback is False
    assert resolution.pack.event_key == "wedding"
    assert resolution.warnings == []


def test_resolution_never_raises_for_arbitrary_input() -> None:
    for event in ("", "  ", "?!", "x" * 500, "a wedding but also a funeral"):
        resolution = resolve_event(event)
        assert resolution.pack.event_key


def test_resolve_event_exposes_event_key() -> None:
    assert resolve_event("graduation").event_key == "graduation"


# --------------------------------------------------------------------------
# Sensitive events
# --------------------------------------------------------------------------


@pytest.mark.parametrize("key", sorted(SENSITIVE_EVENTS))
def test_sensitive_packs_carry_handling(packs: dict[str, ConventionPack], key: str) -> None:
    pack = packs[key]
    assert pack.sensitive_handling, f"{key} must carry sensitive_handling rules"
    assert pack.is_sensitive is True


def test_ordinary_pack_is_not_sensitive() -> None:
    """A wedding carries care notes but does NOT need conservative mode."""
    wedding = get_pack("wedding")
    assert wedding.is_sensitive is False
    assert wedding.has_care_notes is True


def test_memorial_pack_forbids_celebratory_framing(packs: dict[str, ConventionPack]) -> None:
    memorial = packs["memorial"]
    joined = " ".join(memorial.composition_taboos + memorial.sensitive_handling).casefold()
    assert "celebrat" in joined or "dignity" in joined or "grief" in joined


# --------------------------------------------------------------------------
# Content sanity: packs must be usable by the planner
# --------------------------------------------------------------------------


def test_wedding_has_the_expected_arc(packs: dict[str, ConventionPack]) -> None:
    wedding = packs["wedding"]
    assert {"preparation", "ceremony", "reception"} <= set(wedding.phases)
    lower = [moment.casefold() for moment in wedding.canonical_moments]
    for expected in ("vow", "ring", "dance", "cake"):
        assert any(expected in moment for moment in lower), f"wedding missing {expected}"


def test_graduation_mentions_certificate_and_cap(packs: dict[str, ConventionPack]) -> None:
    joined = " ".join(packs["graduation"].canonical_moments).casefold()
    assert "certificate" in joined
    assert "cap" in joined


def test_wardrobe_expectations_are_non_empty_for_formal_events(packs: dict[str, ConventionPack]) -> None:
    for key in ("wedding", "graduation", "memorial"):
        assert packs[key].wardrobe_expectations, f"{key} should state wardrobe expectations"


def test_composition_taboos_exist_for_every_pack(packs: dict[str, ConventionPack]) -> None:
    for key, pack in packs.items():
        assert pack.composition_taboos, f"{key} should list at least one taboo"


def test_moment_for_activity_matches_exactly(packs: dict[str, ConventionPack]) -> None:
    moment = packs["wedding"].moment_for_activity("cutting the cake")
    assert moment is not None
    assert "cake" in moment.casefold()


def test_moment_for_activity_returns_none_for_unrelated(packs: dict[str, ConventionPack]) -> None:
    assert packs["wedding"].moment_for_activity("skateboarding") is None


def test_moment_for_activity_uses_substring_matching(packs: dict[str, ConventionPack]) -> None:
    moment = packs["graduation"].moment_for_activity("tossing caps in the air")
    assert moment is not None
    assert "cap" in moment.casefold()
