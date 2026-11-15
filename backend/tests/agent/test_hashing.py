"""Step 0.3: hashing primitives that make the agent's caches trustworthy.

The load-bearing property is **cross-process determinism**: a worker restart
must not miss the cache and re-bill the user. Anything that could vary between
processes (hash randomisation, dict order, float formatting) is tested for.
"""

import json
import subprocess
import sys
from pathlib import Path

import pytest

from pigamagic_agent.agent.config import AgentSettings
from pigamagic_agent.agent.hashing import (
    canonical_json,
    node_id,
    normalize_venue_key,
    prompt_hash,
    reference_analysis_key,
    sha256_bytes,
    sha256_file,
    short_hash,
    stable_digest,
    venue_research_key,
)

BACKEND_DIR = Path(__file__).resolve().parents[2]

# --------------------------------------------------------------------------
# Content hashing
# --------------------------------------------------------------------------


def test_sha256_bytes_matches_known_vector() -> None:
    # Well-known digest of b"abc".
    assert sha256_bytes(b"abc") == "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"


def test_sha256_bytes_is_stable_and_byte_sensitive() -> None:
    assert sha256_bytes(b"photo") == sha256_bytes(b"photo")
    assert sha256_bytes(b"photo") != sha256_bytes(b"photo ")


def test_sha256_bytes_rejects_non_bytes() -> None:
    with pytest.raises(TypeError):
        sha256_bytes("a string, not bytes")  # type: ignore[arg-type]


def test_sha256_file_matches_in_memory_hash(tmp_path: Path) -> None:
    payload = b"x" * (3 << 20)  # 3 MiB, forcing multiple read chunks
    path = tmp_path / "ref.jpg"
    path.write_bytes(payload)

    assert sha256_file(path) == sha256_bytes(payload)


def test_short_hash_bounds() -> None:
    digest = sha256_bytes(b"abc")
    assert short_hash(digest, 8) == digest[:8]
    assert len(short_hash(digest)) == 12
    with pytest.raises(ValueError):
        short_hash(digest, 2)
    with pytest.raises(ValueError):
        short_hash(digest, len(digest) + 1)


# --------------------------------------------------------------------------
# Canonical JSON
# --------------------------------------------------------------------------


def test_canonical_json_ignores_dict_key_order() -> None:
    assert canonical_json({"b": 1, "a": 2}) == canonical_json({"a": 2, "b": 1})


def test_canonical_json_is_compact_and_unicode_preserving() -> None:
    assert canonical_json({"a": 1, "b": 2}) == '{"a":1,"b":2}'
    # Non-ASCII stays readable rather than becoming \uXXXX escapes.
    assert "café" in canonical_json({"name": "café"})


def test_canonical_json_handles_nested_and_ordered_containers() -> None:
    left = {"outer": [{"y": 1, "x": 2}, {"z": 3}]}
    right = {"outer": [{"x": 2, "y": 1}, {"z": 3}]}
    assert canonical_json(left) == canonical_json(right)

    # Lists are ordered: reordering them must change the digest.
    assert canonical_json([1, 2]) != canonical_json([2, 1])


def test_canonical_json_sorts_sets_deterministically() -> None:
    assert canonical_json({"tags": {"b", "a", "c"}}) == canonical_json({"tags": {"c", "b", "a"}})


def test_canonical_json_hashes_bytes_instead_of_embedding_them() -> None:
    """Reference image bytes must never be inlined into a prompt payload."""
    rendered = canonical_json({"blob": b"hello"})
    assert json.loads(rendered)["blob"]["__bytes_sha256__"] == sha256_bytes(b"hello")
    assert "hello" not in rendered


def test_canonical_json_rejects_nan() -> None:
    with pytest.raises(ValueError):
        canonical_json({"value": float("nan")})


def test_stable_digest_changes_with_content() -> None:
    assert stable_digest("a", {"x": 1}) == stable_digest("a", {"x": 1})
    assert stable_digest("a", {"x": 1}) != stable_digest("a", {"x": 2})
    assert stable_digest("a", {"x": 1}) != stable_digest("b", {"x": 1})


# --------------------------------------------------------------------------
# Venue normalisation
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("The Sarova Stanley Hotel, Nairobi!", "sarova stanley hotel nairobi"),
        ("  the   sarova   stanley  ", "sarova stanley"),
        ("Sarova-Stanley Hotel", "sarova stanley hotel"),
        ("CAFÉ MOMBASA", "cafe mombasa"),
        # "venue" is content, not filler: it is part of the name here.
        ("Venue: Hall of Fame", "venue hall of fame"),
        # Mid-name words must survive: dropping "of" would merge distinct venues.
        ("Hall of Fame", "hall of fame"),
        # Possessives stay one token rather than becoming "aunt s".
        ("My aunt's garden", "my aunts garden"),
        ("My aunt\u2019s garden", "my aunts garden"),
    ],
)
def test_normalize_venue_key_collapses_variants(raw: str, expected: str) -> None:
    assert normalize_venue_key(raw) == expected


def test_normalize_venue_key_includes_event() -> None:
    key = normalize_venue_key("Kempinski", "Wedding Ceremony")
    assert key == "kempinski|wedding ceremony"


def test_different_venues_do_not_collide() -> None:
    assert normalize_venue_key("Kempinski") != normalize_venue_key("Serena")


def test_normalize_venue_key_never_returns_empty() -> None:
    """An unusable name gets an explicit low-information key, not ""."""
    assert normalize_venue_key("") == "unknown"
    assert normalize_venue_key("   ") == "unknown"
    assert normalize_venue_key("!!!") == "unknown"


def test_normalize_venue_key_does_not_erase_a_name_made_of_filler() -> None:
    # "The Venue" is a real (if unhelpful) name; it must not become "unknown".
    assert normalize_venue_key("The Venue") == "venue"


def test_venue_research_key_is_shared_across_descriptions() -> None:
    """Two users describing one place two ways must hit one cache entry."""
    a = venue_research_key("The Sarova Stanley Hotel, Nairobi", "Wedding")
    b = venue_research_key("  sarova stanley hotel  nairobi ", "wedding")
    assert a == b
    assert a != venue_research_key("The Sarova Stanley Hotel, Nairobi", "Graduation")


# --------------------------------------------------------------------------
# Node IDs
# --------------------------------------------------------------------------


def test_node_id_is_zero_padded_and_ordered() -> None:
    assert node_id(0) == "s00"
    assert node_id(3) == "s03"
    assert node_id(12) == "s12"
    # Lexical order matches numeric order, which sorting and diffs rely on.
    assert [node_id(i) for i in (2, 10)] == sorted([node_id(2), node_id(10)])


def test_node_id_rejects_out_of_range() -> None:
    with pytest.raises(ValueError):
        node_id(-1)
    with pytest.raises(ValueError):
        node_id(1000)


# --------------------------------------------------------------------------
# Reference analysis key
# --------------------------------------------------------------------------


def test_reference_analysis_key_is_stable_and_content_sensitive() -> None:
    assert reference_analysis_key(b"img") == reference_analysis_key(b"img")
    assert reference_analysis_key(b"img") != reference_analysis_key(b"img ")


# --------------------------------------------------------------------------
# Prompt hashing
# --------------------------------------------------------------------------


def _settings(**overrides: object) -> AgentSettings:
    return AgentSettings(_env_file=None, **overrides)  # type: ignore[arg-type]


def test_prompt_hash_is_deterministic() -> None:
    kwargs = {
        "resolved_prompt": "a wedding at dusk",
        "params": {"steps": 30, "guidance": 7.5},
        "identity_refs": {"p1": [b"face-1"]},
        "settings": _settings(),
    }
    assert prompt_hash(**kwargs) == prompt_hash(**kwargs)


def test_prompt_hash_ignores_dict_key_order_in_params() -> None:
    settings = _settings()
    a = prompt_hash(
        resolved_prompt="p", params={"steps": 30, "guidance": 7.5}, settings=settings
    )
    b = prompt_hash(
        resolved_prompt="p", params={"guidance": 7.5, "steps": 30}, settings=settings
    )
    assert a == b


def test_prompt_hash_changes_with_prompt_text() -> None:
    settings = _settings()
    assert prompt_hash(resolved_prompt="a", settings=settings) != prompt_hash(
        resolved_prompt="b", settings=settings
    )


def test_prompt_hash_changes_with_negative_prompt() -> None:
    settings = _settings()
    assert prompt_hash(
        resolved_prompt="a", negative_prompt="", settings=settings
    ) != prompt_hash(resolved_prompt="a", negative_prompt="blurry", settings=settings)


def test_prompt_hash_changes_with_plan_node() -> None:
    settings = _settings()
    assert prompt_hash(
        resolved_prompt="a", plan_node_id="s01", settings=settings
    ) != prompt_hash(resolved_prompt="a", plan_node_id="s02", settings=settings)


def test_prompt_hash_changes_when_prompt_version_bumps() -> None:
    """Bumping prompt_version must invalidate every cached generation."""
    base = prompt_hash(resolved_prompt="a", settings=_settings(prompt_version="v1"))
    bumped = prompt_hash(resolved_prompt="a", settings=_settings(prompt_version="v2"))
    assert base != bumped


def test_prompt_hash_changes_when_model_changes() -> None:
    base = prompt_hash(resolved_prompt="a", settings=_settings(image_model="model-a"))
    other = prompt_hash(resolved_prompt="a", settings=_settings(image_model="model-b"))
    assert base != other


def test_prompt_hash_ignores_credentials_and_budgets() -> None:
    """A key rotation must not re-bill every photo."""
    base = prompt_hash(
        resolved_prompt="a",
        settings=_settings(image_api_key="key-one", max_shots=8),
    )
    rotated = prompt_hash(
        resolved_prompt="a",
        settings=_settings(image_api_key="key-two", max_shots=16, media_dir="./other"),
    )
    assert base == rotated


def test_prompt_hash_folds_identity_refs_by_content() -> None:
    settings = _settings()
    one = prompt_hash(resolved_prompt="a", identity_refs={"p1": [b"face-1"]}, settings=settings)
    same = prompt_hash(resolved_prompt="a", identity_refs={"p1": [b"face-1"]}, settings=settings)
    different = prompt_hash(
        resolved_prompt="a", identity_refs={"p1": [b"face-2"]}, settings=settings
    )
    assert one == same
    assert one != different


def test_prompt_hash_sensitive_to_identity_ref_order_and_person_key() -> None:
    settings = _settings()
    ordered = prompt_hash(
        resolved_prompt="a", identity_refs={"p1": [b"x", b"y"]}, settings=settings
    )
    reordered = prompt_hash(
        resolved_prompt="a", identity_refs={"p1": [b"y", b"x"]}, settings=settings
    )
    other_person = prompt_hash(
        resolved_prompt="a", identity_refs={"p2": [b"x", b"y"]}, settings=settings
    )
    assert ordered != reordered
    assert ordered != other_person


def test_prompt_hash_tolerates_missing_optional_inputs() -> None:
    settings = _settings()
    assert prompt_hash(resolved_prompt="a", settings=settings) == prompt_hash(
        resolved_prompt="a",
        params=None,
        identity_refs=None,
        negative_prompt="",
        plan_node_id="",
        settings=settings,
    )


# --------------------------------------------------------------------------
# Cross-process determinism (the property the caches depend on)
# --------------------------------------------------------------------------


def _run_hash_in_fresh_process(code: str) -> str:
    result = subprocess.run(
        [sys.executable, "-c", code],
        cwd=BACKEND_DIR,
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert result.returncode == 0, result.stderr
    return result.stdout.strip()


_HASH_SNIPPET = """
from pigamagic_agent.agent.config import AgentSettings
from pigamagic_agent.agent.hashing import prompt_hash, normalize_venue_key, stable_digest
s = AgentSettings(_env_file=None)
print(prompt_hash(
    resolved_prompt="a wedding at dusk",
    params={"steps": 30, "guidance": 7.5, "ratio": "4:3"},
    identity_refs={"p1": [b"face-1"]},
    plan_node_id="s03",
    settings=s,
))
"""


def test_prompt_hash_is_identical_across_processes() -> None:
    """Guards against PYTHONHASHSEED or dict-ordering drift between workers."""
    in_process = prompt_hash(
        resolved_prompt="a wedding at dusk",
        params={"steps": 30, "guidance": 7.5, "ratio": "4:3"},
        identity_refs={"p1": [b"face-1"]},
        plan_node_id="s03",
        settings=AgentSettings(_env_file=None),
    )
    from_subprocess = _run_hash_in_fresh_process(_HASH_SNIPPET)
    assert in_process == from_subprocess


def test_prompt_hash_is_identical_across_processes_with_different_hash_seeds() -> None:
    """Explicitly vary PYTHONHASHSEED: set iteration and str hashing must not leak in."""
    outputs = set()
    for seed in ("0", "1", "4242"):
        result = subprocess.run(
            [sys.executable, "-c", _HASH_SNIPPET],
            cwd=BACKEND_DIR,
            capture_output=True,
            text=True,
            timeout=120,
            env={**__import__("os").environ, "PYTHONHASHSEED": seed},
        )
        assert result.returncode == 0, result.stderr
        outputs.add(result.stdout.strip())
    assert len(outputs) == 1, f"hash varied with PYTHONHASHSEED: {outputs}"


def test_venue_key_is_identical_across_processes() -> None:
    in_process = normalize_venue_key("The Sarova Stanley Hotel, Nairobi!", "Wedding")
    from_subprocess = _run_hash_in_fresh_process(
        "from pigamagic_agent.agent.hashing import normalize_venue_key\n"
        "print(normalize_venue_key('The Sarova Stanley Hotel, Nairobi!', 'Wedding'))"
    )
    assert in_process == from_subprocess
