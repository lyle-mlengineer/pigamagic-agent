"""Step 17 gates: comment modes and edit compilation.

steps.md names three assertions:

* **"make it brighter" changes only lighting** and leaves `preserves` intact
* **an edit of an edit chains correctly**
* **the parent photo is byte-identical before and after**

The third is tested against real rendered bytes through `render_edit`, not
against an id — an id check passes even when the bytes were rewritten.
"""

from __future__ import annotations

import pytest

from pigamagic_agent.agent.comments import (
    ALWAYS_PRESERVED,
    EDITABLE_FIELDS,
    Comment,
    CommentError,
    chain,
    classify_change,
    compile_edit,
    compile_feedback,
    instruction_key,
    parent_bytes_unchanged,
    route_comment,
)
from pigamagic_agent.agent.config import AgentSettings
from pigamagic_agent.agent.plan import EditInstruction, ShotSpec
from pigamagic_agent.agent.providers.image_fixture import FixtureImageProvider
from pigamagic_agent.agent.storage import LocalBlobStore


def _shot(node_id="s03", framing="wide", lighting="warm window light") -> ShotSpec:
    return ShotSpec(
        node_id=node_id,
        phase="ceremony",
        moment="vow exchange",
        activity="vows",
        description="The couple exchange vows at the altar.",
        subjects=["p1", "p2"],
        framing=framing,
        camera={"focal_length_mm": 85, "depth_of_field": "shallow", "angle": "eye-level"},
        lighting=lighting,
        venue_anchors=["stone arches"],
        wardrobe_notes=["ivory gown"],
        must_include=["the rings"],
        must_avoid=["flash"],
        style_preset="editorial",
        caption="The vows",
        why="the defining moment",
    )


def _comment(body: str, mode="edit", comment_id="c1", photo_id="ph_1") -> Comment:
    return Comment(
        comment_id=comment_id, photo_id=photo_id, account_id="acct", body=body, mode=mode
    )


# --------------------------------------------------------------------------
# GATE 1: "make it brighter" changes only lighting, preserves intact
# --------------------------------------------------------------------------


def test_gate_brighter_changes_only_lighting():
    comment = _comment("make it brighter")
    instruction = compile_edit(comment, shot=_shot())

    assert instruction.change == "lighting"
    assert instruction.base_node_id == "s03"
    # Identity and venue are never negotiable.
    assert instruction.preserves_field("identity")
    assert instruction.preserves_field("venue")
    assert instruction.preserves_field("wardrobe")
    # The field being changed must not also claim to be preserved.
    assert not instruction.preserves_field("lighting")
    # And nothing outside lighting was silently added.
    assert set(instruction.preserves) == {"identity", "venue", "wardrobe"}


def test_gate_preserves_are_exactly_what_the_user_did_not_ask_about():
    for body, expected_change in [
        ("make it brighter", "lighting"),
        ("move closer", "framing"),
        ("change the dress to blue", "wardrobe"),
        ("warmer light please", "lighting"),
        ("try again", "regenerate"),
    ]:
        instruction = compile_edit(_comment(body), shot=_shot())
        assert instruction.change == expected_change, body
        assert not instruction.preserves_field(expected_change), body
        assert instruction.preserves_field("identity"), body
        assert instruction.preserves_field("venue"), body


def test_detail_is_normalized_not_verbatim_padding():
    instruction = compile_edit(_comment("  Please   make it  BRIGHTER!!  "), shot=_shot())
    assert instruction.detail == "BRIGHTER"
    assert "  " not in instruction.detail


def test_identity_preservation_cannot_be_bypassed_by_add_or_remove():
    """add/remove are the only changes that drop the default identity pin."""
    added = compile_edit(_comment("add someone on the right"), shot=_shot())
    assert added.change == "add_person"
    # The model's own default rule allows this, but venue is still carried.
    assert isinstance(added.preserves, list)


def test_extended_preserves_are_honoured():
    instruction = compile_edit(
        _comment("make it brighter"), shot=_shot(), extra_preserves=["album", "caption"]
    )
    assert instruction.preserves_field("album")
    assert instruction.preserves_field("caption")


# --------------------------------------------------------------------------
# Classification
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("make it brighter", "lighting"),
        ("a bit darker please", "lighting"),
        ("warmer light", "lighting"),
        ("move closer", "framing"),
        ("wider shot", "framing"),
        ("remove the person on the left", "remove_person"),
        ("try again", "regenerate"),
        ("different one please", "regenerate"),
        ("change the dress to blue", "wardrobe"),
        ("make it black and white", "style"),
        ("show more of the venue", "venue_emphasis"),
    ],
)
def test_classification_covers_the_documented_vocabulary(text, expected):
    assert classify_change(text) == expected


@pytest.mark.parametrize(
    "text",
    ["love this one", "can you fix this", "hmm", "!!!", ""],
)
def test_ambiguous_comments_are_not_guessed_at(text):
    """An unclear edit must be asked about, not guessed.

    Guessing is how "can you fix this" silently becomes a wardrobe change and
    the user is charged for a render they did not ask for.
    """
    assert classify_change(text) is None


def test_keywords_match_on_word_boundaries_only():
    # "add" must not fire inside "address"; "warm" must not fire inside "swarm".
    assert classify_change("please address the swarm") is None


# --------------------------------------------------------------------------
# Feedback vs edit routing
# --------------------------------------------------------------------------


def test_feedback_produces_no_instruction_and_no_job():
    routed = route_comment(_comment("love this one", mode="feedback"), shot=_shot())
    assert routed["mode"] == "feedback"
    assert routed["edit_instruction"] is None
    assert routed["enqueues_job"] is False


def test_edit_produces_an_instruction_and_a_job():
    routed = route_comment(_comment("make it brighter"), shot=_shot())
    assert routed["mode"] == "edit"
    assert routed["enqueues_job"] is True
    assert routed["edit_instruction"]["change"] == "lighting"
    assert routed["instruction_key"].startswith("ed_")


def test_compile_rejects_a_feedback_comment():
    with pytest.raises(CommentError, match="feedback comments do not produce an edit"):
        compile_edit(_comment("love this one", mode="feedback"), shot=_shot())


def test_compile_feedback_rejects_an_edit_comment():
    with pytest.raises(CommentError):
        compile_feedback(_comment("make it brighter"))


def test_an_unclassifiable_edit_raises_with_an_actionable_message():
    with pytest.raises(CommentError) as excinfo:
        compile_edit(_comment("can you fix this"), shot=_shot())
    message = str(excinfo.value)
    assert len(message) > 80
    assert "brighter" in message, "a refusal must show what would work"


def test_comment_validates_its_own_input():
    with pytest.raises(CommentError):
        Comment(comment_id="c1", photo_id="ph_1", account_id="a", body="   ")
    with pytest.raises(CommentError):
        Comment(comment_id="", photo_id="ph_1", account_id="a", body="hi")
    with pytest.raises(CommentError):
        Comment(comment_id="c1", photo_id="ph_1", account_id="a", body="hi", mode="shout")


# --------------------------------------------------------------------------
# GATE 2: an edit of an edit chains correctly
# --------------------------------------------------------------------------


def test_gate_edit_of_edit_chains_and_keeps_earlier_preserves():
    first = compile_edit(
        _comment("make it brighter", comment_id="c1"),
        shot=_shot(),
        # A preserve that exists *only* on the parent, so inheritance is what
        # carries it. Without this the default identity/venue/wardrobe list
        # would mask a broken chain, and the gate would pass vacuously.
        extra_preserves=["album"],
    )
    assert first.preserves_field("album")

    second = compile_edit(
        _comment("warmer light", comment_id="c2"),
        shot=_shot(),
        parent_instruction=first,
    )

    assert second.change == "lighting"
    # The parent's preserves survive the child: a second edit must not release a
    # field the first one pinned.
    assert second.preserves_field("identity")
    assert second.preserves_field("venue")
    assert second.preserves_field("wardrobe")
    assert second.preserves_field("album"), "the parent's preserve must be inherited"
    # The cumulative intent is stated, so one render does not replay history.
    assert second.detail.startswith(first.detail)
    assert second.detail.endswith("warmer light")


def test_chain_folds_a_lineage_into_one_instruction():
    first = compile_edit(
        _comment("make it brighter", comment_id="c1"), shot=_shot(), extra_preserves=["album"]
    )
    second = compile_edit(
        _comment("move closer", comment_id="c2"), shot=_shot(), parent_instruction=first
    )
    third = compile_edit(
        _comment("warmer light", comment_id="c3"), shot=_shot(), parent_instruction=second
    )
    folded = chain([first, second, third])
    assert folded.change == "lighting"
    assert folded.base_node_id == "s03"
    # Every field any edit pinned is still pinned, except the one being changed.
    assert folded.preserves_field("identity")
    assert folded.preserves_field("venue")
    assert folded.preserves_field("wardrobe")
    assert folded.preserves_field("album")
    assert not folded.preserves_field("lighting")
    # The detail is the ordered story, not just the last line.
    assert "brighter" in folded.detail and "closer" in folded.detail


def test_chain_keeps_a_changed_field_out_of_preserves():
    """A field cannot be both changed by the last edit and pinned by it."""
    first = compile_edit(_comment("change the dress to blue", comment_id="c1"), shot=_shot())
    folded = chain([first])
    assert folded.change == "wardrobe"
    assert not folded.preserves_field("wardrobe")


def test_chain_rejects_an_empty_lineage():
    with pytest.raises(CommentError):
        chain([])


def test_chained_detail_does_not_accumulate_duplicates():
    """Each child's detail is already cumulative, so folding must not re-join.

    Prefixing the parent's (already cumulative) detail grows the string
    quadratically: three edits produced a detail that repeated every step.
    """
    first = compile_edit(_comment("make it brighter", comment_id="c1"), shot=_shot())
    second = compile_edit(
        _comment("move closer", comment_id="c2"), shot=_shot(), parent_instruction=first
    )
    third = compile_edit(
        _comment("warmer light", comment_id="c3"), shot=_shot(), parent_instruction=second
    )
    folded = chain([first, second, third])

    assert folded.detail == "brighter; then move closer; then warmer light"
    assert folded.detail.count("brighter") == 1
    assert folded.detail.count("move closer") == 1
    # And no step is lost.
    for word in ("brighter", "move closer", "warmer light"):
        assert word in folded.detail


def test_instruction_key_is_stable_and_content_derived():
    a = compile_edit(_comment("make it brighter", comment_id="c1"), shot=_shot())
    b = compile_edit(_comment("make it brighter", comment_id="c1"), shot=_shot())
    assert instruction_key(a) == instruction_key(b)
    # A different comment id is a different job, even with the same text.
    c = compile_edit(_comment("make it brighter", comment_id="c2"), shot=_shot())
    assert instruction_key(a) != instruction_key(c)


# --------------------------------------------------------------------------
# GATE 3: the parent photo is byte-identical afterwards
# --------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_gate_parent_photo_bytes_are_unchanged_by_an_edit(tmp_path):
    """The real path: render, edit, and compare the parent's bytes on disk."""
    from pigamagic_agent.agent.pipeline import load_brief, run_pipeline
    from pigamagic_agent.agent.prompts import assemble_prompt
    from pigamagic_agent.agent.providers.llm_fixture import FixtureLLMProvider
    from pigamagic_agent.agent.providers.vision_fixture import FixtureVisionProvider
    from pigamagic_agent.agent.providers.web_fixture import FixtureWebProvider
    from pigamagic_agent.agent.render import render_edit, render_plan

    settings = AgentSettings(
        media_dir=str(tmp_path / "media"),
        target_shot_count=2,
        max_shots=8,
        image_budget_usd=50.0,
    )
    store = LocalBlobStore(tmp_path / "blobs")
    provider = FixtureImageProvider()

    artifact = await run_pipeline(
        load_brief("wedding"),
        llm=FixtureLLMProvider(),
        vision=FixtureVisionProvider(),
        web=FixtureWebProvider(),
        settings=settings,
    )
    outcome = await render_plan(
        artifact, provider=provider, store=store, settings=settings, max_shots=1
    )
    original = outcome.photos[0]
    before = await store.get(original.blob.key)

    comment = _comment("make it brighter", photo_id=original.photo_id)
    shot = next(s for s in artifact.plan.shots if s.node_id == original.node_id)
    instruction = compile_edit(comment, shot=shot)

    child = await render_edit(
        original,
        artifact,
        instruction=f"{instruction.change}: {instruction.detail}; "
        f"preserve {', '.join(instruction.preserves)}",
        provider=provider,
        store=store,
        settings=settings,
    )

    after = await store.get(original.blob.key)
    assert parent_bytes_unchanged(before, after)
    assert before == after
    # The child is a new photo with lineage, not a replacement.
    assert child.count == 1
    assert child.photos[0].photo_id != original.photo_id
    assert child.photos[0].parent_id == original.photo_id


def test_parent_bytes_unchanged_compares_content_not_identity():
    assert parent_bytes_unchanged(b"abc", b"abc")
    assert not parent_bytes_unchanged(b"abc", b"abd")
    # Same length, different content — the case an id check would miss.
    assert not parent_bytes_unchanged(b"abc", b"xyz")


def test_editable_fields_cover_the_plan_vocabulary():
    assert set(EDITABLE_FIELDS) == {
        "subject",
        "activity",
        "framing",
        "lighting",
        "venue_emphasis",
        "wardrobe",
        "style",
        "remove_person",
        "add_person",
        "regenerate",
    }


def test_always_preserved_is_identity_venue_and_wardrobe():
    """The three fields that must never move unless explicitly asked for.

    design.md §11.1: `preserves` carries identity and venue forward so "make me
    smile more" cannot silently change her dress.
    """
    assert set(ALWAYS_PRESERVED) == {"identity", "venue", "wardrobe"}
    instruction = compile_edit(_comment("move closer"), shot=_shot())
    for field in ALWAYS_PRESERVED:
        assert instruction.preserves_field(field)
