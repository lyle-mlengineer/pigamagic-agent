"""Comments as edit instructions (design.md §11.1, steps.md Step 17).

A comment on a photo is *simultaneously* social feedback and a possible
generation instruction. The two are distinguished **explicitly** at the API
boundary, never inferred:

- `mode="feedback"` — stored, shown on the card, no job. "Love this one" must
  never spend a credit.
- `mode="edit"` — compiled into an `EditInstruction` and enqueued as a child job.

Why the compiler is deterministic and not an LLM call: `preserves` is a *safety*
list. If a model generates the list of fields held constant, a hallucination
drops "wardrobe" and the next render silently restyles the dress — the exact
trust violation design.md §11.1 is written to prevent. So the change *kind* is
classified by keywords, and every field the user did not ask about is carried
forward. The LLM (when configured) may only propose a `detail` phrasing, never
the preserve set.

The three properties the gate checks:

1. **"make it brighter" changes only lighting** and leaves `preserves` intact.
2. **An edit of an edit chains correctly** — the child's instruction composes
   with its parent's, and a later edit cannot drop an earlier preserve.
3. **The parent photo is byte-identical afterwards** — asserted against the
   stored bytes, not against an id.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Iterable, Literal, Sequence

from .hashing import sha256_bytes, stable_digest
from .plan import EditChange, EditInstruction, ShotSpec

COMMENTS_VERSION = "comments-v1"

CommentMode = Literal["feedback", "edit"]

#: Fields an edit may change. Anything outside this set is not editable by
#: comment — there is no "change the person's face" path here, by design.
EDITABLE_FIELDS: tuple[str, ...] = (
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
)

#: Fields every non-destructive edit holds constant. Identity and venue are not
#: negotiable: an edit must never produce a stranger in a different room. Wardrobe
#: is included for the same reason — design.md §11.1 states it explicitly: "make me
#: smile more" must not silently change her dress.
ALWAYS_PRESERVED: tuple[str, ...] = ("identity", "venue", "wardrobe")


class CommentError(ValueError):
    """A comment that cannot be compiled into a safe edit."""


# --------------------------------------------------------------------------
# The comment
# --------------------------------------------------------------------------


@dataclass(slots=True)
class Comment:
    """One comment on one photo."""

    comment_id: str
    photo_id: str
    account_id: str
    body: str
    mode: CommentMode = "feedback"
    reference_media_ids: list[str] = field(default_factory=list)
    parent_comment_id: str = ""
    created_at: str = ""
    comment_version: str = COMMENTS_VERSION

    def __post_init__(self) -> None:
        self.body = " ".join(str(self.body or "").split())
        if not self.comment_id.strip():
            raise CommentError("a comment needs a comment_id")
        if not self.photo_id.strip():
            raise CommentError("a comment needs a photo_id")
        if not self.body:
            raise CommentError("a comment must not be empty")
        if self.mode not in ("feedback", "edit"):
            raise CommentError(
                f"comment mode must be 'feedback' or 'edit', not {self.mode!r}"
            )

    @property
    def is_edit(self) -> bool:
        return self.mode == "edit"

    def as_dict(self) -> dict[str, Any]:
        return {
            "comment_version": self.comment_version,
            "comment_id": self.comment_id,
            "photo_id": self.photo_id,
            "account_id": self.account_id,
            "body": self.body,
            "mode": self.mode,
            "reference_media_ids": list(self.reference_media_ids),
            "parent_comment_id": self.parent_comment_id,
            "created_at": self.created_at,
        }


# --------------------------------------------------------------------------
# Classification
# --------------------------------------------------------------------------

#: Keyword -> change kind, checked in order. Ordered deliberately: a phrase that
#: names two things ("warmer light", "wider and brighter") resolves to the first
#: match in this list, which is the more specific intent.
_CHANGE_KEYWORDS: tuple[tuple[EditChange, tuple[str, ...]], ...] = (
    ("remove_person", ("remove", "take out", "delete the person", "without the", "crop out")),
    ("add_person", ("add ", "include ", "put someone", "bring in")),
    ("regenerate", ("try again", "another take", "different one", "regenerate", "start over")),
    ("wardrobe", ("dress", "outfit", "suit", "shirt", "clothes", "attire", "wardrobe", "jacket", "gown")),
    ("framing", ("closer", "wider", "crop", "zoom", "framing", "tighter", "angle", "headshot", "full body")),
    ("lighting", ("bright", "brighter", "darker", "warm", "warmer", "cooler", "light", "lighting", "exposure", "shadow", "sunset glow", "golden hour")),
    ("venue_emphasis", ("venue", "background", "room", "hall", "garden", "architecture", "windows", "setting")),
    ("style", ("style", "film", "vintage", "black and white", "monochrome", "moody", "editorial", "documentary", "candid")),
    ("activity", ("doing", "action", "activity", "smile", "smiling", "laughing", "dancing", "pose", "posed")),
    ("subject", ("face", "hair", "look", "expression", "eyes")),
)

#: Order to try keyword groups. Destructive and structural intents first, so
#: "remove the person and make it brighter" reads as a removal, not a lighting
#: change that quietly leaves the person in.
_ORDERED_CHANGES: tuple[EditChange, ...] = (
    "remove_person",
    "add_person",
    "regenerate",
    "wardrobe",
    "framing",
    "lighting",
    "venue_emphasis",
    "style",
    "activity",
    "subject",
)

_STRIP_PREFIXES = (
    "please ",
    "can you ",
    "could you ",
    "i want ",
    "i'd like ",
    "make it ",
    "make ",
    "the ",
    "a bit ",
    "little ",
    "more ",
    "less ",
)

_KEYWORD_RE = re.compile(r"[a-z0-9]+(?:[a-z0-9']*)?")


def _normalize(text: str) -> str:
    """Lowercase, fold punctuation to spaces, collapse whitespace.

    Punctuation is folded rather than deleted so "brighter!" and "brighter"
    classify identically, while word boundaries stay intact for matching.
    """
    lowered = text.casefold()
    lowered = re.sub(r"[^\w\s']", " ", lowered, flags=re.UNICODE)
    return " ".join(lowered.split())


def classify_change(text: str) -> EditChange | None:
    """Return the edit kind implied by a comment, or `None` if unclear.

    `None` matters: an ambiguous edit must be *asked about*, not guessed at.
    Silently picking a kind is how "can you fix this" becomes a wardrobe change.
    """
    normalized = _normalize(text)
    if not normalized:
        return None
    for change in _ORDERED_CHANGES:
        for keyword in dict(_CHANGE_KEYWORDS)[change]:
            # Word-boundary match: "warm" must not fire inside "warmthless"
            # nonsense, and crucially "add" must not fire inside "address".
            pattern = r"\b" + re.escape(keyword.strip()) + r"\b"
            if re.search(pattern, normalized):
                return change
    return None


def _clean_detail(text: str) -> str:
    """The user's intent, normalized and stripped of conversational padding."""
    detail = " ".join(text.split())
    lowered = detail.casefold()
    for prefix in _STRIP_PREFIXES:
        if lowered.startswith(prefix):
            detail = detail[len(prefix):]
            lowered = detail.casefold()
    detail = detail.strip(" .!?,")
    return detail or " ".join(text.split())


# --------------------------------------------------------------------------
# Compiling
# --------------------------------------------------------------------------


def compile_edit(
    comment: Comment,
    *,
    shot: ShotSpec,
    parent_instruction: EditInstruction | None = None,
    extra_preserves: Sequence[str] = (),
) -> EditInstruction:
    """Compile one comment into an `EditInstruction` delta against `shot`.

    `parent_instruction` is how an edit of an edit chains: the parent's preserve
    set is carried into the child, so a second edit cannot quietly release a
    field the first one pinned. Without this, "make it brighter" then "make it
    warmer" can lose the wardrobe guarantee from the first edit.
    """
    if not comment.is_edit:
        raise CommentError(
            "feedback comments do not produce an edit; pass mode='edit' to generate one"
        )

    change = classify_change(comment.body)
    if change is None:
        raise CommentError(
            "cannot tell what to change from this comment. Name one thing — for "
            "example 'make it brighter', 'move closer', 'warmer light' or "
            "'remove the person on the left'."
        )

    if change not in EDITABLE_FIELDS:
        raise CommentError(f"{change!r} is not an editable field")

    preserves: list[str] = []
    for item in (
        *ALWAYS_PRESERVED,
        *(parent_instruction.preserves if parent_instruction else ()),
        *extra_preserves,
    ):
        if item and item not in preserves:
            preserves.append(item)

    # The changed field must NOT be in `preserves`: a delta that both changes
    # and preserves a field is contradictory, and readers would have to guess
    # which wins.
    preserves = [item for item in preserves if item.casefold() != change]

    detail = _clean_detail(comment.body)
    if parent_instruction is not None and parent_instruction.detail:
        # The chained instruction states the cumulative intent, so a single
        # render call does not have to replay the edit history itself. The
        # parent's detail is already cumulative, so append rather than rebuild —
        # prefixing each time would grow the string quadratically.
        detail = f"{parent_instruction.detail}; then {detail}"

    return EditInstruction(
        base_node_id=shot.node_id,
        change=change,
        detail=detail,
        preserves=preserves,
        reference_media_ids=list(comment.reference_media_ids),
        new_seed=True,
        comment_id=comment.comment_id,
    )


def compile_feedback(comment: Comment) -> dict[str, Any]:
    """What a feedback comment produces: a card comment, and nothing else.

    Returned as a dict so the caller can store it verbatim; the point is that
    no `EditInstruction` exists and therefore no job can be enqueued.
    """
    if comment.is_edit:
        raise CommentError("an edit comment is not feedback")
    return {
        "comment_id": comment.comment_id,
        "photo_id": comment.photo_id,
        "account_id": comment.account_id,
        "body": comment.body,
        "mode": "feedback",
        "edit_instruction": None,
        "enqueues_job": False,
        "comments_version": COMMENTS_VERSION,
    }


def route_comment(
    comment: Comment,
    *,
    shot: ShotSpec,
    parent_instruction: EditInstruction | None = None,
    extra_preserves: Sequence[str] = (),
) -> dict[str, Any]:
    """The single entry point: decide feedback vs edit, and compile if needed.

    Returning both shapes from one function keeps the decision in one place, so
    a route cannot accidentally treat a feedback comment as an edit (and charge
    for it) by forgetting a branch.
    """
    if not comment.is_edit:
        return compile_feedback(comment)

    instruction = compile_edit(
        comment,
        shot=shot,
        parent_instruction=parent_instruction,
        extra_preserves=extra_preserves,
    )
    return {
        "comment_id": comment.comment_id,
        "photo_id": comment.photo_id,
        "account_id": comment.account_id,
        "body": comment.body,
        "mode": "edit",
        "edit_instruction": instruction.model_dump(),
        "preserves": list(instruction.preserves),
        "enqueues_job": True,
        "instruction_key": instruction_key(instruction),
        "comments_version": COMMENTS_VERSION,
    }


def instruction_key(instruction: EditInstruction) -> str:
    """Stable key for an instruction, so an identical retry is one job.

    Content-derived, not time-derived: re-posting the same edit after a network
    error must reconcile with the first job instead of charging twice.
    """
    return "ed_" + stable_digest(
        COMMENTS_VERSION,
        instruction.base_node_id,
        instruction.change,
        instruction.detail,
        sorted(instruction.preserves),
        instruction.comment_id,
    )[:20]


def chain(
    instructions: Iterable[EditInstruction],
) -> EditInstruction:
    """Fold a list of instructions into the cumulative one they represent.

    Used by lineage displays and by retries: the last instruction already carries
    every earlier `preserves` (because `compile_edit` inherits them), so the fold
    only needs to union the preserve sets and keep the final detail.
    """
    ordered = list(instructions)
    if not ordered:
        raise CommentError("cannot chain an empty instruction list")

    preserves: list[str] = []
    for instruction in ordered:
        for item in instruction.preserves:
            if item not in preserves:
                preserves.append(item)

    last = ordered[-1]
    return last.model_copy(
        update={
            "preserves": [
                item for item in preserves if item.casefold() != last.change.casefold()
            ],
            # The final instruction's detail is already cumulative when it was
            # compiled from its parent. Joining every step again would duplicate
            # the history, so prefer the last cumulative detail and only fold
            # when the lineage was compiled without a parent.
            "detail": (
                last.detail
                if last.detail
                else " then ".join(
                    dict.fromkeys(item.detail for item in ordered if item.detail)
                )
            ),
        }
    )


def parent_bytes_unchanged(before: bytes, after: bytes) -> bool:
    """Whether a delivered photo survived an edit untouched (SR-4).

    Compared by content hash rather than by id, because an id check passes even
    if the bytes under it were rewritten.
    """
    return sha256_bytes(before) == sha256_bytes(after)


__all__ = [
    "ALWAYS_PRESERVED",
    "COMMENTS_VERSION",
    "Comment",
    "CommentError",
    "CommentMode",
    "EDITABLE_FIELDS",
    "chain",
    "classify_change",
    "compile_edit",
    "compile_feedback",
    "instruction_key",
    "parent_bytes_unchanged",
    "route_comment",
]
