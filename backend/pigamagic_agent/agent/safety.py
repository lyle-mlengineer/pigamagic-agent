"""Pre-generation content policy gate (design.md §12, PRD SR-7).

The gate runs **before** any provider call, so a request we will not fulfil is
refused without spending money on a render we would have to throw away.

Two different things happen here, and conflating them is the classic mistake:

- **Refusal** — we will not generate this at all. Always paired with a specific,
  actionable reason ("photos of children at a pool party need a guardian present
  in the frame") rather than a bare "not allowed".
- **Care flag** — we will generate it, but the subject matter is sensitive
  (memorial, religious ceremony) and the planner should use restraint. A care
  flag must never read as a soft refusal.

Deliberate design choices:

- Matching is on **normalised phrases**, not substrings, so "underwater" is
  allowed (a real wedding shot) while "nude" is not, and "drunken" does not
  trip on "drunk" inside another word.
- Rules are **data**, not branching code, so they can be read, audited and
  extended without touching logic.
- The gate never inspects or stores biometric data. It reads the brief, the
  plan and the cast labels the user themselves supplied (SR-5).
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Iterable, Literal, Sequence

from pigamagic_agent.agent.compile import ShootBrief
from pigamagic_agent.agent.conventions import SENSITIVE_EVENTS
from pigamagic_agent.agent.plan import ShootPlan

SAFETY_VERSION = "1"

Verdict = Literal["allow", "refuse"]

#: Which refusal to lead with when several fire. Higher wins.
_SEVERITY_ORDER: dict[str, int] = {
    "minor_sexual_context": 100,
    "minor_unsupervised_context": 80,
    "sexual_content": 60,
    "violence": 50,
    "impersonation": 40,
}


@dataclass(frozen=True, slots=True)
class SafetyFinding:
    """One rule fired. `code` is stable; `message` is what a user reads."""

    code: str
    message: str
    severity: Literal["refuse", "care"] = "refuse"
    matched: str = ""

    def as_dict(self) -> dict[str, Any]:
        return {
            "code": self.code,
            "message": self.message,
            "severity": self.severity,
            "matched": self.matched,
        }


@dataclass(frozen=True, slots=True)
class SafetyVerdict:
    """The gate's answer. Frozen: a verdict never changes after it is issued."""

    verdict: Verdict
    findings: tuple[SafetyFinding, ...] = ()
    care_flags: tuple[SafetyFinding, ...] = ()
    safety_version: str = SAFETY_VERSION

    @property
    def allowed(self) -> bool:
        return self.verdict == "allow"

    @property
    def refused(self) -> bool:
        return self.verdict == "refuse"

    @property
    def reason(self) -> str:
        """A single actionable sentence for the most serious refusal.

        Severity order matters: 'intimate content involving a child' must not be
        reported as the milder 'intimate content' just because that rule happened
        to run first.
        """
        refusals = [item for item in self.findings if item.severity == "refuse"]
        if not refusals:
            return ""
        ranked = sorted(refusals, key=lambda item: _SEVERITY_ORDER.get(item.code, 0), reverse=True)
        return ranked[0].message

    @property
    def codes(self) -> tuple[str, ...]:
        return tuple(item.code for item in self.findings if item.severity == "refuse")

    def as_dict(self) -> dict[str, Any]:
        return {
            "verdict": self.verdict,
            "safety_version": self.safety_version,
            "reason": self.reason,
            "codes": list(self.codes),
            "findings": [item.as_dict() for item in self.findings],
            "care_flags": [item.as_dict() for item in self.care_flags],
        }


class SafetyRefusal(Exception):
    """Raised when a run is refused. Carries the verdict for the caller to report."""

    def __init__(self, verdict: SafetyVerdict) -> None:
        self.verdict = verdict
        super().__init__(verdict.reason or "This request was refused by the safety gate.")


# --------------------------------------------------------------------------
# Rules as data
# --------------------------------------------------------------------------

#: Intimate or sexual content. Refused outright: PlusOne photographs events, and
#: none of these are an event photograph.
SEXUAL_MARKERS: tuple[str, ...] = (
    "nude",
    "nudity",
    "naked",
    "topless",
    "bottomless",
    "lingerie",
    "underwear photo",
    "in bed intimate",
    "sexual",
    "erotic",
    "sensual pose",
    "swimwear shoot",
    "bikini shoot",
)

#: Violence and weapons. "Fighting" is refused but a boxing match is an event —
#: the distinction is carried by the phrasing, not the topic.
VIOLENCE_MARKERS: tuple[str, ...] = (
    "gore",
    "blood spill",
    "dismember",
    "beheading",
    "corpse",
    "dead body",
    "graphic injury",
    "weapon",
    "gun",
    "firearm",
    "knife fight",
    "fight scene",
    "violent attack",
)

#: Impersonating a specific real, identifiable person. We do not render a
#: famous stranger's face onto the user's cast.
IMPERSONATION_MARKERS: tuple[str, ...] = (
    "look like",
    "impersonate",
    "as a celebrity",
    "celebrity lookalike",
    "famous person",
    "put me in the body of",
    "face swap",
    "faceswap",
    "deepfake",
)

#: Contexts that are only acceptable with a guardian present. Combined with a
#: `child`/`teen` cast member these refuse; alone they do not.
UNSUPERVISED_MINOR_CONTEXTS: tuple[str, ...] = (
    "pool party",
    "sleepover",
    "beach party",
    "locker room",
    "bathroom",
    "shower",
    "changing room",
)

#: Explicit minimisation requests. Not refused, but flagged for care: the user
#: has told us the event involves grief or restraint.
CARE_MARKERS: tuple[str, ...] = (
    "open casket",
    "coffin",
    "funeral",
    "grave",
    "hospital",
    "wheelchair",
    "illness",
    "seriously ill",
    "hospitalised",
    "hospitalized",
    "crying",
    "grieving",
)


def _normalise(text: str) -> str:
    """Casefold and collapse punctuation/whitespace into single spaces.

    Phrase matching on a normalised string means "nude," and "Nude." behave like
    "nude", while avoiding the substring trap where "underwater" would match a
    rule for "under".
    """
    lowered = text.casefold()
    lowered = re.sub(r"[^a-z0-9\s']", " ", lowered)
    return re.sub(r"\s+", " ", lowered).strip()


def _contains_phrase(haystack: str, phrase: str) -> bool:
    normalised = _normalise(phrase)
    if not normalised:
        return False
    # Word-boundary match so "drunk" does not fire inside "drunkenly" by accident
    # and, more importantly, "gun" does not fire inside "begun".
    return re.search(rf"(?<![a-z0-9]){re.escape(normalised)}(?![a-z0-9])", haystack) is not None


# --------------------------------------------------------------------------
# Gate
# --------------------------------------------------------------------------


@dataclass
class SafetyGate:
    """Evaluates a brief and optionally an already-compiled plan.

    Stateless and cheap, so it can be called at brief time (fail early) and again
    immediately before rendering (defence in depth).
    """

    sexual_markers: tuple[str, ...] = SEXUAL_MARKERS
    violence_markers: tuple[str, ...] = VIOLENCE_MARKERS
    impersonation_markers: tuple[str, ...] = IMPERSONATION_MARKERS
    minor_contexts: tuple[str, ...] = UNSUPERVISED_MINOR_CONTEXTS
    care_markers: tuple[str, ...] = CARE_MARKERS

    def check(
        self,
        brief: ShootBrief,
        plan: ShootPlan | None = None,
        *,
        cast: Sequence[Any] | None = None,
    ) -> SafetyVerdict:
        """Return a verdict for this shoot.

        `cast` accepts anything with an `age_band` (PersonProfile or CastEntry),
        so the gate works before and after profile fusion.
        """
        findings: list[SafetyFinding] = []
        care: list[SafetyFinding] = []

        text = _normalise(
            " ".join(
                [
                    brief.title,
                    brief.event,
                    brief.venue_name,
                    " ".join(brief.activities),
                    brief.story,
                ]
            )
        )
        if plan is not None:
            text = f"{text} {' '.join(_plan_text(plan))}"
            text = _normalise(text)

        findings.extend(self._scan(text, self.sexual_markers, "sexual_content", self._sexual_message))
        findings.extend(self._scan(text, self.violence_markers, "violence", self._violence_message))
        findings.extend(
            self._scan(text, self.impersonation_markers, "impersonation", self._impersonation_message)
        )

        minors = self._minor_kinds(cast)
        if minors:
            findings.extend(self._minor_findings(text, brief, minors))

        care.extend(self._care_findings(text, brief))

        # A refusal dominates: if anything refuses, the verdict refuses.
        verdict: Verdict = "refuse" if findings else "allow"
        return SafetyVerdict(
            verdict=verdict,
            findings=tuple(findings),
            care_flags=tuple(care),
        )

    # -- rule scanners ---------------------------------------------------

    def _scan(
        self,
        text: str,
        markers: Iterable[str],
        code: str,
        message_for: Any,
    ) -> list[SafetyFinding]:
        for marker in markers:
            if _contains_phrase(text, marker):
                return [
                    SafetyFinding(
                        code=code,
                        message=message_for(marker),
                        severity="refuse",
                        matched=marker,
                    )
                ]
        return []

    def _minor_findings(
        self, text: str, brief: ShootBrief, minors: set[str]
    ) -> list[SafetyFinding]:
        """Children are welcome at family events; not in these contexts."""
        found: list[SafetyFinding] = []
        bands = " and ".join(sorted(minors))
        for marker in self.minor_contexts:
            if _contains_phrase(text, marker):
                found.append(
                    SafetyFinding(
                        code="minor_unsupervised_context",
                        severity="refuse",
                        matched=marker,
                        message=(
                            f"This shoot includes a {bands} cast member and is set in a "
                            f"'{marker}' context. PlusOne does not photograph children in "
                            "private or changing contexts. Remove that scene, or make it a "
                            "supervised, fully-clothed group activity — for example a "
                            "backyard party with a guardian in frame."
                        ),
                    )
                )
                break

        # Any sexual marker plus a minor is a hard refusal even if the sexual
        # marker alone would already have refused; the message must name the
        # reason a user can act on.
        if any(_contains_phrase(text, marker) for marker in self.sexual_markers):
            found.append(
                SafetyFinding(
                    code="minor_sexual_context",
                    severity="refuse",
                    matched="sexual content with a minor in cast",
                    message=(
                        f"This shoot combines intimate content with a {bands} cast member. "
                        "PlusOne will not generate this under any circumstances. Remove the "
                        "minor from the shoot or choose a different event."
                    ),
                )
            )
        return found

    def _care_findings(self, text: str, brief: ShootBrief) -> list[SafetyFinding]:
        """Sensitive subject matter we *will* render, with restraint."""
        found: list[SafetyFinding] = []
        event_key = brief.event.strip().casefold()
        if event_key in SENSITIVE_EVENTS:
            found.append(
                SafetyFinding(
                    code="sensitive_event",
                    severity="care",
                    message=(
                        "This is a sensitive event. The planner will favour candid, "
                        "warm framing and avoid dramatising grief."
                    ),
                )
            )
        for marker in self.care_markers:
            if _contains_phrase(text, marker):
                found.append(
                    SafetyFinding(
                        code="care_marker",
                        severity="care",
                        matched=marker,
                        message=(
                            f"The brief mentions '{marker}'. Photos will keep this "
                            "off-centre and avoid graphic depiction."
                        ),
                    )
                )
        return found

    # -- messages --------------------------------------------------------

    @staticmethod
    def _minor_kinds(cast: Sequence[Any] | None) -> set[str]:
        kinds: set[str] = set()
        for member in cast or ():
            band = getattr(member, "age_band", None)
            if band is None and isinstance(member, dict):
                band = member.get("age_band")
            if band in {"child", "teen"}:
                kinds.add(str(band))
        return kinds

    @staticmethod
    def _sexual_message(marker: str) -> str:
        return (
            f"PlusOne photographs social events, and this brief asks for intimate "
            f"content ('{marker}'). Change the scene to a clothed, public moment — "
            "for example the couple's first dance or a group toast — and it can run."
        )

    @staticmethod
    def _violence_message(marker: str) -> str:
        return (
            f"This brief asks for violence or weaponry ('{marker}'), which PlusOne "
            "does not generate. A sport or re-enactment can be photographed without "
            "the prop or the injury — describe the action and the crowd instead."
        )

    @staticmethod
    def _impersonation_message(marker: str) -> str:
        return (
            f"This brief asks to depict a specific other person ('{marker}'). PlusOne "
            "only renders the people whose own reference photos you supplied. Remove "
            "the lookalike request and the shoot can proceed."
        )


def _plan_text(plan: ShootPlan) -> list[str]:
    """Text the gate must also see once a plan exists."""
    parts: list[str] = []
    for shot in plan.shots:
        parts.extend([shot.moment, shot.activity, shot.description, shot.caption])
    return parts


# --------------------------------------------------------------------------
# Convenience API
# --------------------------------------------------------------------------

_DEFAULT_GATE = SafetyGate()


def check_brief(
    brief: ShootBrief,
    plan: ShootPlan | None = None,
    *,
    cast: Sequence[Any] | None = None,
) -> SafetyVerdict:
    return _DEFAULT_GATE.check(brief, plan, cast=cast)


def require_allowed(
    brief: ShootBrief,
    plan: ShootPlan | None = None,
    *,
    cast: Sequence[Any] | None = None,
) -> SafetyVerdict:
    """Like `check_brief`, but raises `SafetyRefusal` instead of returning."""
    verdict = check_brief(brief, plan, cast=cast)
    if verdict.refused:
        raise SafetyRefusal(verdict)
    return verdict


__all__ = [
    "CARE_MARKERS",
    "IMPERSONATION_MARKERS",
    "SAFETY_VERSION",
    "SEXUAL_MARKERS",
    "SafetyFinding",
    "SafetyGate",
    "SafetyRefusal",
    "SafetyVerdict",
    "UNSUPERVISED_MINOR_CONTEXTS",
    "VIOLENCE_MARKERS",
    "check_brief",
    "require_allowed",
]
