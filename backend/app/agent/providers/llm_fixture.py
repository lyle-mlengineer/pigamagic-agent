"""Deterministic offline `LLMProvider` for plan compilation (Steps 5-7, 17).

Real providers arrive at Step 8. Until then this fixture must exercise the
compiler end to end — prompt construction, validation, repair, fallback — with
no network and fully deterministically.

It works by building a **plausible plan** from the structured payload embedded
in the prompt. The compiler puts a JSON block in the user message; the fixture
reads it back out and derives shots from the moment list. That keeps the
fixture honest: it exercises the same path a real model would.

Failure modes are selectable so the repair/fallback paths are testable:
    mode="good"      valid plan
    mode="bad_json"  unparseable output, twice
    mode="short"     a plan that omits coverage (forces repair by injection)
    mode="unsafe"    a plan containing a policy-violating node
"""

from __future__ import annotations

import json
import re
from typing import Any, Mapping

from app.agent.hashing import canonical_json
from app.agent.providers.base import CostEstimate, LLMResult, ProviderError

# The compiler embeds its structured payload between these markers so the
# fixture (and only the fixture) can read it back without guessing.
PAYLOAD_START = "<<<BRIEF_JSON"
PAYLOAD_END = "BRIEF_JSON>>>"

_FRAMINGS = ("wide", "medium", "close", "candid", "detail")


def extract_payload(user: str) -> dict[str, Any] | None:
    """Read the structured brief payload out of a compiled prompt."""
    start = user.find(PAYLOAD_START)
    end = user.find(PAYLOAD_END)
    if start == -1 or end == -1 or end <= start:
        return None
    blob = user[start + len(PAYLOAD_START) : end].strip()
    try:
        return json.loads(blob)
    except json.JSONDecodeError:
        return None


class FixtureLLMProvider:
    """Offline, deterministic `LLMProvider`."""

    name = "fixture"

    def __init__(self, mode: str = "good", fail_times: int | None = None) -> None:
        if mode not in {"good", "bad_json", "short", "unsafe"}:
            raise ValueError(f"unknown fixture mode {mode!r}")
        self.mode = mode
        # When set, the first N calls behave as `bad_json` then recover.
        self._remaining_failures = fail_times
        self.calls: list[dict[str, Any]] = []

    async def complete_json(
        self,
        *,
        system: str,
        user: str,
        schema: dict[str, Any],
        model: str | None = None,
        temperature: float = 0.0,
        max_tokens: int | None = None,
    ) -> LLMResult:
        self.calls.append({"system": system, "user": user, "model": model})

        if self._remaining_failures is not None and self._remaining_failures > 0:
            self._remaining_failures -= 1
            raise ProviderError(
                "fixture: simulated planner failure",
                provider=self.name,
                retryable=True,
            )

        if self.mode == "bad_json":
            return LLMResult(
                data={},
                raw_text="not json at all",
                provider=self.name,
                model=model or "fixture-llm",
                cost=CostEstimate.zero(),
            )

        payload = extract_payload(user)
        if payload is None:
            raise ProviderError(
                "fixture: prompt carried no brief payload",
                provider=self.name,
                retryable=False,
            )

        plan = self._plan_for(payload)
        return LLMResult(
            data=plan,
            raw_text=canonical_json(plan),
            provider=self.name,
            model=model or "fixture-llm",
            cost=CostEstimate(credits=0, usd=0.002),
            latency_ms=0,
        )

    async def estimate_cost(self, *, user: str, model: str | None = None) -> CostEstimate:
        return CostEstimate(credits=0, usd=0.002)

    # --- fixture behaviour ------------------------------------------------

    def _plan_for(self, payload: Mapping[str, Any]) -> dict[str, Any]:
        cast: list[dict[str, Any]] = list(payload.get("cast", []))
        moments: list[str] = list(payload.get("moments", []))
        activities: list[str] = list(payload.get("activities", []))
        venue_anchors: list[str] = list(payload.get("venue_anchors", []))
        style_preset: str = str(payload.get("style_preset", "editorial"))
        count: int = int(payload.get("shot_count", 8))

        person_keys = [person["person_key"] for person in cast] or ["p1"]

        shots: list[dict[str, Any]] = []
        # Per-moment set of (framing, subjects) already used, so the fixture never
        # proposes two shots the validator would consider identical.
        used_signatures: dict[str, set[tuple[str, tuple[str, ...]]]] = {}
        for index in range(count):
            activity = activities[index % len(activities)] if activities else (moments[0] if moments else "the main event")
            # Prefer a moment that actually matches the activity, so the fixture
            # exercises the same activity/moment alignment a real model should
            # produce. Falling back to round-robin keeps every moment reachable.
            #
            # `cycle` is how many times we have wrapped around the activity list.
            # Without it, any activity that matches a canonical moment pins that
            # exact moment on every cycle, so wrap-around would regenerate an
            # identical (moment, framing) pair and the dedupe stage would silently
            # drop real shots. Advancing through the matching moments instead keeps
            # every repetition a distinct frame.
            cycle = index // len(activities) if activities else 0
            matched = _moments_for(activity, moments)
            if matched:
                moment = matched[cycle % len(matched)]
            else:
                moment = moments[index % len(moments)] if moments else "the main event"
            # Framing is offset by the cycle so it does not move in lockstep with
            # the activity index. With equal-length activity and framing cycles
            # (5 and 5), `index % 5` advances identically on both and a repeat is
            # byte-identical to its original, which the dedupe stage then drops.
            #
            # Offset alone is not enough: two different moments shot at the same
            # framing with the same cast are also the same photograph. So walk the
            # framings and take the first (framing, subjects) pair this moment has
            # not used yet. The fixture must propose `count` genuinely distinct
            # shots — a duplicate here would be silently dropped by the validator
            # and a real coverage bug would hide behind a fixture artefact.
            used = used_signatures.setdefault(moment.casefold(), set())
            framing, subjects = self._distinct_frame(
                index, person_keys, used, preferred=_FRAMINGS[(index + cycle) % len(_FRAMINGS)]
            )
            used.add((framing, tuple(sorted(subjects))))
            shots.append(
                {
                    "node_id": f"s{index:02d}",
                    "phase": self._phase_for(index, payload),
                    "moment": moment,
                    "activity": activity,
                    "description": f"{activity} at {moment}.",
                    "subjects": subjects,
                    "framing": framing,
                    "camera": {
                        "focal_length_mm": 85 if framing == "close" else 35,
                        "angle": "eye-level",
                        "depth_of_field": "shallow" if framing == "close" else "medium",
                        "aperture_f": 1.8 if framing == "close" else 2.8,
                    },
                    "lighting": str(payload.get("lighting", "") or ""),
                    "venue_anchors": venue_anchors[:2],
                    "wardrobe_notes": [],
                    "must_include": [],
                    "must_avoid": [],
                    "style_preset": style_preset,
                    "caption": moment.title(),
                    "why": f"Covers the {moment} moment.",
                }
            )

        if self.mode == "short":
            # Drop most shots so coverage rules must inject replacements.
            shots = shots[:1]

        if self.mode == "unsafe":
            # Prepend (not append) a genuinely policy-violating shot, so it is in
            # the candidate set before any trimming and must be dropped by the
            # validator rather than silently trimmed away by position.
            shots.insert(
                0,
                {
                    "node_id": "s99",
                    "phase": "afterwards",
                    "moment": "behind the scenes",
                    "activity": "a nude photoshoot by the pool",
                    "description": "An unrequested scene.",
                    "subjects": person_keys,
                    "framing": "wide",
                    "camera": {"focal_length_mm": 35, "angle": "eye-level", "depth_of_field": "medium", "aperture_f": 2.8},
                    "lighting": "",
                    "venue_anchors": [],
                    "wardrobe_notes": [],
                    "must_include": [],
                    "must_avoid": [],
                    "style_preset": style_preset,
                    "caption": "Unrequested",
                    "why": "Not supported by the brief.",
                },
            )

        return {
            "shots": shots,
            "warnings": [],
            "confidence": 0.8 if self.mode == "good" else 0.5,
        }


    def _subjects_for(self, index: int, person_keys: list[str], framing: str) -> list[str]:
        """Vary who appears in each shot, the way a real plan does.

        A plan where every frame contains everyone is photographically dull and,
        more importantly, would leave subject-variation bugs untested. Detail and
        candid frames favour one or two people; wide frames take the group.
        """
        if not person_keys:
            return []
        if len(person_keys) == 1:
            return list(person_keys)
        if framing in {"detail", "close"}:
            return [person_keys[index % len(person_keys)]]
        if framing == "candid":
            second = person_keys[(index + 1) % len(person_keys)]
            first = person_keys[index % len(person_keys)]
            return [first] if second == first else [first, second]
        return list(person_keys)

    def _distinct_frame(
        self,
        index: int,
        person_keys: list[str],
        used: set[tuple[str, tuple[str, ...]]],
        *,
        preferred: str,
    ) -> tuple[str, list[str]]:
        """Pick a (framing, subjects) pair not already used for this moment.

        Tries the preferred framing first, then the rest, and finally shifts the
        subject rotation. Only a moment photographed more times than there are
        distinguishable frames can collide, and then repeating is the honest
        outcome rather than inventing a shot.
        """
        order = [preferred, *(item for item in _FRAMINGS if item != preferred)]
        for offset in range(len(person_keys) + 1):
            for framing in order:
                subjects = self._subjects_for(index + offset, person_keys, framing)
                if (framing, tuple(sorted(subjects))) not in used:
                    return framing, subjects
        # Exhausted every combination: repeat the preferred frame.
        return preferred, self._subjects_for(index, person_keys, preferred)

    def _phase_for(self, index: int, payload: Mapping[str, Any]) -> str:
        phases: list[str] = list(payload.get("phases", [])) or ["main"]
        return phases[min(index, len(phases) - 1)]


class ScriptedLLMProvider:
    """Returns pre-built plan payloads in call order (for repair tests)."""

    name = "scripted"

    def __init__(self, payloads: list[dict[str, Any]]) -> None:
        self._payloads = list(payloads)
        self.calls: list[str] = []

    async def complete_json(
        self,
        *,
        system: str,
        user: str,
        schema: dict[str, Any],
        model: str | None = None,
        temperature: float = 0.0,
        max_tokens: int | None = None,
    ) -> LLMResult:
        self.calls.append(user)
        if not self._payloads:
            raise AssertionError("ScriptedLLMProvider ran out of scripted payloads")
        return LLMResult(
            data=self._payloads.pop(0),
            provider=self.name,
            model=model or "scripted",
            cost=CostEstimate.zero(),
        )

    async def estimate_cost(self, *, user: str, model: str | None = None) -> CostEstimate:
        return CostEstimate.zero()


class FailingLLMProvider:
    """Always raises, to exercise the procedural fallback."""

    name = "failing"

    def __init__(self, retryable: bool = True) -> None:
        self.retryable = retryable
        self.calls = 0

    async def complete_json(self, **_: Any) -> LLMResult:
        self.calls += 1
        raise ProviderError(
            "always fails", provider=self.name, retryable=self.retryable
        )

    async def estimate_cost(self, *, user: str, model: str | None = None) -> CostEstimate:
        return CostEstimate.zero()



def _moments_for(activity: str, moments: list[str]) -> list[str]:
    """Every canonical moment that mentions this activity, in pack order."""
    needle = activity.strip().casefold()
    if not needle:
        return []
    found: list[str] = []
    for moment in moments:
        lowered = moment.casefold()
        if needle in lowered or lowered in needle:
            found.append(moment)
    return found


def _moment_for(activity: str, moments: list[str]) -> str | None:
    """Find the first canonical moment that mentions this activity, if any."""
    found = _moments_for(activity, moments)
    return found[0] if found else None


def slugify(text: str) -> str:
    """Lowercase, hyphen-separated. Shared by caption and node helpers."""
    return re.sub(r"[^a-z0-9]+", "-", text.casefold()).strip("-")


__all__ = [
    "FailingLLMProvider",
    "FixtureLLMProvider",
    "PAYLOAD_END",
    "PAYLOAD_START",
    "ScriptedLLMProvider",
    "extract_payload",
    "slugify",
]
