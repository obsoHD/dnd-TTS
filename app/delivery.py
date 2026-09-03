"""Delivery: one documented control token, chosen by the DM, applied to the text.

Why this is a pure text function and lives at the API boundary (M3 contract,
REBUILD.md §3): ``worker.plan`` hashes the canonical text into the ``render_id``
and ``render_line`` re-canonicalises the same text, so a delivery applied to the
text *before* a job exists travels inside ``job.text``. The cache key then
separates deliveries by itself, a restart re-renders the identical line, and no
render-path signature changes. Nothing here runs during a render.

Why a spice must be armed first: the reference clip is the voice, and a token
that moves the speaker embedding makes the character sound like someone else --
the failure that cost the previous build. ``scripts/arm_spice.py`` measures a
spice in the Lab (median SIM drop <= 0.02, no take under the strict gate) and
only then writes it into ``voice.yaml:armed_spices``. Anything not measured is
refused here, loudly, instead of being rendered bare behind the DM's back.
"""
from __future__ import annotations

import math
import re
from dataclasses import dataclass

from app import canon
from app.voices import Voice

BARE = "bare"
BARE_LABEL = "normálne"
MAX_ARMED = 3


@dataclass(frozen=True)
class Spice:
    """One armable delivery.

    ``id`` is the preset name as REBUILD.md §3 spells it: the key of the
    ``armed_spices`` block in ``voice.yaml``, the ``--spice`` argument of
    ``scripts/arm_spice.py`` and the value ``POST /api/say`` carries. ``label``
    is what the pill shows; both keep their Slovak diacritics.
    """

    id: str
    token: str
    label: str


SPICES: tuple[Spice, ...] = (
    Spice("vzdych", "<|sfx:sigh|>", "vzdych"),
    Spice("smiech", "<|sfx:laughter|>", "smiech"),
    Spice("pobavený", "<|emotion:amusement|>", "pobavený"),
    Spice("nahnevaný", "<|emotion:anger|>", "nahnevaný"),
    Spice("nadšený", "<|emotion:enthusiasm|>", "nadšený"),
    Spice("výrazne", "<|prosody:expressive_high|>", "výrazne"),
    Spice("krik", "<|style:shouting|>", "krik"),
)

_BY_ID: dict[str, Spice] = {spice.id: spice for spice in SPICES}
_TOKEN = re.compile(r"<\|[^|<>]*\|>")
# The first word and everything after the whitespace that follows it. DOTALL so
# a pasted two-line note still finds its second word.
_AFTER_FIRST_WORD = re.compile(r"(\S+\s+)(\S.*)", re.DOTALL)


class NotArmed(ValueError):
    """The requested delivery is unknown, banned, or not measured for this voice."""


def available(voice: Voice) -> list[dict]:
    """What the UI may show for this voice: bare first, then every spice.

    Unarmed spices stay in the list on purpose (greyed in the improv bar): the
    DM sees the mechanism exists and that the Lab is where it is earned. Banned
    tokens are dropped entirely, because ``canonicalize`` would refuse them and
    a selectable 400 is not an option.
    """
    armed = _armed(voice)
    entries = [{"id": BARE, "label": BARE_LABEL, "token": "", "armed": True, "measured": None}]
    entries += [{"id": spice.id, "label": spice.label, "token": spice.token,
                 "armed": spice.id in armed, "measured": voice.armed_spices.get(spice.id)}
                for spice in SPICES if not _banned(spice, voice)]
    return entries


def resolve(spice_id: str | None, voice: Voice) -> Spice | None:
    """The spice a request names, or ``None`` for bare.

    Raises ``NotArmed`` rather than falling back to bare: a silent fallback
    would hand the table a take that is not the take it asked for.
    """
    if spice_id is None or not spice_id.strip() or spice_id == BARE:
        return None
    spice = _BY_ID.get(spice_id)
    if spice is None:
        raise NotArmed(f"neznáme podanie {spice_id!r}")
    if spice.id not in _armed(voice):
        raise NotArmed(f"podanie {spice.id!r} nie je overené v Labe pre hlas {voice.id!r}")
    return spice


def apply(text: str, spice_id: str | None, voice: Voice) -> str:
    """Place the spice's token after the first word, attached to the second.

    ``Ten <|emotion:anger|>nie.`` -- never leading, because a line that opens on
    a token loses its first syllable. Pure and idempotent: a line that already
    carries a token (typed by the DM, or placed by an earlier call) is returned
    untouched, so ``apply(apply(t)) == apply(t)`` and two tokens never fight.
    """
    spice = resolve(spice_id, voice)
    if spice is None or _TOKEN.search(text):
        return text
    lead = text[: len(text) - len(text.lstrip())]
    match = _AFTER_FIRST_WORD.fullmatch(text.lstrip())
    if match is None:
        # A one-word line has no second word to carry the token, and it is too
        # short for a delivery to survive the gate anyway.
        return text
    first_word, rest = match.groups()
    return f"{lead}{first_word}{spice.token}{rest}"


def _armed(voice: Voice) -> dict[str, dict]:
    """The spices this voice may actually use, keyed by id.

    A hand-edited ``voice.yaml`` must not be able to widen the selector, so the
    cap is enforced on read as well as on write: at most ``MAX_ARMED``, keeping
    the smallest measured SIM drops, which is the order ``arm_spice.py``
    promotes in. Ties keep file order (``sorted`` is stable).
    """
    measured = [(spice, voice.armed_spices[spice.id]) for spice in SPICES
                if spice.id in voice.armed_spices and not _banned(spice, voice)]
    return {spice.id: entry for spice, entry in sorted(measured, key=lambda pair: _drop(pair[1]))[:MAX_ARMED]}


def _drop(entry: object) -> float:
    """The measured SIM drop, or infinity when the entry says nothing usable --
    an unmeasured hand-written entry loses every tie for a slot."""
    value = entry.get("sim_drop") if isinstance(entry, dict) else None
    return float(value) if isinstance(value, (int, float)) else math.inf


def _banned(spice: Spice, voice: Voice) -> bool:
    """WHY canon's own key function: ``available`` must drop exactly what
    ``canonicalize`` would refuse. A second copy of the matching rule here would
    drift, and the UI would offer a spice the render then answers 400 to."""
    keys = {canon._token_key(item) for item in voice.banned_tokens}
    key = canon._token_key(spice.token)
    return key in keys or key.rsplit(":", 1)[-1] in keys
