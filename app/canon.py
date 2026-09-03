"""Canonical text: the one deterministic form of a line that the TTS receives.

Why this module exists: renders are cached by ``render_id``, which hashes the
canonical text. Two DMs typing the "same" line with different quotes, dashes or
spacing must land on the same cache entry, and everything the model sees must be
reproducible from the typed text alone. Every rule here is pure; the rule set is
versioned through ``CANON_VERSION``, which the recipe hash folds in, so a rule
change invalidates the bank instead of silently mixing old and new renders.

Contract: docs/M1-contracts.md, section ``app/canon.py``.
"""
from __future__ import annotations

import json
import re
import unicodedata
from collections.abc import Iterable
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

CANON_VERSION = "1"
MAX_CHARS = 250
MAX_PAUSES = 3
TOKENS_PATH = Path(__file__).resolve().parent.parent / "configs" / "tokens.json"

PAUSE = "<|prosody:pause|>"
LONG_PAUSE = "<|prosody:long_pause|>"

# The cookbook list inside the TTS image (see the contract). Used when
# configs/tokens.json is absent so canon works before the config is written.
_DOCUMENTED_TOKENS: dict[str, tuple[str, ...]] = {
    "emotion": (
        "elation", "amusement", "enthusiasm", "determination", "pride", "contentment",
        "affection", "relief", "contemplation", "confusion", "surprise", "awe", "longing",
        "arousal", "anger", "fear", "disgust", "bitterness", "sadness", "shame", "helplessness",
    ),
    "style": ("singing", "shouting", "whispering"),
    "sfx": ("cough", "laughter", "crying", "screaming", "burping", "humming", "sigh", "sniff", "sneeze"),
    "prosody": (
        "speed_very_slow", "speed_slow", "speed_fast", "speed_very_fast", "pause", "long_pause",
        "pitch_low", "pitch_high", "expressive_high", "expressive_low",
    ),
}

_TOKEN = re.compile(r"<\|[^|<>]*\|>")
_PAUSE_TOKEN = re.compile(r"<\|prosody:(?:long_)?pause\|>")
_LONG_BEAT = re.compile(r"\s*(?:…|\.{3,})\s*")
# An em dash is never a hyphen, so it may touch the words; a plain hyphen only
# counts as a beat when spaced, so "Slovensko-maďarský" stays one word.
_BEAT = re.compile(r"\s*—\s*|\s+-\s+")
_PUNCT_AFTER_TOKEN = re.compile(r" (<\|[^|<>]*\|>) ([.!?,;:]+)(?=\s|$)")

_VOWELS = "aeiouyáéíóúýäô"
_DRAWL = re.compile(rf"([{_VOWELS}])(\*+)", re.IGNORECASE)
_STRAY_STARS = re.compile(r"\*+")
_WORD = re.compile(r"[a-záéíóúýäôčšžťďňľŕĺ']+")
_DIPHTHONG = re.compile(r"i[aeu]")
_SYLLABIC_RL = re.compile(rf"(?:^|[^{_VOWELS}\W])[rlŕĺ](?=[^{_VOWELS}\W]|$)")

_TERMINAL = ".!?"
_CLOSERS = "\"')»]"
_TYPOGRAPHY = str.maketrans({
    "„": '"', "“": '"', "”": '"', "«": '"', "»": '"',
    "‚": "'", "‘": "'", "’": "'",
    "–": "—", "―": "—", "‒": "—",
})


class TooLong(ValueError):
    """Spoken text exceeds MAX_CHARS; callers split at sentence ends, never join WAVs."""


class BannedToken(ValueError):
    """A typed control token is on the voice's banned list (e.g. shouting on Bag)."""


@dataclass
class Canon:
    text: str
    spoken: str
    syllables: int
    warnings: list[str]


def canonicalize(text: str, lang: str = "sk", banned: Iterable[str] = frozenset()) -> Canon:
    """Turn typed text into the model input and its spoken counterpart.

    ``lang`` is part of the contract for later language-specific rules; v1
    applies the same Slovak-first rules to English lines. ``banned`` accepts any
    iterable because ``voice.yaml`` stores the list as a YAML sequence.
    """
    warnings: list[str] = []
    out = unicodedata.normalize("NFC", text).translate(_TYPOGRAPHY)
    out = _resolve_typed_tokens(out, banned, warnings)
    out = _drawl(out)
    out = _beats(out)
    out = _cap_pauses(out, warnings)
    out = _ensure_terminal(_collapse(out))
    spoken = _spoken(out)
    if len(spoken) > MAX_CHARS:
        raise TooLong(f"{len(spoken)} spoken chars, max {MAX_CHARS}")
    return Canon(text=out, spoken=spoken, syllables=syllables(spoken), warnings=warnings)


def syllables(text: str) -> int:
    """Slovak-aware nucleus count, ported from the legacy orchestrator.

    Vowels are nuclei; ``ia ie iu`` are one nucleus (``ô`` is a single letter,
    so it already counts once); ``r l ŕ ĺ`` between consonants are syllabic
    (``vlk``, ``prst``). Vowel-less clitics (``z v k s``) lean on the next word
    and count 0. The floor of 1 per line exists because the duration gate and
    the token budget both scale by this number.
    """
    total = 0
    for word in _WORD.findall(text.lower()):
        nuclei = sum(ch in _VOWELS for ch in word) - len(_DIPHTHONG.findall(word))
        total += nuclei + len(_SYLLABIC_RL.findall(word))
    return max(1, total)


def known_tokens() -> frozenset[str]:
    """Control tokens the TTS server understands, as ``category:name`` keys."""
    return _load_tokens(TOKENS_PATH)


@lru_cache(maxsize=None)
def _load_tokens(path: Path) -> frozenset[str]:
    """Read tokens.json on first use only: another builder writes the file, and
    importing canon must not depend on it existing yet."""
    if path.is_file():
        found = _collect_tokens(json.loads(path.read_text(encoding="utf-8")), None)
        if found:
            return frozenset(found)
    return frozenset(f"{cat}:{name}" for cat, names in _DOCUMENTED_TOKENS.items() for name in names)


def _collect_tokens(node: object, category: str | None) -> set[str]:
    """Accept the plausible tokens.json shapes: category-keyed dicts (nested at
    any depth), lists of bare names under a category, or full ``<|cat:name|>``
    / ``cat:name`` strings anywhere. A ``cat:name`` string only counts when
    ``cat`` is a real category, so metadata such as the file's own
    ``"syntax": "<|category:name|>"`` example never becomes a token."""
    if isinstance(node, dict):
        found: set[str] = set()
        for key, value in node.items():
            found |= _collect_tokens(value, _category(key) or category)
        return found
    if isinstance(node, list):
        return set().union(*(_collect_tokens(item, category) for item in node))
    if isinstance(node, str):
        prefix, _, name = _token_key(node).rpartition(":")
        if prefix:
            category = _category(prefix)
        return {f"{category}:{name}"} if category and name else set()
    return set()


def _category(key: str) -> str | None:
    singular = key.lower().removesuffix("s")
    return singular if singular in _DOCUMENTED_TOKENS else None


def _token_key(raw: str) -> str:
    """``<|Emotion:Anger|>``, ``emotion:anger`` and `` anger `` compare equal."""
    return raw.strip().removeprefix("<|").removesuffix("|>").strip().lower()


def _resolve_typed_tokens(text: str, banned: Iterable[str], warnings: list[str]) -> str:
    """Keep typed tokens the server knows, refuse banned ones loudly, drop the rest.

    Banned entries may be ``category:name``, bare ``name`` or the full token, so
    whichever form voice.yaml uses matches. Banned is checked first: a token the
    DM typed on purpose that the voice forbids must fail, not vanish.
    """
    known = known_tokens()
    banned_keys = {_token_key(item) for item in banned}

    def resolve(match: re.Match[str]) -> str:
        key = _token_key(match.group(0))
        if key in banned_keys or key.rsplit(":", 1)[-1] in banned_keys:
            raise BannedToken(f"<|{key}|>")
        if key in known:
            return f" <|{key}|> "
        warnings.append(f"unknown token stripped: {match.group(0)}")
        return " "

    return _TOKEN.sub(resolve, text)


def _drawl(text: str) -> str:
    """``Brá**cho`` -> ``Bráácho``: the model performs the hold when the spelling
    carries it. 1-2 stars add one vowel, 3+ add two (longer spellings sound
    broken). Stars not following a vowel are markup noise and are dropped."""
    def stretch(match: re.Match[str]) -> str:
        vowel, stars = match.group(1), match.group(2)
        return vowel * (2 if len(stars) <= 2 else 3)

    return _STRAY_STARS.sub("", _DRAWL.sub(stretch, text))


def _beats(text: str) -> str:
    """Written pauses become model pauses: Bag's comedic timing lives in his
    dashes and ellipses. Punctuation left dangling after a token is re-attached
    to the word before it so ``spoken`` stays well formed (``Čo? <pause>``)."""
    out = _LONG_BEAT.sub(f" {LONG_PAUSE} ", text)
    out = _BEAT.sub(f" {PAUSE} ", out)
    return _PUNCT_AFTER_TOKEN.sub(r"\2 \1", _collapse(out))


def _cap_pauses(text: str, warnings: list[str]) -> str:
    """Keep the first MAX_PAUSES pause tokens; a line full of holds stops
    sounding like timing and starts sounding like a stall."""
    seen = 0

    def keep(match: re.Match[str]) -> str:
        nonlocal seen
        seen += 1
        return match.group(0) if seen <= MAX_PAUSES else " "

    out = _PAUSE_TOKEN.sub(keep, text)
    if seen > MAX_PAUSES:
        warnings.append(f"{seen - MAX_PAUSES} pause token(s) beyond {MAX_PAUSES} dropped")
    return out


def _ensure_terminal(text: str) -> str:
    """Give the last spoken word a full stop if it has none: without terminal
    punctuation the model trails off or invents a continuation."""
    words = text.split()
    for i in range(len(words) - 1, -1, -1):
        if _TOKEN.fullmatch(words[i]):
            continue
        word = words[i].rstrip(",;:")
        if word.rstrip(_CLOSERS)[-1:] not in _TERMINAL:
            word += "."
        words[i] = word
        return " ".join(words)
    raise ValueError("empty line")


def _spoken(text: str) -> str:
    return _collapse(_TOKEN.sub(" ", text))


def _collapse(text: str) -> str:
    return " ".join(text.split())
