"""Unit tests for app.canon (contract: docs/M1-contracts.md). No network, no GPU."""
from __future__ import annotations

import json
import unicodedata
from pathlib import Path

import pytest

from app import canon
from app.canon import (
    LONG_PAUSE, MAX_CHARS, PAUSE, BannedToken, Canon, TooLong, canonicalize, syllables,
)

DOCUMENTED = {
    "emotion": ["amusement", "anger", "enthusiasm"],
    "style": ["shouting", "whispering"],
    "sfx": ["sigh", "laughter"],
    "prosody": ["pause", "long_pause", "pitch_high", "expressive_low"],
}


@pytest.fixture(autouse=True)
def tokens_file(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Point canon at a private tokens.json so tests do not depend on whether
    the real configs/tokens.json has been written yet (another builder owns it)."""
    path = tmp_path / "tokens.json"
    path.write_text(json.dumps(DOCUMENTED), encoding="utf-8")
    monkeypatch.setattr(canon, "TOKENS_PATH", path)
    return path


# --------------------------------------------------------------------- drawls
@pytest.mark.parametrize("typed", ["Brá*cho.", "Brá**cho."])
def test_drawl_one_or_two_stars_adds_one_vowel(typed: str) -> None:
    c = canonicalize(typed)
    assert c.text == "Bráácho."
    assert c.spoken == "Bráácho."


@pytest.mark.parametrize("typed", ["Brá***cho.", "Brá******cho."])
def test_drawl_three_or_more_stars_adds_two_vowels(typed: str) -> None:
    assert canonicalize(typed).text == "Brááácho."


def test_stars_not_after_a_vowel_are_dropped() -> None:
    assert canonicalize("Hm** prst**.").text == "Hm prst."


def test_drawl_keeps_the_vowel_case() -> None:
    assert canonicalize("Á**no.").text == "ÁÁno."


# --------------------------------------------------------------------- beats
@pytest.mark.parametrize("dash", [" — ", " - ", " – "])
def test_spaced_dashes_become_pause_tokens(dash: str) -> None:
    c = canonicalize(f"Ten{dash}nie.")
    assert c.text == f"Ten {PAUSE} nie."
    assert c.spoken == "Ten nie."


def test_unspaced_em_dash_is_a_pause_but_hyphen_is_not() -> None:
    assert canonicalize("Ten—nie.").text == f"Ten {PAUSE} nie."
    assert canonicalize("Slovensko-maďarský spor.").text == "Slovensko-maďarský spor."


@pytest.mark.parametrize("ellipsis", ["…", "...", "....", "… "])
def test_ellipsis_becomes_long_pause(ellipsis: str) -> None:
    c = canonicalize(f"Hmm{ellipsis}ten nie.")
    assert c.text == f"Hmm {LONG_PAUSE} ten nie."
    assert c.spoken == "Hmm ten nie."


def test_trailing_ellipsis_filler_keeps_terminal_punctuation() -> None:
    c = canonicalize("Hmm…")
    assert c.text == f"Hmm. {LONG_PAUSE}"
    assert c.spoken == "Hmm."


def test_punctuation_after_a_beat_reattaches_to_the_word() -> None:
    c = canonicalize("Čo…?")
    assert c.text == f"Čo? {LONG_PAUSE}"
    assert c.spoken == "Čo?"


def test_max_three_pause_tokens_extras_dropped_with_warning() -> None:
    c = canonicalize("a — b — c… d — e - f.")
    assert c.text == f"a {PAUSE} b {PAUSE} c {LONG_PAUSE} d e f."
    assert c.spoken == "a b c d e f."
    assert len(c.warnings) == 1 and "2 pause token(s)" in c.warnings[0]


def test_typed_pause_tokens_count_toward_the_cap() -> None:
    c = canonicalize(f"a {PAUSE} b {PAUSE} c {PAUSE} d — e.")
    assert c.text.count("pause") == 3
    assert c.warnings


# ----------------------------------------------------------------------- cap
def test_spoken_over_max_chars_raises_too_long() -> None:
    with pytest.raises(TooLong):
        canonicalize("x" * MAX_CHARS + ".")


def test_spoken_at_exactly_max_chars_is_accepted() -> None:
    assert len(canonicalize("x" * (MAX_CHARS - 1) + ".").spoken) == MAX_CHARS


def test_cap_measures_spoken_text_not_tokens() -> None:
    body = "x" * (MAX_CHARS - 1) + "."
    c = canonicalize(f"<|emotion:anger|> {body}")
    assert len(c.spoken) == MAX_CHARS
    assert len(c.text) > MAX_CHARS


# -------------------------------------------------------------------- tokens
def test_known_typed_token_is_kept_and_stripped_from_spoken() -> None:
    c = canonicalize("Ten <|emotion:anger|>nie.")
    assert c.text == "Ten <|emotion:anger|> nie."
    assert c.spoken == "Ten nie."
    assert c.warnings == []


def test_unknown_typed_token_is_stripped_with_warning() -> None:
    c = canonicalize("Ten <|emotion:rage|> nie.")
    assert c.text == "Ten nie."
    assert c.warnings == ["unknown token stripped: <|emotion:rage|>"]


@pytest.mark.parametrize("banned", [{"style:shouting"}, {"shouting"}, {"<|style:shouting|>"}, ["style:shouting"]])
def test_banned_token_raises(banned) -> None:
    with pytest.raises(BannedToken):
        canonicalize("Ten <|style:shouting|> nie.", banned=banned)


def test_banned_check_beats_unknown_stripping() -> None:
    with pytest.raises(BannedToken):
        canonicalize("Ten <|style:nonsense|> nie.", banned={"nonsense"})


def test_tokens_fall_back_to_documented_list_when_file_absent(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(canon, "TOKENS_PATH", tmp_path / "missing.json")
    known = canon.known_tokens()
    assert {"emotion:anger", "style:whispering", "sfx:burping", "prosody:speed_very_slow"} <= known
    assert canonicalize("Ten <|prosody:speed_slow|> nie.").text == "Ten <|prosody:speed_slow|> nie."


@pytest.mark.parametrize("payload", [
    {"tokens": ["<|emotion:anger|>", "<|sfx:sigh|>"]},
    {"version": 1, "categories": {"emotions": ["anger"], "sfx": ["sigh"]}},
    {"emotion": {"anger": "<|emotion:anger|>"}, "sfx": {"sigh": "<|sfx:sigh|>"}},
    # the shape configs/tokens.json actually uses: metadata strings must not leak in
    {"source": "cookbook", "syntax": "<|category:name|>", "categories": {"emotion": ["anger"], "sfx": ["sigh"]}},
])
def test_tokens_file_shapes(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, payload: dict) -> None:
    path = tmp_path / "shape.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    monkeypatch.setattr(canon, "TOKENS_PATH", path)
    assert canon.known_tokens() == {"emotion:anger", "sfx:sigh"}


# ------------------------------------------------------------- normalisation
def test_nfc_quotes_and_whitespace_are_normalised() -> None:
    c = canonicalize("  „Ahoj“ \tsvet,\n  áno  ")
    assert c.text == '"Ahoj" svet, áno.'
    assert c.text == unicodedata.normalize("NFC", c.text)


@pytest.mark.parametrize("typed, expected", [
    ("Ahoj", "Ahoj."),
    ("Ahoj,", "Ahoj."),
    ("Ahoj!", "Ahoj!"),
    ("Ahoj?", "Ahoj?"),
    ('"Ahoj."', '"Ahoj."'),
    ("Ahoj <|sfx:laughter|>", "Ahoj. <|sfx:laughter|>"),
])
def test_terminal_punctuation(typed: str, expected: str) -> None:
    assert canonicalize(typed).text == expected


def test_empty_line_is_refused() -> None:
    with pytest.raises(ValueError):
        canonicalize("  <|emotion:anger|> ")


def test_canonicalize_is_deterministic() -> None:
    a, b = canonicalize("Brá**cho — ty… <|sfx:sigh|> ideš?"), canonicalize("Brá**cho — ty… <|sfx:sigh|> ideš?")
    assert a == b
    assert isinstance(a, Canon)


def test_english_lang_is_accepted() -> None:
    assert canonicalize("Not that one", lang="en").text == "Not that one."


# ---------------------------------------------------------------- syllables
def test_syllables_contract_sentences() -> None:
    assert canonicalize("Ste ako hrdinovia z balady.").syllables == 10
    assert canonicalize("Pozri, kamoš, toto nie je hra.").syllables == 9


def test_syllables_vowelless_clitics_count_zero() -> None:
    assert syllables("z") == 1  # floor: a line is never zero syllables
    assert syllables("vlk z hôr") == 2
    assert syllables("Prst v krku.") == 3
    assert syllables("k nám s ním") == 2


def test_syllables_diphthongs_and_o_circumflex_count_once() -> None:
    assert syllables("Kôň stojí.") == 3
    assert syllables("vieme") == 2
    assert syllables("Ázia") == 2


def test_syllables_syllabic_r_l() -> None:
    assert syllables("vŕba") == 2
    assert syllables("Bráácho") == 3
    assert syllables("krk a vlk") == 3
