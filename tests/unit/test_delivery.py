"""A delivery is one token the DM chose, placed in the text and nowhere else.

These tests pin the three properties the rebuild depends on: the token lands
after the first word (never leading, never twice), a delivery that has not been
measured in the Lab is refused instead of silently rendered bare, and whatever
``apply`` returns still survives ``canonicalize`` -- because the string it
produces is the string that gets hashed into the ``render_id`` and handed to the
TTS. No network, no GPU, no LLM: everything here is a pure function or a
voice.yaml in a temp dir.
"""
from __future__ import annotations

import importlib

import pytest
from fastapi import HTTPException

from app import canon, config, delivery, voices
from app.api import say as say_api
from app.tts_client import Sampler
from app.voices import Voice

SIGH = "<|sfx:sigh|>"
ANGER = "<|emotion:anger|>"
LINE = "Ten nie."
MEASURED = {"sim_drop": 0.011, "min_sim": 0.918, "n": 20, "armed_at": "2026-09-04T10:00:00+00:00"}


def make_voice(**overrides) -> Voice:
    """Bag as the Lab leaves him: shouting and pitch_high banned (§3), nothing
    armed unless a test arms it."""
    base = dict(id="bag", label="Mr. Bag", lang="sk", version=1, ref_file="ref.wav", ref_sha256="0" * 64,
                ref_transcript="Popravia? Dostane tretí obed.", ref_tts_path="/refs/bag/ref.wav",
                sampler=Sampler(), golden_seed=0, gate=dict(voices.UNCALIBRATED_GATE),
                master={"energy": 65}, banned_tokens=["style:shouting", "prosody:pitch_high"],
                armed_spices={})
    return Voice(**{**base, **overrides})


def armed(*ids: str, **overrides) -> Voice:
    return make_voice(armed_spices={sid: dict(MEASURED) for sid in ids}, **overrides)


@pytest.fixture
def data_dir(tmp_path, monkeypatch):
    """A temp DATA_DIR, as the container gets through BAG_DATA."""
    monkeypatch.setenv("BAG_DATA", str(tmp_path))
    importlib.reload(config)
    yield tmp_path
    monkeypatch.undo()
    importlib.reload(config)


def test_spice_table_matches_the_contract():
    assert delivery.BARE == "bare" and delivery.MAX_ARMED == 3
    assert [(s.id, s.token) for s in delivery.SPICES] == [
        ("vzdych", SIGH), ("smiech", "<|sfx:laughter|>"), ("pobavený", "<|emotion:amusement|>"),
        ("nahnevaný", ANGER), ("nadšený", "<|emotion:enthusiasm|>"),
        ("výrazne", "<|prosody:expressive_high|>"), ("krik", "<|style:shouting|>")]
    assert [s.label for s in delivery.SPICES if s.id == "pobavený"] == ["pobavený"]
    assert all(s.token.strip("<|>") in canon.known_tokens() for s in delivery.SPICES)


def test_token_goes_after_the_first_word():
    assert delivery.apply(LINE, "nahnevaný", armed("nahnevaný")) == f"Ten {ANGER}nie."


def test_leading_quote_stays_with_its_word():
    """The quote belongs to the first word, so the token still lands on the second."""
    assert delivery.apply('„Ten nie," povedal.', "vzdych", armed("vzdych")) == f'„Ten {SIGH}nie," povedal.'


def test_one_word_line_is_returned_unchanged():
    """No second word to carry the token, and one word never survives the gate
    with a delivery on it anyway."""
    for text in ("Nie.", "  Nie!  "):
        assert delivery.apply(text, "vzdych", armed("vzdych")) == text


def test_bare_and_none_leave_the_text_alone():
    v = armed("vzdych")
    for spice_id in (None, "", "   ", delivery.BARE):
        assert delivery.apply(LINE, spice_id, v) == LINE
        assert delivery.resolve(spice_id, v) is None


def test_apply_is_idempotent_and_never_adds_a_second_token():
    v = armed("nahnevaný")
    once = delivery.apply(LINE, "nahnevaný", v)
    assert delivery.apply(once, "nahnevaný", v) == once
    typed = f"Ten {'<|prosody:pause|>'}nie."
    assert delivery.apply(typed, "nahnevaný", v) == typed


def test_unknown_and_unarmed_are_refused():
    v = armed("vzdych")
    with pytest.raises(delivery.NotArmed):
        delivery.resolve("sarkasticky", v)
    with pytest.raises(delivery.NotArmed):
        delivery.apply(LINE, "nahnevaný", v)          # a real spice, not measured for this voice
    with pytest.raises(delivery.NotArmed):
        delivery.apply(LINE, "vzdych", make_voice())  # nothing armed at all


def test_a_banned_token_is_dropped_from_available_and_refused():
    """Bag forbids shouting, so ``krik`` is not offered even if voice.yaml was
    hand-edited to arm it: canonicalize would refuse the token anyway."""
    v = armed("krik")
    assert "krik" not in {entry["id"] for entry in delivery.available(v)}
    with pytest.raises(delivery.NotArmed):
        delivery.resolve("krik", v)


def test_available_lists_bare_first_then_every_spice_with_its_state():
    entries = delivery.available(armed("vzdych"))
    assert entries[0] == {"id": "bare", "label": "normálne", "token": "", "armed": True, "measured": None}
    assert [e["id"] for e in entries[1:]] == [s.id for s in delivery.SPICES if s.id != "krik"]
    assert all(set(e) == {"id", "label", "token", "armed", "measured"} for e in entries)
    by_id = {e["id"]: e for e in entries}
    assert by_id["vzdych"]["armed"] is True and by_id["vzdych"]["measured"] == MEASURED
    assert by_id["smiech"]["armed"] is False and by_id["smiech"]["measured"] is None


def test_more_than_max_armed_in_yaml_arms_only_the_smallest_drops():
    """A hand-edited voice.yaml must not widen the selector past MAX_ARMED."""
    v = make_voice(armed_spices={
        "vzdych": {**MEASURED, "sim_drop": 0.019},
        "smiech": {**MEASURED, "sim_drop": 0.004},
        "pobavený": {**MEASURED, "sim_drop": 0.008},
        "nadšený": {**MEASURED, "sim_drop": 0.012},
    })
    armed_ids = {e["id"] for e in delivery.available(v) if e["armed"] and e["id"] != delivery.BARE}
    assert armed_ids == {"smiech", "pobavený", "nadšený"}
    with pytest.raises(delivery.NotArmed):
        delivery.resolve("vzdych", v)


def test_applied_text_still_canonicalises():
    """What apply returns is what gets hashed and sent, so canon must accept it:
    the token survives, spaced, and the spoken text is unchanged."""
    v = armed("nahnevaný")
    text = delivery.apply(LINE, "nahnevaný", v)
    c = canon.canonicalize(text, lang=v.lang, banned=set(v.banned_tokens))
    assert c.text == f"Ten {ANGER} nie."
    assert c.spoken == canon.canonicalize(LINE, lang=v.lang).spoken
    assert c.warnings == []


def test_armed_spices_round_trips_and_a_yaml_without_it_still_loads(data_dir):
    voices.save_voice(armed("vzdych"))
    assert voices.load_voice("bag").armed_spices == {"vzdych": MEASURED}

    # A voice.yaml written before M3 has no armed_spices block at all.
    voices.voice_yaml("bag").write_text("id: bag\nlang: sk\nversion: 1\n", encoding="utf-8")
    assert voices.load_voice("bag").armed_spices == {}


def test_lock_reference_clears_the_armed_spices(data_dir, monkeypatch):
    """A new clip is a new voice: the old measurements described an embedding
    that no longer exists."""
    voices.save_voice(armed("vzdych"))
    monkeypatch.setattr(voices, "_convert_reference", lambda path: b"\x00\x10" * 24_000)
    clip = data_dir / "new.wav"
    clip.write_bytes(b"")

    v = voices.lock_reference("bag", clip, "Nová nahrávka.")
    assert v.armed_spices == {} and v.version == 2
    assert voices.load_voice("bag").armed_spices == {}


def test_say_applies_the_delivery_to_the_job_text(data_dir):
    """The delivery reaches the job as text; the worker never learns it exists."""
    voices.save_voice(armed("nahnevaný"))
    req = say_api.SayRequest(voice="bag", text=LINE, delivery="nahnevaný")
    assert say_api._delivered(req) == f"Ten {ANGER}nie."
    assert say_api._plan(req, say_api._delivered(req)).render_id != say_api._plan(req, LINE).render_id


def test_say_answers_400_for_an_unarmed_delivery_and_404_for_a_ghost_voice(data_dir):
    voices.save_voice(make_voice())
    with pytest.raises(HTTPException) as unarmed:
        say_api._delivered(say_api.SayRequest(voice="bag", text=LINE, delivery="nahnevaný"))
    assert unarmed.value.status_code == 400 and "nahnevaný" in unarmed.value.detail

    with pytest.raises(HTTPException) as ghost:
        say_api._delivered(say_api.SayRequest(voice="ghost", text=LINE, delivery="vzdych"))
    assert ghost.value.status_code == 404

    bare = say_api.SayRequest(voice="bag", text=LINE, delivery=delivery.BARE)
    assert say_api._delivered(bare) == LINE


def test_say_400_when_the_text_carries_a_banned_token(data_dir):
    """canon.BannedToken is the same 400: the DM typed a token Bag forbids."""
    voices.save_voice(make_voice())
    req = say_api.SayRequest(voice="bag", text="Ten <|style:shouting|>nie.")
    with pytest.raises(HTTPException) as banned:
        say_api._plan(req, req.text)
    assert banned.value.status_code == 400
