"""The Writer, with ``requests`` stubbed out entirely.

Nothing here reaches the network, the GPU box or ollama: every test replaces
``llm.requests.get`` / ``llm.requests.post`` with a callable that records what
was asked and answers from a canned payload. What is being tested is the half
of ``fix`` that does not trust the model - the guards that run after the answer
comes back - plus the residency reading the whole feature hangs on.
"""
from __future__ import annotations

import pytest
import requests
from fastapi import FastAPI
from fastapi.testclient import TestClient

from app import canon, config, llm
from app.api import write as write_api
from app.tts_client import Sampler
from app.voices import Voice

TEXT = "Ten nie."
LONG = "a" * (canon.MAX_CHARS + 10)


def make_voice(**over) -> Voice:
    """Bag as the roster knows him, with both personas and shouting banned."""
    data = dict(id="bag", label="Mr. Bag", lang="sk", version=1, ref_file="ref.wav", ref_sha256="0" * 64,
                ref_transcript="Popravia? Dostane tretí obed.", ref_tts_path="/refs/bag/ref.wav",
                sampler=Sampler(), golden_seed=0, gate={}, master={"energy": 65},
                banned_tokens=["style:shouting"],
                persona={"sk": "Si Vak, sarkastický predmet.", "en": "You are Bag, a sarcastic item."})
    return Voice(**{**data, **over})


def stub_ps(monkeypatch, payload, *, exc: Exception | None = None) -> list[dict]:
    """``GET /api/ps`` answered from ``payload`` (or raising ``exc``)."""
    calls: list[dict] = []

    def get(url, timeout=None, **kw):
        calls.append({"url": url, "timeout": timeout})
        if exc is not None:
            raise exc
        return _Response(payload)

    monkeypatch.setattr(llm.requests, "get", get)
    return calls


def stub_chat(monkeypatch, content: str | Exception, status: int = 200) -> list[dict]:
    """``POST /api/chat`` answered with one assistant message (or an exception)."""
    calls: list[dict] = []

    def post(url, json=None, timeout=None, **kw):
        calls.append({"url": url, "body": json, "timeout": timeout})
        if isinstance(content, Exception):
            raise content
        return _Response({"message": {"content": content}}, status)

    monkeypatch.setattr(llm.requests, "post", post)
    return calls


class _Response:
    def __init__(self, payload: dict, status: int = 200) -> None:
        self._payload = payload
        self.status_code = status

    def json(self) -> dict:
        return self._payload

    def raise_for_status(self) -> None:
        if self.status_code >= 400:
            raise requests.HTTPError(str(self.status_code))


def resident(monkeypatch) -> None:
    monkeypatch.setattr(llm, "residency", lambda: "resident")


# --------------------------------------------------------------- residency

def test_residency_resident(monkeypatch):
    stub_ps(monkeypatch, {"models": [{"name": config.LLM_MODEL, "size": 100, "size_vram": 95}]})
    assert llm.residency() == "resident"


def test_residency_loaded_when_spilled_to_cpu(monkeypatch):
    """Below the fraction ollama is running part of the model on CPU: it would
    answer, far too slowly for a DM mid-scene, so it is not ``resident``."""
    stub_ps(monkeypatch, {"models": [{"model": config.LLM_MODEL, "size": 100, "size_vram": 50}]})
    assert llm.residency() == "loaded"


def test_residency_absent_when_another_model_is_up(monkeypatch):
    stub_ps(monkeypatch, {"models": [{"name": "qwen3:14b", "size": 100, "size_vram": 100}]})
    assert llm.residency() == "absent"


def test_residency_absent_when_ollama_is_down(monkeypatch):
    stub_ps(monkeypatch, {}, exc=requests.ConnectionError("refused"))
    assert llm.residency() == "absent"


def test_residency_probe_is_short(monkeypatch):
    """The probe must not sit on the chat timeout; the pencil goes grey fast."""
    calls = stub_ps(monkeypatch, {"models": []})
    llm.residency()
    assert calls[0]["url"].endswith("/api/ps") and calls[0]["timeout"] == llm.PROBE_TIMEOUT_S


# ------------------------------------------------------------ brain not ready

@pytest.mark.parametrize("state", ["loaded", "absent"])
def test_fix_raises_brain_not_ready_and_never_calls_chat(monkeypatch, state):
    monkeypatch.setattr(llm, "residency", lambda: state)
    calls = stub_chat(monkeypatch, "čokoľvek")
    with pytest.raises(llm.BrainNotReady):
        llm.fix(TEXT, make_voice())
    assert calls == []


# ---------------------------------------------------------------- the call

def test_chat_request_follows_the_contract(monkeypatch):
    resident(monkeypatch)
    calls = stub_chat(monkeypatch, "Ten nie, kamoš.")
    llm.fix("ten nie kamos", make_voice())
    body = calls[0]["body"]
    assert calls[0]["url"].endswith("/api/chat") and calls[0]["timeout"] == llm.CHAT_TIMEOUT_S
    assert body["model"] == config.LLM_MODEL and body["stream"] is False
    assert body["think"] is False and body["keep_alive"] == -1
    assert body["options"] == {"temperature": 0.3, "num_ctx": 4096}
    assert body["messages"][1] == {"role": "user", "content": "ten nie kamos"}


def test_one_attempt_only(monkeypatch):
    """An empty answer is not retried: a second roll costs the table 8 s more."""
    resident(monkeypatch)
    calls = stub_chat(monkeypatch, "   ")
    out = llm.fix(TEXT, make_voice())
    assert len(calls) == 1 and out["changed"] is False and out["note"]


def test_prompt_is_monolingual_per_language(monkeypatch):
    resident(monkeypatch)
    calls = stub_chat(monkeypatch, "Not that one.")
    llm.fix("not that one", make_voice(), lang="en")
    system = calls[0]["body"]["messages"][0]["content"]
    assert llm.EN_FIX_SYSTEM in system and "You are Bag" in system
    assert llm.SK_FIX_SYSTEM not in system and "Si Vak" not in system


def test_persona_is_optional(monkeypatch):
    """A voice with no persona still gets fixed; it just gets no character note."""
    resident(monkeypatch)
    calls = stub_chat(monkeypatch, "Ten nie.")
    llm.fix("ten nie", make_voice(persona={}))
    assert calls[0]["body"]["messages"][0]["content"] == llm.SK_FIX_SYSTEM


# ----------------------------------------------------------------- success

def test_successful_fix_reports_changed(monkeypatch):
    resident(monkeypatch)
    stub_chat(monkeypatch, "Ľudia, počkajte — toto sa vám nebude páčiť.")
    out = llm.fix("ludia pockajte toto sa vam nebude pacit", make_voice())
    assert out["changed"] is True and out["note"] == ""
    assert out["text"] == "Ľudia, počkajte — toto sa vám nebude páčiť."
    assert out["original"] == "ludia pockajte toto sa vam nebude pacit"


def test_identical_answer_is_not_a_change(monkeypatch):
    """Nothing to correct still owes the DM a note; a silent no-op reads as a
    broken button."""
    resident(monkeypatch)
    stub_chat(monkeypatch, TEXT)
    out = llm.fix(TEXT, make_voice())
    assert out["changed"] is False and out["text"] == TEXT and out["note"]


# ------------------------------------------------------------------ guards

def test_control_tokens_are_stripped(monkeypatch):
    """A delivery is the DM's choice, armed in the Lab. The Writer never picks one."""
    resident(monkeypatch)
    stub_chat(monkeypatch, "Ten <|emotion:anger|>nie, <|sfx:sigh|>kamoš.")
    out = llm.fix("ten nie kamos", make_voice())
    assert "<|" not in out["text"] and out["text"] == "Ten nie, kamoš."


def test_beat_marks_are_capped_at_three(monkeypatch):
    resident(monkeypatch)
    stub_chat(monkeypatch, "Nie — nie — nie — nie — nie.")
    out = llm.fix("nie nie nie nie nie", make_voice())
    assert out["text"].count("—") == llm.MAX_BEATS
    assert out["text"] == "Nie — nie — nie — nie nie."


def test_beat_cap_counts_every_beat_shape(monkeypatch):
    """Em dash, ellipsis and spaced hyphen are all beats to canon, so all three
    count against the same cap."""
    resident(monkeypatch)
    stub_chat(monkeypatch, "No… tak - dobre — teda… fajn.")
    out = llm.fix("no tak dobre teda fajn", make_voice())
    assert sum(out["text"].count(m) for m in ("…", " - ", "—")) == llm.MAX_BEATS


def test_thinking_trace_is_removed(monkeypatch):
    resident(monkeypatch)
    stub_chat(monkeypatch, "<think>Užívateľ chce opravu.</think>\nTen nie, kamoš.")
    assert llm.fix("ten nie kamos", make_voice())["text"] == "Ten nie, kamoš."


def test_wrapping_quotes_are_removed(monkeypatch):
    resident(monkeypatch)
    stub_chat(monkeypatch, '„Ten nie, kamoš."')
    assert llm.fix("ten nie kamos", make_voice())["text"] == "Ten nie, kamoš."


def test_too_long_an_answer_returns_the_original(monkeypatch):
    resident(monkeypatch)
    stub_chat(monkeypatch, LONG)
    out = llm.fix("a" * (canon.MAX_CHARS - 10), make_voice())
    assert out["changed"] is False and out["text"] == "a" * (canon.MAX_CHARS - 10)
    assert str(canon.MAX_CHARS) in out["note"]


@pytest.mark.parametrize("answer", ["Ten.", "Ten nie, kamoš, a ani ten druhý, ani tretí, vôbec nikdy nie."])
def test_length_ratio_guard_returns_the_original(monkeypatch, answer):
    """Shorter than half or longer than double means the model rewrote the line
    instead of fixing it - the DM gets their own words back."""
    resident(monkeypatch)
    stub_chat(monkeypatch, answer)
    out = llm.fix("Ten nie, kamoš.", make_voice())
    assert out == {"text": "Ten nie, kamoš.", "original": "Ten nie, kamoš.", "changed": False,
                   "note": "oprava príliš zmenila dĺžku repliky"}


def test_a_banned_token_is_stripped_before_canon_sees_it(monkeypatch):
    """Guards run in order: stripping happens first, so a token the voice bans
    (shouting on Bag) can never reach canon and raise BannedToken."""
    resident(monkeypatch)
    stub_chat(monkeypatch, "Ten <|style:shouting|>nie!")
    out = llm.fix("ten nie", make_voice())
    assert out["changed"] is True and out["text"] == "Ten nie!"


def test_timeout_returns_the_original_with_a_note(monkeypatch):
    resident(monkeypatch)
    stub_chat(monkeypatch, requests.Timeout("slow"))
    out = llm.fix(TEXT, make_voice())
    assert out == {"text": TEXT, "original": TEXT, "changed": False,
                   "note": "mozog neodpovedal do 8 sekúnd"}


def test_transport_error_returns_the_original_with_a_note(monkeypatch):
    resident(monkeypatch)
    stub_chat(monkeypatch, requests.ConnectionError("refused"))
    out = llm.fix(TEXT, make_voice())
    assert out["changed"] is False and out["note"] == "mozog neodpovedal"


def test_http_error_returns_the_original(monkeypatch):
    resident(monkeypatch)
    stub_chat(monkeypatch, "Ten nie.", status=500)
    assert llm.fix(TEXT, make_voice())["changed"] is False


# -------------------------------------------------------------- /api/fix

@pytest.fixture()
def client() -> TestClient:
    """The writer's router alone on a bare app: no boot, no worker, no disk.
    ``app.state.voices`` is what ``app/main.py`` puts there at startup."""
    app = FastAPI()
    app.include_router(write_api.router)
    app.state.voices = {"bag": make_voice()}
    return TestClient(app)


def test_api_fix_returns_the_dict(client, monkeypatch):
    monkeypatch.setattr(write_api.llm, "fix",
                        lambda text, voice, lang="sk": {"text": "Ten nie.", "original": text,
                                                        "changed": True, "note": ""})
    r = client.post("/api/fix", json={"voice": "bag", "text": "ten nie"})
    assert r.status_code == 200
    assert r.json() == {"text": "Ten nie.", "original": "ten nie", "changed": True, "note": ""}


def test_api_fix_503_when_the_brain_is_down(client, monkeypatch):
    def raise_not_ready(text, voice, lang="sk"):
        raise llm.BrainNotReady(config.LLM_MODEL)

    monkeypatch.setattr(write_api.llm, "fix", raise_not_ready)
    r = client.post("/api/fix", json={"voice": "bag", "text": "ten nie"})
    assert r.status_code == 503 and r.json()["detail"] == "mozog nie je pripravený"


def test_api_fix_404_on_an_unknown_voice(client):
    assert client.post("/api/fix", json={"voice": "nikto", "text": "ten nie"}).status_code == 404


def test_api_deliveries_delegates_to_the_delivery_module(client, monkeypatch):
    """The router owns the route so ``app/delivery.py`` needs no FastAPI import."""
    listed = [{"id": "bare", "label": "normálne", "token": "", "armed": True, "measured": True}]

    class FakeDelivery:
        @staticmethod
        def available(voice):
            assert voice.id == "bag"
            return listed

    monkeypatch.setattr(write_api, "_delivery", lambda: FakeDelivery)
    r = client.get("/api/deliveries", params={"voice": "bag"})
    assert r.status_code == 200 and r.json() == listed
