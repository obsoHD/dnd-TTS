"""Assembled-service smoke for M3: the delivery selector and the Writer as the
table actually reaches them, through ``app.main``'s real boot sequence.

The unit suites already prove ``app.delivery`` and ``app.llm`` in isolation.
What is proved here is that the four M3 lanes fit together behind one
``TestClient``: the two Writer routes are mounted, ``GET /api/deliveries``
answers the shape the improv bar reads, the pencil's 503 carries the Slovak the
banner shows verbatim, and a delivery chosen at the boundary ends up inside
``job.text`` -- which is the whole reason the delivery is a text function and
never a worker, store or ``render_line`` argument.

Nothing here touches the network, the GPU box or a real LLM: ``render_line`` is
a half second of silence, the readiness probes are stubbed, and ``app.llm.fix``
and ``app.llm.residency`` are replaced before the app is built.
"""
from __future__ import annotations

import importlib
from pathlib import Path
from typing import Iterator

import pytest
import yaml
from fastapi.testclient import TestClient

from app import canon, config, delivery, llm, render, store, tts_client, voices, worker
from app.api import health as health_api
from app.render import RenderResult
from app.tts_client import Sampler
from app.voices import Voice

SR = 24_000
VOICE = "bag"
TEXT = "Ten nie je na predaj, to je rodinné dedičstvo."
SPICE = "vzdych"                       # <|sfx:sigh|>: documented in canon, not banned on Bag
BRAIN_DOWN = "mozog nie je pripravený"
FIXED = "Ten nie je na predaj — to je rodinné dedičstvo."
ARMED = {"sim_drop": 0.011, "min_sim": 0.918, "n": 20, "armed_at": "2026-09-04T10:00:00"}
DELIVERY_KEYS = {"id", "label", "token", "armed", "measured"}

# What ``app.llm.fix`` does on the next call: a dict is returned, an exception is
# raised. One switch keeps the stub a plain function that never sees a socket.
FIX_RESULT: list[object] = []


def _fix(text: str, voice: Voice, lang: str = "sk") -> dict:
    outcome = FIX_RESULT[-1]
    if isinstance(outcome, Exception):
        raise outcome
    return dict(outcome)


def make_voice() -> Voice:
    """A locked, uncalibrated Bag: locked so boot pre-renders its bank, and
    uncalibrated so every spice starts unarmed, which is what the contract says
    a voice ships as ("Bag ships with zero armed spices")."""
    return Voice(id=VOICE, label="Mr. Bag", lang="sk", version=1, ref_file="ref.wav", ref_sha256="0" * 64,
                 ref_transcript="Popravia? Dostane tretí obed.", ref_tts_path="/refs/bag/ref.wav",
                 sampler=Sampler(), golden_seed=0, gate=dict(voices.UNCALIBRATED_GATE), master={"energy": 65})


def fake_render_line(voice: Voice, text: str, take_no: int = 0, n_takes: int = 2, *,
                     cancel=None) -> RenderResult:
    """Silence under the real cache key, so no TTS, GPU or model is involved."""
    c = canon.canonicalize(text, lang=voice.lang, banned=set(voice.banned_tokens))
    pcm = bytes(SR)
    return RenderResult(render_id=render.render_id(voice, c.text, take_no), voice_id=voice.id, text=text,
                        canon=c, seed=1, take_no=take_no, raw_pcm=pcm, pcm=pcm, sr=SR, sim=0.91, cer=0.04,
                        verified=True, gate="pass", scores=[], timings={}, voice_version=voice.version,
                        recipe_version=render.RECIPE_VERSION, master_version="m" * 64, lufs=-18.0)


@pytest.fixture(scope="module")
def client(tmp_path_factory) -> Iterator[TestClient]:
    """One booted app for the module. ``BAG_DATA`` is set and ``config``
    reloaded before ``app.main`` is imported, exactly as the container does."""
    data = tmp_path_factory.mktemp("m3")
    mp = pytest.MonkeyPatch()
    mp.setenv("BAG_DATA", str(data))
    importlib.reload(config)
    voices.save_voice(make_voice())
    mp.setattr(render, "render_line", fake_render_line)
    mp.setattr(worker, "render_line", fake_render_line)       # bound by ``from app.render import render_line``
    mp.setattr(tts_client, "health", lambda: True)
    mp.setattr(health_api, "stt_up", lambda: True)
    mp.setattr(health_api, "llm_residency", lambda: "resident")
    mp.setattr(llm, "residency", lambda: "resident")
    mp.setattr(llm, "fix", _fix)
    from app.main import create_app

    with TestClient(create_app()) as c:
        yield c
    mp.undo()
    importlib.reload(config)


def arm(spice_id: str | None) -> None:
    """Write (or clear) an ``armed_spices`` block straight into voice.yaml, the
    way ``scripts/arm_spice.py --apply`` does. ``POST /api/say`` re-reads the
    file, so no app state has to be reached into."""
    path = voices.voice_yaml(VOICE)
    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    data["armed_spices"] = {spice_id: dict(ARMED)} if spice_id else {}
    path.write_text(yaml.safe_dump(data, allow_unicode=True, sort_keys=False), encoding="utf-8")


def say(client: TestClient, **extra):
    return client.post("/api/say", json={"voice": VOICE, "text": TEXT, "lang": "sk", **extra})


def test_deliveries_lists_bare_first_and_nothing_armed(client: TestClient) -> None:
    """An uncalibrated voice has earned nothing: the DM sees every spice, greyed."""
    arm(None)
    r = client.get("/api/deliveries", params={"voice": VOICE})
    assert r.status_code == 200, r.text
    items = r.json()
    assert all(set(item) == DELIVERY_KEYS for item in items)
    assert items[0] == {"id": delivery.BARE, "label": delivery.BARE_LABEL, "token": "",
                        "armed": True, "measured": None}
    spices = items[1:]
    assert [item["id"] for item in spices] == [s.id for s in delivery.SPICES]
    assert not any(item["armed"] for item in spices)
    assert all(item["measured"] is None for item in spices)


def test_deliveries_unknown_voice_is_404(client: TestClient) -> None:
    assert client.get("/api/deliveries", params={"voice": "nikto"}).status_code == 404


def test_fix_returns_the_writers_line(client: TestClient) -> None:
    FIX_RESULT.append({"text": FIXED, "original": TEXT, "changed": True, "note": ""})
    r = client.post("/api/fix", json={"voice": VOICE, "text": TEXT, "lang": "sk"})
    assert r.status_code == 200, r.text
    assert r.json() == {"text": FIXED, "original": TEXT, "changed": True, "note": ""}
    FIX_RESULT.pop()


def test_fix_without_a_resident_brain_is_503_in_slovak(client: TestClient) -> None:
    """The banner shows this string verbatim, so the wording is part of the contract."""
    FIX_RESULT.append(llm.BrainNotReady(config.LLM_MODEL))
    r = client.post("/api/fix", json={"voice": VOICE, "text": TEXT, "lang": "sk"})
    assert r.status_code == 503
    assert r.json()["detail"] == BRAIN_DOWN
    FIX_RESULT.pop()


def test_say_bare_is_accepted(client: TestClient) -> None:
    arm(None)
    r = say(client, delivery="bare")
    assert r.status_code == 200, r.text
    assert set(r.json()) == {"job_id", "render_id", "cached", "position"}


def test_say_with_an_unarmed_spice_is_400(client: TestClient) -> None:
    """The UI never offers it; a hand-made request still must not render bare
    behind the DM's back, which is why this is a refusal and not a fallback."""
    arm(None)
    r = say(client, delivery=SPICE)
    assert r.status_code == 400
    assert SPICE in r.json()["detail"]


def test_armed_spice_travels_inside_the_job_text(client: TestClient) -> None:
    """The point of the whole design: the token is placed at the API boundary,
    after the first word, and the worker only ever sees text."""
    arm(SPICE)
    token = next(s.token for s in delivery.SPICES if s.id == SPICE)
    r = say(client, delivery=SPICE)
    assert r.status_code == 200, r.text
    job = client.get(f"/api/jobs/{r.json()['job_id']}").json()
    first, rest = TEXT.split(" ", 1)
    assert job["text"] == f"{first} {token}{rest}"
    arm(None)


def test_no_delivery_reaches_the_render_path(client: TestClient) -> None:
    """A guard on the contract itself: ``delivery`` is a field on ``POST /api/say``
    and nowhere else, so a refactor that threads it deeper fails here."""
    src = "".join(Path(m.__file__).read_text(encoding="utf-8") for m in (worker, store, render))
    assert "delivery" not in src
