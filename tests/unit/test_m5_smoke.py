"""End-to-end smoke of M5 on the assembled service: a tile remembers its tone.

M1..M4 have their own smoke tests (``test_api_smoke.py``, ``test_m3_smoke.py``,
``test_m4_smoke.py``); this one exists for the single thing M5 adds and that can
only be proved once the board owner's and the web owner's work sit behind one
``TestClient``:

* a line saved while the gear said *smiech* keeps that tone through
  ``POST /api/lines``, reports it on ``GET /api/board`` as ``delivery`` +
  ``delivery_armed``, and the job its tap creates carries the token **after the
  first word** -- the M3 placement rule, reached through a tile instead of the
  improv bar;
* ``"bare"`` clears the tone and the next job goes plain again;
* an unarmed tone is refused (400) on both ``POST /api/lines`` and
  ``PATCH /api/lines/{id}``, and refusing it leaves the tile as it was;
* a tone the Lab **disarms after the save** degrades the tile to a plain line
  instead of erroring. This is the failure mode the contract cares about most:
  the Lab re-measures between a session and the next one, and silence at the
  table is worse than a flat delivery.

Nothing here touches the network, the GPU box or a real LLM: ``render_line``
returns silence under the real cache key, the three readiness probes are
stubbed and the brain is never warmed. The bank is a two-line stand-in written
into ``BAG_DATA`` before boot, so ``_seed_bank`` leaves it alone and the boot
pre-render is over in a moment; the shipped bank is exercised by the M2 smoke.
"""
from __future__ import annotations

import importlib
import json
from pathlib import Path
from typing import Iterator

import pytest
from fastapi.testclient import TestClient

from app import canon, config, llm, render, tts_client, voices, worker
from app.api import health as health_api
from app.render import RenderResult
from app.tts_client import Sampler
from app.voices import Voice

SR = 24_000
SPICE = "smiech"                      # armed for the test voice below
TOKEN = "<|sfx:laughter|>"            # app.delivery.SPICES: smiech
UNARMED = "vzdych"                    # a real spice this voice never measured
MEASURED = {"sim_drop": 0.011, "min_sim": 0.918, "n": 20, "armed_at": "2026-09-04T12:00:00Z"}

TEXT = "Ten mešec nie je na predaj."
TONED = f"Ten {TOKEN}mešec nie je na predaj."
SIGNATURE = "Ten nie."                # board.SIGNATURE_CATEGORY["sk"]["bag"]
BANK = {"sk": {"bag": {"Vítanie": ["Vitaj, pocestný.", "Sadni si."], SIGNATURE: ["Ten nie."]}}}
# The keys every board entry carries after M5; frozen here so a silent change to
# the read model the Play page draws from fails a test instead of a tile.
LINE_KEYS = {"id", "text", "category", "status", "render_id", "favourite", "slot",
             "delivery", "delivery_armed"}

# The line the walk creates, shared between the ordered tests below.
STATE: dict[str, str] = {}


def make_voice(armed: dict) -> Voice:
    """A locked Bag (``ref_sha256`` set, so boot pre-renders its bank) with one
    spice armed, exactly as ``scripts/arm_spice.py --apply`` would leave it."""
    return Voice(id="bag", label="Mr. Bag", lang="sk", version=1, ref_file="ref.wav", ref_sha256="0" * 64,
                 ref_transcript="Popravia? Dostane tretí obed.", ref_tts_path="/refs/bag/ref.wav",
                 sampler=Sampler(), golden_seed=0, gate=dict(voices.UNCALIBRATED_GATE),
                 master={"energy": 65}, armed_spices=armed)


def fake_render_line(voice: Voice, text: str, take_no: int = 0, n_takes: int = 2, *,
                     cancel=None) -> RenderResult:
    """Half a second of silence under the real cache key, so the worker's
    ``render_id`` and the row it stores agree exactly as they would with the TTS."""
    c = canon.canonicalize(text, lang=voice.lang, banned=set(voice.banned_tokens))
    pcm = bytes(SR)
    return RenderResult(render_id=render.render_id(voice, c.text, take_no), voice_id=voice.id, text=text,
                        canon=c, seed=1, take_no=take_no, raw_pcm=pcm, pcm=pcm, sr=SR, sim=0.91, cer=0.04,
                        verified=True, gate="pass", scores=[], timings={}, voice_version=voice.version,
                        recipe_version=render.RECIPE_VERSION, master_version="m" * 64, lufs=-18.0)


@pytest.fixture(scope="module")
def client(tmp_path_factory) -> Iterator[TestClient]:
    """One booted app for the module: the tests below are one walk through the
    life of a single tile, in order. ``BAG_DATA`` is set and ``config`` reloaded
    before ``app.main`` is imported, as the container does."""
    data: Path = tmp_path_factory.mktemp("m5")
    mp = pytest.MonkeyPatch()
    mp.setenv("BAG_DATA", str(data))
    mp.delenv("BAG_REMOTE_KEY", raising=False)
    importlib.reload(config)
    (data / "phrases.json").write_text(json.dumps(BANK, ensure_ascii=False), encoding="utf-8")
    voices.save_voice(make_voice({SPICE: MEASURED}))
    mp.setattr(render, "render_line", fake_render_line)
    mp.setattr(worker, "render_line", fake_render_line)       # bound by ``from app.render import render_line``
    mp.setattr(tts_client, "health", lambda: True)
    mp.setattr(health_api, "stt_up", lambda: True)
    mp.setattr(health_api, "llm_residency", lambda: "resident")
    mp.setattr(llm, "warm_in_background", lambda: None)       # no thread, no ollama, no network
    from app.main import create_app

    with TestClient(create_app()) as c:
        yield c
    mp.undo()
    importlib.reload(config)


def entry(client: TestClient, line_id: str) -> dict:
    """The board's own row for one tile: everything M5 promises the UI is read
    back through ``GET /api/board``, never from the writer's response alone."""
    board = client.get("/api/board", params={"voice": "bag", "lang": "sk"}).json()
    found = [line for line in board["lines"] if line["id"] == line_id]
    assert found, f"{line_id} is not on the board"
    return found[0]


def job_text(client: TestClient, job_id: str) -> str:
    """The text the worker was actually handed. This is where a delivery has to
    be visible: it travels inside ``job.text`` and nowhere else."""
    r = client.get(f"/api/jobs/{job_id}")
    assert r.status_code == 200, r.text
    return r.json()["text"]


def say(client: TestClient, line: dict) -> str:
    """A tap on a tile, the way ``web/app.js`` sends it: the line's own tone,
    and only while it is still armed (``lineDelivery``)."""
    r = client.post("/api/say", json={"voice": "bag", "lang": "sk", "text": line["text"],
                                      "line_id": line["id"],
                                      "delivery": line["delivery"] if line["delivery_armed"] else None})
    assert r.status_code == 200, r.text
    return job_text(client, r.json()["job_id"])


def disarm(armed: dict) -> None:
    """Rewrite ``voice.yaml``'s ``armed_spices`` the way the Lab does, in place,
    while the app keeps running -- the board re-reads the file per read."""
    voice = voices.load_voice("bag")
    voice.armed_spices = dict(armed)
    voices.save_voice(voice)


def test_deliveries_offer_the_armed_spice(client: TestClient) -> None:
    """The gear's list is what the UI may offer; the tone under test must be in
    it as armed, or the rest of this walk proves nothing."""
    r = client.get("/api/deliveries", params={"voice": "bag"})
    assert r.status_code == 200, r.text
    armed = {d["id"] for d in r.json() if d["armed"]}
    assert armed == {"bare", SPICE}


def test_saving_a_line_keeps_the_tone(client: TestClient) -> None:
    r = client.post("/api/lines", json={"voice": "bag", "lang": "sk", "text": TEXT, "delivery": SPICE})
    assert r.status_code == 200, r.text
    line = r.json()
    assert set(line) == LINE_KEYS
    assert line["delivery"] == SPICE and line["delivery_armed"] is True
    assert line["text"] == TEXT                      # the tone is stored beside the text, never inside it
    assert line["category"] == "Moje"                # M4's SAVED_CATEGORY, unchanged
    STATE["line_id"] = line["id"]


def test_board_reports_the_tone(client: TestClient) -> None:
    board = client.get("/api/board", params={"voice": "bag", "lang": "sk"}).json()
    assert all(set(line) == LINE_KEYS for line in board["lines"])
    line = entry(client, STATE["line_id"])
    assert line["delivery"] == SPICE and line["delivery_armed"] is True
    bank = [b for b in board["lines"] if b["category"] == "Vítanie"]
    assert bank and all(b["delivery"] is None and b["delivery_armed"] is False for b in bank), \
        "import_bank must never set a tone"


def test_tap_places_the_token_after_the_first_word(client: TestClient) -> None:
    text = say(client, entry(client, STATE["line_id"]))
    first, _, rest = text.partition(" ")
    assert first == "Ten" and rest.startswith(TOKEN), text
    assert text == TONED


def test_regenerate_keeps_the_tiles_tone(client: TestClient) -> None:
    """``line_text`` is the one rule, so Regenerate sounds like the tile did."""
    r = client.post(f"/api/lines/{STATE['line_id']}/regenerate")
    assert r.status_code == 200, r.text
    assert job_text(client, r.json()["job_id"]) == TONED


def test_saving_an_unarmed_tone_is_refused(client: TestClient) -> None:
    r = client.post("/api/lines", json={"voice": "bag", "lang": "sk", "text": "Toto neplatí.",
                                        "delivery": UNARMED})
    assert r.status_code == 400
    assert UNARMED in r.json()["detail"]


def test_patching_an_unarmed_tone_is_refused_and_changes_nothing(client: TestClient) -> None:
    r = client.patch(f"/api/lines/{STATE['line_id']}", json={"delivery": UNARMED, "favourite": True})
    assert r.status_code == 400
    assert UNARMED in r.json()["detail"]
    line = entry(client, STATE["line_id"])
    assert line["delivery"] == SPICE, "a refused tone must leave the stored one alone"
    assert line["favourite"] is False, "the whole patch is one transaction"


def test_bare_clears_the_tone_and_the_job_goes_plain(client: TestClient) -> None:
    r = client.patch(f"/api/lines/{STATE['line_id']}", json={"delivery": "bare"})
    assert r.status_code == 200, r.text
    assert r.json()["delivery"] is None and r.json()["delivery_armed"] is False
    assert entry(client, STATE["line_id"])["delivery"] is None
    assert say(client, entry(client, STATE["line_id"])) == TEXT
    r = client.post(f"/api/lines/{STATE['line_id']}/regenerate")
    assert job_text(client, r.json()["job_id"]) == TEXT


def test_re_toning_the_same_text_makes_no_second_tile(client: TestClient) -> None:
    """The id hashes voice+lang+category+text and not the tone, so the star
    re-tones the tile the DM already has instead of growing a twin."""
    before = len(client.get("/api/board", params={"voice": "bag", "lang": "sk"}).json()["lines"])
    r = client.post("/api/lines", json={"voice": "bag", "lang": "sk", "text": TEXT, "delivery": SPICE})
    assert r.status_code == 200, r.text
    assert r.json()["id"] == STATE["line_id"] and r.json()["delivery"] == SPICE
    after = client.get("/api/board", params={"voice": "bag", "lang": "sk"}).json()["lines"]
    assert len(after) == before


def test_a_disarmed_spice_degrades_to_a_plain_line(client: TestClient) -> None:
    """The Lab re-measures and the tone loses its arming *after* the save. The
    tile must still be there, still playable, and plainly marked -- never a 400
    and never a missing tile."""
    disarm({})
    line = entry(client, STATE["line_id"])
    assert line["delivery"] == SPICE, "the DM's choice is remembered, not forgotten"
    assert line["delivery_armed"] is False
    assert client.get("/api/deliveries", params={"voice": "bag"}).json()[1]["armed"] is False

    assert say(client, line) == TEXT                              # the tap the web sends: bare
    r = client.post(f"/api/lines/{STATE['line_id']}/regenerate")  # the server's own path: bare, not an error
    assert r.status_code == 200, r.text
    assert job_text(client, r.json()["job_id"]) == TEXT

    disarm({SPICE: MEASURED})                                     # re-armed: the same tile tones again
    assert entry(client, STATE["line_id"])["delivery_armed"] is True
