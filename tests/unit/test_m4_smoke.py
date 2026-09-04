"""End-to-end smoke of M4 on the assembled service: the real ``app.main`` boot
sequence with only the edges faked.

M1..M3 already have a smoke test (``test_api_smoke.py``); this one exists for the
four things the table asked for in M4 and can only be proved once every builder's
module is behind one ``TestClient``:

* the Voice Creator is actually reachable -- ``/creator`` and ``/api/creator/*``
  answer, which is exactly what a missing line in ``PEER_ROUTERS`` would break;
* a draft walks upload -> clip -> transcript -> character -> phrases -> commit
  and comes out the other side as a locked, calibrated, banked, pre-rendering
  voice, with the status the page reads asserted at every step;
* a phrase the DM saves becomes one tile (and only one, saved twice), a bank
  line refuses to be deleted, and their own line does not;
* a tile's ``line_id`` on ``POST /api/say`` makes the render the line's own, so
  the tile reads ``ready`` on the next ``/api/board`` instead of falling back to
  grey -- the exact bug §1 of the contract describes.

Nothing here touches the network, the GPU box, whisper, ffmpeg or a real LLM:
``render_line`` returns silence, ``app.stt.transcribe`` and ``app.llm``'s two
authoring calls are replaced, both ffmpeg conversions are synthetic sine, and
``voices.calibrate`` writes plausible numbers instead of spending three GPU
minutes. What is left running is everything the five builders wrote.
"""
from __future__ import annotations

import importlib
import math
import shutil
import time
import traceback
from pathlib import Path
from typing import Iterator

import numpy as np
import pytest
from fastapi.testclient import TestClient

from app import board as board_mod
from app import canon, config, creator, llm, render, store, stt, tts_client, voices, worker
from app.api import health as health_api
from app.api import remote as remote_api
from app.render import RenderResult
from app.tts_client import Sampler
from app.voices import Voice

REPO = Path(__file__).resolve().parents[2]
PHRASES_JSON = REPO / "data" / "phrases.json"
SR = 24_000
JOB_TIMEOUT_S = 20.0
IDLE_TIMEOUT_S = 90.0        # the boot pre-render of the whole bag bank, at fake-render speed

LABEL = "Krčmár Bruno"
VOICE_ID = "krcmar"
LANG = "sk"
DESCRIPTION = "Zhovorčivý krčmár, ktorý si pamätá každý dlh."
HEARD = "Vitaj,  pocestný, sadni si k ohňu."      # whisper's spacing, normalised by the creator
CATEGORIES = ["Vítanie hostí", "Účet za pivo", "Nalej ešte", "Toto ti nenalejem"]
PHRASES = {
    "Vítanie hostí": ["Vitaj, pocestný.", "Sadni si k ohňu.", "Dnes máme čerstvé pivo."],
    "Účet za pivo": ["To robí tri strieborné.", "Platíš ty, alebo tá elfka?", "Účet rastie, kamoš."],
    "Nalej ešte": ["Ešte jedno?", "Nalejem, ale posledné.", "Máš na to ešte hrdlo?"],
    "Toto ti nenalejem": ["Toto ti nenalejem.", "Dnes už nie.", "Choď domov, chlape."],
}
LINE_COUNT = sum(len(lines) for lines in PHRASES.values())
CLIP_HZ = 150.0                                   # -> f0 band [98, 225] through the contract's arithmetic
SAVED_TEXT = "Toto si zapíšem do knihy dlhov."
SAVED_TWICE = "Ešte raz to isté, kamoš."
DELETE_ME = "Túto si zase zmažem."

# What the stubs recorded, so a test can assert the order commit ran them in.
CALLS: list[str] = []
# The commit thread's traceback, if it had one (see ``recording_commit``).
COMMIT_ERROR: list[str] = []
_REAL_COMMIT = creator.commit


def sine(seconds: float, hz: float = CLIP_HZ) -> bytes:
    """int16 mono PCM of a pure tone: enough for pitch, length and file writing."""
    n = max(1, int(seconds * SR))
    t = np.arange(n, dtype=np.float32) / SR
    return (0.3 * np.sin(2 * math.pi * hz * t) * 32767).astype(np.int16).tobytes()


def make_bag() -> Voice:
    """A locked Bag (``ref_sha256`` set) so boot pre-renders its bank, exactly as
    the M2 smoke builds it."""
    return Voice(id="bag", label="Mr. Bag", lang="sk", version=1, ref_file="ref.wav", ref_sha256="0" * 64,
                 ref_transcript="Popravia? Dostane tretí obed.", ref_tts_path="/refs/bag/ref.wav",
                 sampler=Sampler(), golden_seed=0, gate=dict(voices.UNCALIBRATED_GATE), master={"energy": 65})


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


def fake_persona(label: str, description: str, lang: str = "sk") -> dict:
    """The Writer's step 3 answer. The last category is the signature refusal,
    which is the order ``commit`` relies on."""
    CALLS.append("persona")
    return {"sk": "Krčmár, ktorý si pamätá každý dlh.", "en": "An innkeeper who remembers every debt.",
            "label": label, "description": description, "categories": list(CATEGORIES)}


def fake_phrases(persona: dict, categories: list[str], lang: str = "sk", per_category: int = 10) -> dict:
    CALLS.append("phrases")
    return {name: list(PHRASES.get(name, [])) for name in categories}


def recording_commit(draft_id: str, **kw) -> dict:
    """``creator.commit`` with its exception kept.

    ``app/api/creator.py`` runs the commit on a daemon thread that logs whatever
    goes wrong and returns, because there is no request left to answer. That is
    right for the service and useless for a test: without this the only symptom
    would be ``status: failed`` and no reason at all.
    """
    try:
        return _REAL_COMMIT(draft_id, **kw)
    except BaseException as exc:                        # noqa: BLE001 - recorded, then re-raised untouched
        COMMIT_ERROR.append("".join(traceback.format_exception(exc)))
        raise


def fake_calibrate(v: Voice, lines: list[str], n_takes: int = 3) -> Voice:
    """``voices.calibrate`` without the 90 takes: the numbers are plausible and,
    more to the point, written to voice.yaml the way the real one writes them,
    so the roster and the final card see a calibrated voice."""
    CALLS.append("calibrate")
    v.gate.update(baseline=0.93, p10=0.88, strict=0.89, loose=0.85,
                  calibrated_at="2026-09-04T10:00:00+00:00")
    v.master["gain_db"] = -1.5
    voices.save_voice(v)
    return v


@pytest.fixture(scope="module")
def client(tmp_path_factory) -> Iterator[TestClient]:
    """One booted app for the module. ``BAG_DATA`` is set and ``config`` reloaded
    before ``app.main`` is imported, as the container does."""
    data = tmp_path_factory.mktemp("m4smoke")
    mp = pytest.MonkeyPatch()
    mp.setenv("BAG_DATA", str(data))
    mp.delenv("BAG_REMOTE_KEY", raising=False)
    importlib.reload(config)
    shutil.copyfile(PHRASES_JSON, data / "phrases.json")
    voices.save_voice(make_bag())

    mp.setattr(render, "render_line", fake_render_line)
    mp.setattr(worker, "render_line", fake_render_line)     # bound by ``from app.render import render_line``
    mp.setattr(tts_client, "health", lambda: True)
    mp.setattr(health_api, "stt_up", lambda: True)
    mp.setattr(health_api, "llm_residency", lambda: "resident")
    mp.setattr(remote_api, "REMOTE_KEY", "bag")
    # The Creator's four edges: whisper, the Writer's two authoring calls, the
    # residency probe behind them, and the two ffmpeg conversions.
    mp.setattr(llm, "residency", lambda: "resident")
    mp.setattr(llm, "persona", fake_persona)
    mp.setattr(llm, "phrases", fake_phrases)
    mp.setattr(stt, "transcribe", lambda pcm, sr, lang, timeout=60: HEARD)
    mp.setattr(creator, "_convert", lambda src, start_s, dur_s: sine(dur_s))
    mp.setattr(creator, "_pyin", lambda y, sr: (np.full(max(1, y.size // 512), CLIP_HZ),
                                                np.ones(max(1, y.size // 512), dtype=bool)))
    mp.setattr(voices, "_convert_reference", lambda wav_in: sine(5.0))
    mp.setattr(voices, "calibrate", fake_calibrate)
    mp.setattr(creator, "commit", recording_commit)
    CALLS.clear()
    COMMIT_ERROR.clear()
    from app.main import create_app

    with TestClient(create_app()) as c:
        yield c
    mp.undo()
    importlib.reload(config)


def wait_until(check, timeout_s: float, what: str):
    """Poll until ``check`` returns something truthy; the failure names what was
    being waited for, because a hang in a worker thread is otherwise mute."""
    deadline = time.monotonic() + timeout_s
    last = None
    while time.monotonic() < deadline:
        last = check()
        if last:
            return last
        time.sleep(0.05)
    raise AssertionError(f"timed out waiting for {what}; last saw {last!r}")


def wait_done(client: TestClient, job_id: str) -> dict:
    job = wait_until(lambda: (lambda j: j if j.get("status") in ("done", "failed", "cancelled") else None)(
        client.get(f"/api/jobs/{job_id}").json()), JOB_TIMEOUT_S, f"job {job_id}")
    assert job["status"] == "done", f"job never finished cleanly: {job}"
    return job


def wait_idle(client: TestClient) -> None:
    """Let the boot pre-render drain. The Creator's commit rebuilds the ``lines``
    table (``creator.ensure_source``), and a quiet queue keeps that from racing
    the worker's own writes for no reason a test would learn anything from."""
    wait_until(lambda: client.get("/readyz").json()["queue_depth"] == 0, IDLE_TIMEOUT_S, "an idle queue")


def board_of(client: TestClient, voice: str = "bag", lang: str = LANG) -> dict:
    r = client.get("/api/board", params={"voice": voice, "lang": lang})
    assert r.status_code == 200, r.text
    return r.json()


def tile(client: TestClient, line_id: str, voice: str = "bag") -> dict:
    found = next((line for line in board_of(client, voice)["lines"] if line["id"] == line_id), None)
    assert found is not None, f"line {line_id} is not on the {voice} board"
    return found


def save_line(client: TestClient, text: str, voice: str = "bag") -> dict:
    """What the improv bar's star does: no category, so the board defaults it."""
    r = client.post("/api/lines", json={"voice": voice, "lang": LANG, "text": text})
    assert r.status_code == 200, r.text
    return r.json()


# -- the Creator is mounted at all -------------------------------------------


def test_creator_router_and_page_are_mounted(client: TestClient) -> None:
    """Without ``creator`` in ``PEER_ROUTERS`` and the ``/creator`` route, every
    assertion below would be a 404 and the whole milestone invisible to the DM."""
    page = client.get("/creator")
    assert page.status_code == 200
    assert page.headers["content-type"].startswith("text/html")
    assert "/static/" in page.text
    assert client.get("/static/creator.js").status_code == 200
    drafts = client.get("/api/creator/drafts")
    assert drafts.status_code == 200 and isinstance(drafts.json(), list)


# -- the walk from a file to a table-ready voice ------------------------------


@pytest.fixture(scope="module")
def committed(client: TestClient) -> dict:
    """Walk one draft the way the Creator page does, recording the status after
    every step, and commit it. Returns the trail plus the finished draft."""
    upload = store.wav_bytes(sine(40.0), SR)
    r = client.post("/api/creator/drafts",
                    files={"file": ("bruno.wav", upload, "audio/wav")},
                    data={"label": LABEL, "lang": LANG})
    assert r.status_code == 200, r.text
    draft = r.json()
    trail = [("create", draft["status"])]
    did = draft["id"]

    steps = [
        ("clip", lambda: client.post(f"/api/creator/{did}/clip", json={"start_s": 0.0, "end_s": 45.0})),
        ("transcribe", lambda: client.post(f"/api/creator/{did}/transcribe")),
        ("describe", lambda: client.patch(f"/api/creator/{did}",
                                          json={"transcript": HEARD, "voice_id": VOICE_ID,
                                                "description": DESCRIPTION})),
        ("phrases", lambda: client.post(f"/api/creator/{did}/phrases", json={"per_category": 3})),
    ]
    seen = {"create": draft}
    for name, call in steps:
        resp = call()
        assert resp.status_code == 200, f"{name}: {resp.text}"
        draft = resp.json()
        seen[name] = draft
        trail.append((name, draft["status"]))

    wait_idle(client)                       # the bag bank is pre-rendered; keep the commit's own writes clean
    started = client.post(f"/api/creator/{did}/commit")
    assert started.status_code == 200, started.text
    assert started.json() == {"job_id": did}
    # Wait on the roster, not on the draft file. The commit thread rewrites
    # draft.json at every step, and on Windows an ``os.replace`` over a file a
    # reader has open raises PermissionError -- so polling GET /api/creator/{id}
    # here would make the test itself the thing that fails the commit it is
    # watching. The voice appears in ``app.state.voices`` only once ``commit``
    # has returned, which is the same "finished" this needs and touches no file.
    wait_until(lambda: COMMIT_ERROR or VOICE_ID in {v["id"] for v in client.get("/api/voices").json()},
               JOB_TIMEOUT_S, "the commit to finish")
    assert not COMMIT_ERROR, COMMIT_ERROR[0]
    return {"id": did, "trail": trail, "steps": seen,
            "draft": client.get(f"/api/creator/{did}").json()}


def test_the_draft_walks_the_contract_status_ladder(client: TestClient, committed: dict) -> None:
    assert committed["trail"] == [("create", "new"), ("clip", "clipped"), ("transcribe", "transcribed"),
                                  ("describe", "described"), ("phrases", "described")]
    clipped = committed["steps"]["clip"]
    assert clipped["end_s"] == creator.CLIP_MAX_S          # 45 s asked for, 30 s locked
    assert clipped["f0_band"] == [98, 225]                 # measured from the clip, not the default band
    assert committed["steps"]["transcribe"]["transcript"] == "Vitaj, pocestný, sadni si k ohňu."
    described = committed["steps"]["describe"]
    assert described["categories"] == CATEGORIES and described["persona"]["sk"]
    assert described["signature_category"] == CATEGORIES[-1]
    phrased = committed["steps"]["phrases"]
    assert phrased["phrases"] == PHRASES and phrased["thin"] == []
    assert client.get(f"/api/creator/{committed['id']}/clip.wav").status_code == 200


def test_commit_locks_calibrates_banks_and_queues_the_prerender(client: TestClient, committed: dict) -> None:
    draft = committed["draft"]
    assert draft["status"] == "done", COMMIT_ERROR or draft
    result = draft["result"]
    assert result["voice_id"] == VOICE_ID
    assert result["lines"] == LINE_COUNT and result["queued"] == LINE_COUNT
    assert result["gate"]["calibrated_at"] and result["gain_db"] == -1.5
    # The Writer ran before the GPU did, and the lock before the calibration.
    assert CALLS == ["persona", "phrases", "calibrate"]

    v = client.get(f"/api/voices/{VOICE_ID}")
    assert v.status_code == 200, "the commit must add the voice to the roster without a restart"
    assert v.json()["ref_sha256"], "the clip is the voice's locked reference"
    roster = {row["id"]: row for row in client.get("/api/voices").json()}
    assert roster[VOICE_ID]["locked"] is True and roster[VOICE_ID]["calibrated"] is True
    assert roster[VOICE_ID]["label"] == LABEL

    board = board_of(client, VOICE_ID)
    assert board["categories"] == CATEGORIES
    assert len(board["lines"]) == LINE_COUNT
    # The last category is the signature refusal: its first line owns the T tile.
    assert tile(client, board["ten_nie"], VOICE_ID)["category"] == CATEGORIES[-1]
    # One favourite per ordinary category, in order, and never the signature one.
    assert board["favourites"][:3] == [line["id"] for line in board["lines"] if line["slot"]]
    assert board["favourites"][3:] == [None] * 5


def test_the_new_voices_bank_renders_itself_after_the_commit(client: TestClient, committed: dict) -> None:
    """The queued pre-render is a real job list on the real worker, so the tiles
    fill in on their own -- which is what the DM walks back to Play to find."""
    wait_idle(client)
    statuses = {line["status"] for line in board_of(client, VOICE_ID)["lines"]}
    assert statuses == {"ready"}


# -- saving a phrase (contract §2) -------------------------------------------


def test_a_saved_phrase_is_one_tile_in_the_saved_category(client: TestClient) -> None:
    saved = save_line(client, SAVED_TEXT)
    assert saved["category"] == board_mod.SAVED_CATEGORY[LANG] == "Moje"
    assert tile(client, saved["id"])["text"] == SAVED_TEXT


def test_saving_the_same_phrase_twice_yields_one_tile(client: TestClient) -> None:
    first = save_line(client, SAVED_TWICE)
    again = save_line(client, SAVED_TWICE)
    assert again["id"] == first["id"]
    lines = [line for line in board_of(client)["lines"] if line["text"] == SAVED_TWICE]
    assert len(lines) == 1, "a second save must return the tile that already exists"


def test_delete_removes_a_saved_tile_and_refuses_a_bank_line(client: TestClient) -> None:
    board = board_of(client)
    bank_line = next(line for line in board["lines"] if line["category"] == board["categories"][0])
    assert client.delete(f"/api/lines/{bank_line['id']}").status_code == 409
    assert tile(client, bank_line["id"])["id"] == bank_line["id"], "a refused delete changes nothing"

    mine = save_line(client, DELETE_ME)
    assert client.delete(f"/api/lines/{mine['id']}").status_code == 200
    assert all(line["id"] != mine["id"] for line in board_of(client)["lines"])
    assert client.delete(f"/api/lines/{mine['id']}").status_code == 404


# -- a rendered tile stays lit (contract §1) ---------------------------------


def test_say_with_a_line_id_leaves_the_tile_ready_on_the_next_board_read(client: TestClient) -> None:
    """The bug §1 names: without ``line_id`` the render is never adopted, so the
    tile lights up from local state and is grey again on the next ``/api/board``.
    A line saved after boot is in no pre-render plan, so nothing but this ``say``
    can have rendered it."""
    mine = save_line(client, "Tento riadok si vypočujem hneď teraz.")
    assert tile(client, mine["id"])["status"] == "pending"
    assert tile(client, mine["id"])["render_id"] is None

    said = client.post("/api/say", json={"voice": "bag", "text": mine["text"], "lang": LANG,
                                         "priority": "live", "take_no": 0, "line_id": mine["id"]})
    assert said.status_code == 200, said.text
    body = said.json()
    wait_done(client, body["job_id"])

    after = tile(client, mine["id"])
    assert after["status"] == "ready", "the board itself must agree the tile is lit"
    assert after["render_id"] == body["render_id"], "the line adopted this render"
    assert client.get(f"/api/renders/{body['render_id']}.wav").status_code == 200
