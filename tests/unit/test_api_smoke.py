"""End-to-end smoke of the assembled M2 service: the real ``app.main`` boot
sequence (init_db, bank import, voices, hub, player, worker, pre-render) with
only the edges faked. ``render_line`` returns a silent take so no TTS or GPU is
touched, and the three readiness probes are stubbed so nothing on the network
is ever contacted. The point is not to re-test each module (their own suites
do that) but to prove the modules five builders wrote actually fit together
behind one ``TestClient``: the contract's routes exist, the worker thread
finishes a job the API submitted, and the file it wrote is served back."""
from __future__ import annotations

import importlib
import json
import shutil
import time
from pathlib import Path
from typing import Iterator

import pytest
from fastapi.testclient import TestClient

from app import canon, config, render, tts_client, voices, worker
from app.api import health as health_api
from app.api import remote as remote_api
from app.render import RenderResult
from app.tts_client import Sampler
from app.voices import Voice

REPO = Path(__file__).resolve().parents[2]
PHRASES = REPO / "data" / "phrases.json"
SR = 24_000
JOB_TIMEOUT_S = 5.0
READYZ_KEYS = {"tts", "tts_warm", "stt", "llm", "speaker", "bank_ready", "queue_depth"}
LINE_KEYS = {"id", "text", "category", "status", "render_id", "favourite", "slot"}
TEXT = "Vy nie ste družina, vy ste kolektívna diagnóza."
# Counted from the bank, never typed: the population pass grows these files and a
# hand-written total would go stale the next time a category gains a line.
BAG_SK_LINES = sum(len(texts) for texts in json.loads(PHRASES.read_text(encoding="utf-8"))["sk"]["bag"].values())


def make_voice() -> Voice:
    """A locked Bag (``ref_sha256`` set) so boot pre-renders its bank."""
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


@pytest.fixture(scope="module")
def client(tmp_path_factory) -> Iterator[TestClient]:
    """One booted app for the module: the bank import costs seconds and every
    test here only reads or leaves the state it found. ``BAG_DATA`` is set and
    ``config`` reloaded before ``app.main`` is imported, as the container does."""
    data = tmp_path_factory.mktemp("smoke")
    mp = pytest.MonkeyPatch()
    mp.setenv("BAG_DATA", str(data))
    mp.delenv("BAG_REMOTE_KEY", raising=False)
    importlib.reload(config)
    shutil.copyfile(PHRASES, data / "phrases.json")
    voices.save_voice(make_voice())
    mp.setattr(render, "render_line", fake_render_line)
    mp.setattr(worker, "render_line", fake_render_line)       # bound by ``from app.render import render_line``
    mp.setattr(tts_client, "health", lambda: True)
    mp.setattr(health_api, "stt_up", lambda: True)
    mp.setattr(health_api, "llm_residency", lambda: "resident")
    mp.setattr(remote_api, "REMOTE_KEY", "bag")
    from app.main import create_app

    with TestClient(create_app()) as c:
        yield c
    mp.undo()
    importlib.reload(config)


def wait_done(client: TestClient, job_id: str) -> dict:
    """Poll the job until the worker thread settles it; the assertion names the
    last state seen so a hang is diagnosable."""
    deadline = time.monotonic() + JOB_TIMEOUT_S
    job = {}
    while time.monotonic() < deadline:
        job = client.get(f"/api/jobs/{job_id}").json()
        if job["status"] in ("done", "failed", "cancelled"):
            break
        time.sleep(0.05)
    assert job.get("status") == "done", f"job never finished: {job}"
    return job


def say(client: TestClient, text: str = TEXT) -> dict:
    r = client.post("/api/say", json={"voice": "bag", "text": text, "lang": "sk"})
    assert r.status_code == 200, r.text
    body = r.json()
    assert set(body) == {"job_id", "render_id", "cached", "position"}
    return body


def test_healthz(client: TestClient) -> None:
    r = client.get("/healthz")
    assert r.status_code == 200
    assert r.json() == {"ok": True}


def test_readyz_has_documented_keys(client: TestClient) -> None:
    r = client.get("/readyz")
    assert r.status_code == 200
    body = r.json()
    assert set(body) == READYZ_KEYS
    assert body["tts"] is True and body["stt"] is True and body["llm"] == "resident"
    assert body["speaker"] is False
    assert isinstance(body["tts_warm"], bool) and isinstance(body["bank_ready"], bool)
    assert isinstance(body["queue_depth"], int)


def test_board_after_import(client: TestClient) -> None:
    r = client.get("/api/board", params={"voice": "bag", "lang": "sk"})
    assert r.status_code == 200
    board = r.json()
    assert set(board) == {"categories", "lines", "favourites", "ten_nie"}
    assert len(board["categories"]) == 8 and board["categories"][-1] == "Ten nie."
    assert len(board["lines"]) == BAG_SK_LINES
    assert all(set(line) == LINE_KEYS for line in board["lines"])
    ids = {line["id"] for line in board["lines"]}
    assert len(board["favourites"]) == 8 and board["favourites"][0] in ids
    assert board["ten_nie"] in ids


def test_say_reaches_job_done_and_serves_wav(client: TestClient) -> None:
    first = say(client)
    job = wait_done(client, first["job_id"])
    assert job["render_id"] == first["render_id"]

    wav = client.get(f"/api/renders/{first['render_id']}.wav")
    assert wav.status_code == 200
    assert wav.headers["content-type"].startswith("audio/wav")
    assert "immutable" in wav.headers["cache-control"]
    assert wav.content[:4] == b"RIFF"

    row = client.get(f"/api/renders/{first['render_id']}")
    assert row.status_code == 200 and row.json()["id"] == first["render_id"]

    again = say(client)
    assert again["cached"] is True and again["render_id"] == first["render_id"]
    assert wait_done(client, again["job_id"])["render_id"] == first["render_id"]


def test_play_shows_in_player(client: TestClient) -> None:
    said = say(client, "Ten nie.")
    rid = said["render_id"]
    wait_done(client, said["job_id"])

    client.post("/api/stop")
    r = client.post("/api/play", json={"render_id": rid, "label": "Ten nie."})
    assert r.status_code == 200
    assert r.json()["position"] == 0 and r.json()["url"] == f"/api/renders/{rid}.wav"

    state = client.get("/api/player").json()
    assert set(state) == {"now", "queue", "last", "speaker"}
    assert state["now"]["render_id"] == rid and state["now"]["label"] == "Ten nie."

    client.post("/api/stop")
    assert client.get("/api/player").json()["now"] is None


def test_remote_key(client: TestClient) -> None:
    ok = client.get("/remote/stop", params={"k": "bag"})
    assert ok.status_code == 200 and ok.json()["action"] == "stop"
    assert client.get("/remote/stop", params={"k": "wrong"}).status_code == 403
    assert client.get("/remote/stop").status_code == 403


def test_pages_and_static(client: TestClient) -> None:
    for path in ("/", "/speaker"):
        r = client.get(path)
        assert r.status_code == 200, path
        assert r.headers["content-type"].startswith("text/html")
        assert "/static/" in r.text
    assert client.get("/static/app.js").status_code == 200


def test_ws_sends_status_first(client: TestClient) -> None:
    with client.websocket_connect("/ws?client=smoke&role=play") as ws:
        frame = ws.receive_json()
    assert frame["type"] == "status"
    assert READYZ_KEYS <= set(frame["data"])
    assert set(frame["data"]["player"]) == {"now", "queue", "last", "speaker"}
    assert isinstance(frame["data"]["queue"], list)
