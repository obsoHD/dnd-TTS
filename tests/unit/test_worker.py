"""The worker is the table's promise that a line arrives in order and nothing
runs twice. These tests drive it with a fake ``render_line`` (no TTS, no GPU,
no network) and pin the scheduling rules of docs/M2-contracts.md: live before
prep before batch, cache short-circuit, cancel, re-queue on restart, live
preemption of a running batch job, and the event sequence the UI relies on.
The cancel kwarg is checked at both ends (``tts_client.synth`` and
``render_line``), and the two routers are smoked with a TestClient."""
from __future__ import annotations

import importlib
import threading
from contextlib import closing

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from app import canon, config, jobs, render, store, tts_client, voices, worker
from app.api import jobs as jobs_api
from app.api import say as say_api
from app.jobs import new_job
from app.render import RenderResult
from app.tts_client import Cancelled, Sampler
from app.voices import Voice
from app.worker import RenderWorker

SR = 24_000
LINE = "Ten nie."
OTHER = "Vy nie ste družina, vy ste kolektívna diagnóza."
WAIT = 5.0


def make_voice(**overrides) -> Voice:
    base = dict(id="bag", label="Mr. Bag", lang="sk", version=1, ref_file="ref.wav", ref_sha256="0" * 64,
                ref_transcript="Popravia? Dostane tretí obed.", ref_tts_path="/refs/bag/ref.wav",
                sampler=Sampler(), golden_seed=0, gate=dict(voices.UNCALIBRATED_GATE), master={"energy": 65})
    return Voice(**{**base, **overrides})


def make_result(voice: Voice, text: str, take_no: int = 0, gate: str = "pass") -> RenderResult:
    """A finished take with the real cache key, so the store sees what a render would give it."""
    c = canon.canonicalize(text, lang=voice.lang)
    pcm = bytes(SR)          # half a second of silence: enough for a file and a duration
    return RenderResult(render_id=render.render_id(voice, c.text, take_no), voice_id=voice.id, text=text,
                        canon=c, seed=1, take_no=take_no, raw_pcm=pcm, pcm=pcm, sr=SR, sim=0.91, cer=0.04,
                        verified=True, gate=gate, scores=[], timings={}, voice_version=voice.version,
                        recipe_version=render.RECIPE_VERSION, master_version="m" * 64, lufs=-18.0)


class FakeRender:
    """Stands in for ``render_line``: records calls, can hold a line open until
    released, honours ``cancel`` while holding, and can be told to blow up."""

    def __init__(self) -> None:
        self.calls: list[tuple[str, int]] = []
        self.hold: set[str] = set()
        self.fail: set[str] = set()
        self.entered = threading.Event()
        self.release = threading.Event()

    def __call__(self, voice, text, take_no=0, n_takes=2, *, cancel=None) -> RenderResult:
        self.calls.append((text, take_no))
        if text in self.fail:
            raise RuntimeError("tts exploded")
        if text in self.hold:
            self.entered.set()
            while not self.release.wait(0.01):
                tts_client.check_cancel(cancel)
        return make_result(voice, text, take_no)


@pytest.fixture
def data_dir(tmp_path, monkeypatch):
    """Temp DATA_DIR by env, the way the container does, with a voice.yaml so
    the worker loads a real Voice."""
    monkeypatch.setenv("BAG_DATA", str(tmp_path))
    importlib.reload(config)
    store.init_db()
    voices.save_voice(make_voice())
    yield tmp_path
    monkeypatch.undo()
    importlib.reload(config)


@pytest.fixture
def fake_render(monkeypatch) -> FakeRender:
    fake = FakeRender()
    monkeypatch.setattr(worker, "render_line", fake)
    return fake


@pytest.fixture
def events() -> list[tuple[str, dict]]:
    return []


@pytest.fixture
def make_worker(data_dir, fake_render, events):
    """Every worker is stopped at teardown so no thread outlives its temp dir."""
    made: list[RenderWorker] = []

    def factory() -> RenderWorker:
        w = RenderWorker(on_event=lambda name, data: events.append((name, data)))
        made.append(w)
        return w

    yield factory
    for w in made:
        w.stop(timeout=WAIT)


def names(events, job_id: str) -> list[str]:
    return [name for name, data in events if name.startswith("job.") and data.get("job_id") == job_id]


def index_of(events, name: str, job_id: str, nth: int = 0) -> int:
    return [i for i, (n, d) in enumerate(events) if n == name and d.get("job_id") == job_id][nth]


def store_render(text: str, gate: str = "pass") -> str:
    r = make_result(voices.load_voice("bag"), text, gate=gate)
    raw, path = store.write_render_files(r)
    store.put_render(r, None, raw, path)
    return r.render_id


# -- jobs.py -------------------------------------------------------------------

def test_new_job_defaults_and_priority_validation():
    job = new_job("say", "live", "bag", LINE)
    assert len(job.id) == 32 and job.status == "queued" and job.take_no == 0 and job.line_id is None
    assert job.render_id is None and job.started is None and job.done is None and job.error is None
    with pytest.raises(ValueError):
        new_job("say", "urgent", "bag", LINE)


def test_ensure_columns_migrates_the_m1_table_idempotently(data_dir):
    with closing(store.db()) as con:
        before = {row["name"] for row in con.execute("PRAGMA table_info(jobs)")}
    assert not {"voice_id", "text", "take_no", "render_id"} & before
    jobs.ensure_columns()
    jobs.ensure_columns()
    with closing(store.db()) as con:
        after = {row["name"] for row in con.execute("PRAGMA table_info(jobs)")}
    assert {"voice_id", "text", "take_no", "render_id", "priority"} <= after


def test_job_row_round_trip(data_dir):
    jobs.ensure_columns()
    job = new_job("say", "prep", "bag", OTHER, line_id="l1", take_no=2)
    job.render_id, job.started = "r" * 64, job.created + 1
    jobs.save(job)
    assert jobs.load(job.id) == job
    assert jobs.load("nope") is None
    assert [j.id for j in jobs.pending()] == [job.id]


# -- scheduling ------------------------------------------------------------------

def test_priority_order_live_before_prep_before_batch(make_worker, fake_render):
    w = make_worker()
    batch1 = w.submit(new_job("bank", "batch", "bag", "Batch jedna."))
    batch2 = w.submit(new_job("bank", "batch", "bag", "Batch dva."))
    prep = w.submit(new_job("script", "prep", "bag", "Prep."))
    live = w.submit(new_job("say", "live", "bag", LINE))
    snapshot = w.queue_snapshot()
    assert [e["id"] for e in snapshot] == [live.id, prep.id, batch1.id, batch2.id]
    assert [e["position"] for e in snapshot] == [1, 2, 3, 4]
    assert w.position(batch2.id) == 4
    w.start()
    for job in (batch1, batch2, prep, live):
        assert w.wait(job.id, WAIT).status == "done"
    assert [text for text, _ in fake_render.calls] == [LINE, "Prep.", "Batch jedna.", "Batch dva."]
    assert w.queue_snapshot() == [] and w.position(live.id) == 0


def test_cache_short_circuit_skips_the_tts(make_worker, fake_render, events):
    rid = store_render(LINE)
    w = make_worker()
    w.start()
    job = w.wait(w.submit(new_job("say", "live", "bag", LINE)).id, WAIT)
    assert job.status == "done" and job.render_id == rid
    assert fake_render.calls == []
    assert names(events, job.id) == ["job.queued", "job.started", "job.done"]
    done = events[index_of(events, "job.done", job.id)][1]
    assert done["cached"] is True and done["render_id"] == rid and done["gate"] == "pass"
    assert (done["sim"], done["cer"], done["verified"], done["dur_s"]) == (0.91, 0.04, True, 0.5)


def test_failed_gate_row_does_not_short_circuit(make_worker, fake_render, events):
    store_render(OTHER, gate="failed")
    w = make_worker()
    w.start()
    job = w.wait(w.submit(new_job("say", "live", "bag", OTHER)).id, WAIT)
    assert job.status == "done" and fake_render.calls == [(OTHER, 0)]
    assert events[index_of(events, "job.done", job.id)][1]["cached"] is False
    assert store.get_render(job.render_id)["gate"] == "pass"


def test_cancel_queued_job(make_worker, fake_render, events):
    w = make_worker()
    first = w.submit(new_job("bank", "batch", "bag", LINE))
    second = w.submit(new_job("bank", "batch", "bag", OTHER))
    assert w.cancel(second.id) is True
    assert second.status == "cancelled" and second.done is not None
    assert jobs.load(second.id).status == "cancelled"
    assert [e["id"] for e in w.queue_snapshot()] == [first.id]
    assert names(events, second.id) == ["job.queued", "job.cancelled"]
    assert w.cancel(second.id) is False and w.cancel("nope") is False
    w.start()
    assert w.wait(first.id, WAIT).status == "done"
    assert w.wait(second.id, WAIT).status == "cancelled"
    assert fake_render.calls == [(LINE, 0)]
    assert w.cancel(first.id) is False


def test_cancel_running_job_drops_it(make_worker, fake_render, events):
    fake_render.hold.add(LINE)
    w = make_worker()
    w.start()
    job = w.submit(new_job("say", "live", "bag", LINE))
    assert fake_render.entered.wait(WAIT)
    assert w.cancel(job.id) is True
    assert w.wait(job.id, WAIT).status == "cancelled"
    assert names(events, job.id) == ["job.queued", "job.started", "job.progress", "job.cancelled"]
    assert fake_render.calls == [(LINE, 0)]


def test_restart_requeues_queued_and_running_jobs(make_worker, fake_render):
    crashed = make_worker()
    a = crashed.submit(new_job("bank", "batch", "bag", LINE))
    b = crashed.submit(new_job("bank", "batch", "bag", OTHER))
    a.status, a.started = "running", a.created + 1        # what a crash mid-render leaves behind
    jobs.save(a)
    fresh = make_worker()
    fresh.start()
    assert fresh.wait(a.id, WAIT).status == "done"
    assert fresh.wait(b.id, WAIT).status == "done"
    assert [text for text, _ in fake_render.calls] == [LINE, OTHER]
    assert fresh.queue_snapshot() == [] and jobs.pending() == []


def test_event_sequence_queued_started_done(make_worker, fake_render, events):
    w = make_worker()
    w.start()
    job = w.submit(new_job("say", "live", "bag", LINE, line_id="l1"))
    assert w.wait(job.id, WAIT).status == "done"
    assert names(events, job.id) == ["job.queued", "job.started", "job.progress", "job.progress", "job.done"]
    queued = events[index_of(events, "job.queued", job.id)][1]
    assert queued["position"] == 1 and queued["priority"] == "live" and queued["line_id"] == "l1"
    assert [d["stage"] for n, d in events if n == "job.progress"] == ["render", "store"]
    done = events[index_of(events, "job.done", job.id)][1]
    assert set(done) >= {"job_id", "render_id", "cached", "sim", "cer", "verified", "gate", "dur_s"}
    assert done["render_id"] == job.render_id == store.get_render(job.render_id)["id"]
    changed = [d for n, d in events if n == "queue.changed"]
    assert changed[-1] == {"depth": 0, "running": None, "by_priority": {"live": 0, "prep": 0, "batch": 0}}


def test_live_job_preempts_a_running_batch_job(make_worker, fake_render, events):
    fake_render.hold.add(OTHER)
    w = make_worker()
    w.start()
    batch = w.submit(new_job("bank", "batch", "bag", OTHER))
    assert fake_render.entered.wait(WAIT)
    live = w.submit(new_job("say", "live", "bag", LINE))
    assert w.wait(live.id, WAIT).status == "done"
    fake_render.release.set()
    assert w.wait(batch.id, WAIT).status == "done"
    assert fake_render.calls == [(OTHER, 0), (LINE, 0), (OTHER, 0)]
    assert names(events, batch.id) == ["job.queued", "job.started", "job.progress",
                                       "job.queued", "job.started", "job.progress", "job.progress", "job.done"]
    assert index_of(events, "job.queued", batch.id, nth=1) < index_of(events, "job.started", live.id)
    assert index_of(events, "job.done", live.id) < index_of(events, "job.started", batch.id, nth=1)


def test_failed_render_marks_the_job_failed_and_keeps_the_loop_alive(make_worker, fake_render, events):
    fake_render.fail.add(OTHER)
    w = make_worker()
    w.start()
    bad = w.wait(w.submit(new_job("say", "live", "bag", OTHER)).id, WAIT)
    assert bad.status == "failed" and bad.error == "RuntimeError: tts exploded"
    assert names(events, bad.id) == ["job.queued", "job.started", "job.progress", "job.failed"]
    assert events[index_of(events, "job.failed", bad.id)][1]["error"] == bad.error
    assert jobs.load(bad.id).status == "failed"
    assert w.wait(w.submit(new_job("say", "live", "bag", LINE)).id, WAIT).status == "done"


def test_done_render_becomes_the_active_one_unless_pinned(make_worker, fake_render):
    fresh = store.upsert_line("bag", "sk", "Ten nie.", LINE, "bank")
    pinned = store.upsert_line("bag", "sk", "Urážka partie", OTHER, "bank")
    store.set_active_render(pinned, "pinned-render")
    w = make_worker()
    w.start()
    a = w.wait(w.submit(new_job("bank", "batch", "bag", LINE, line_id=fresh)).id, WAIT)
    b = w.wait(w.submit(new_job("regenerate", "live", "bag", OTHER, line_id=pinned, take_no=1)).id, WAIT)
    with closing(store.db()) as con:
        active = {r["id"]: r["active_render_id"] for r in con.execute("SELECT id, active_render_id FROM lines")}
    assert active[fresh] == a.render_id
    assert active[pinned] == "pinned-render"
    row = store.get_render(b.render_id)
    assert row["line_id"] == pinned and row["take_no"] == 1


def test_wait_returns_none_on_timeout_and_unknown_job(make_worker, fake_render):
    fake_render.hold.add(LINE)
    w = make_worker()
    w.start()
    job = w.submit(new_job("say", "live", "bag", LINE))
    assert w.wait(job.id, 0.05) is None
    assert w.wait("nope", 0.01) is None
    assert w.get(job.id) is job and w.get("nope") is None
    fake_render.release.set()
    assert w.wait(job.id, WAIT).status == "done"


def test_stop_requeues_the_running_job_for_the_next_start(make_worker, fake_render):
    fake_render.hold.add(LINE)
    w = make_worker()
    w.start()
    job = w.submit(new_job("bank", "batch", "bag", LINE))
    assert fake_render.entered.wait(WAIT)
    w.stop(timeout=WAIT)
    assert job.status == "queued" and jobs.load(job.id).status == "queued"
    fake_render.hold.clear()
    again = make_worker()
    again.start()
    assert again.wait(job.id, WAIT).status == "done"


# -- the cancel kwarg in M1 ----------------------------------------------------------

class FakeResponse:
    """Just enough of ``requests.Response`` for the streaming read loop; can set
    ``cancel`` after a given number of chunks, like a live job arriving mid-stream."""

    headers = {"x-sample-rate": str(SR)}

    def __init__(self, n_chunks: int, cancel: threading.Event | None = None, cancel_after: int = 0):
        self._n, self._cancel, self._after = n_chunks, cancel, cancel_after

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def raise_for_status(self) -> None:
        return None

    def iter_content(self, chunk_size: int):
        for i in range(self._n):
            if self._cancel is not None and i == self._after:
                self._cancel.set()
            yield b"\x00" * chunk_size


def test_synth_honours_cancel_and_stays_backward_compatible(monkeypatch):
    responses: list[FakeResponse] = []
    monkeypatch.setattr(tts_client.requests, "post", lambda *a, **k: responses.pop(0))
    args = ("Ten nie.", "/refs/bag/ref.wav", "ref", 1, Sampler(), 200)
    responses.append(FakeResponse(3))
    assert tts_client.synth(*args) == (b"\x00" * 3 * 4096, SR)
    cancel = threading.Event()
    responses.append(FakeResponse(3, cancel, cancel_after=1))
    with pytest.raises(Cancelled):
        tts_client.synth(*args, cancel=cancel)
    preset = threading.Event()
    preset.set()
    with pytest.raises(Cancelled):
        tts_client.synth_many("Ten nie.", "/refs/bag/ref.wav", "ref", [1, 2], Sampler(), 200, cancel=preset)
    assert responses == []


def test_render_line_checks_cancel_before_each_round(data_dir, monkeypatch):
    ref = config.voice_dir("bag") / "ref.wav"
    store.write_wav(ref, bytes(SR), SR)
    voice = make_voice(ref_sha256=voices.sha256_file(ref))
    monkeypatch.setattr(render, "_speaker_gate", lambda ref, sha: object())
    seen: list[threading.Event | None] = []

    def synth_many(*args, cancel=None, **kwargs):
        seen.append(cancel)
        raise Cancelled("mid-stream")

    monkeypatch.setattr(render.tts_client, "synth_many", synth_many)
    preset = threading.Event()
    preset.set()
    with pytest.raises(Cancelled):
        render.render_line(voice, LINE, cancel=preset)
    assert seen == []                                    # stopped before the first take was asked for
    live = threading.Event()
    with pytest.raises(Cancelled):
        render.render_line(voice, LINE, cancel=live)
    assert seen == [live]                                # and the event reaches the stream


# -- routers --------------------------------------------------------------------------

@pytest.fixture
def client(make_worker):
    app = FastAPI()
    app.include_router(say_api.router)
    app.include_router(jobs_api.router)
    app.state.worker = make_worker()
    app.state.worker.start()
    return TestClient(app)


def test_api_say_returns_the_job_without_blocking_then_cached(client, fake_render):
    fake_render.hold.add(LINE)
    first = client.post("/api/say", json={"voice": "bag", "text": LINE, "lang": "sk"})
    assert first.status_code == 200
    body = first.json()
    assert set(body) == {"job_id", "render_id", "cached", "position"}
    expected = render.render_id(voices.load_voice("bag"), canon.canonicalize(LINE).text, 0)
    assert body["cached"] is False and body["render_id"] == expected and body["position"] in (0, 1)
    fake_render.release.set()
    w = client.app.state.worker
    assert w.wait(body["job_id"], WAIT).status == "done"
    again = client.post("/api/say", json={"voice": "bag", "text": LINE}).json()
    assert again["cached"] is True and again["render_id"] == expected and again["job_id"] != body["job_id"]
    assert w.wait(again["job_id"], WAIT).status == "done"
    assert fake_render.calls == [(LINE, 0)]


def test_api_say_rejects_bad_input(client):
    assert client.post("/api/say", json={"voice": "ghost", "text": LINE}).status_code == 404
    assert client.post("/api/say", json={"voice": "bag", "text": "x" * 300}).status_code == 400
    assert client.post("/api/say", json={"voice": "bag", "text": "   "}).status_code == 400
    assert client.post("/api/say", json={"voice": "bag", "text": LINE, "priority": "urgent"}).status_code == 422


def test_api_jobs_queue_and_cancel(client, fake_render):
    fake_render.hold.add(LINE)
    running = client.post("/api/say", json={"voice": "bag", "text": LINE}).json()
    assert fake_render.entered.wait(WAIT)
    queued = client.post("/api/say", json={"voice": "bag", "text": OTHER, "priority": "batch"}).json()
    assert queued["position"] == 1
    q = client.get("/api/queue").json()
    assert [(e["id"], e["status"], e["position"]) for e in q] == [
        (running["job_id"], "running", 0), (queued["job_id"], "queued", 1)]
    assert set(q[0]) >= {"id", "kind", "priority", "voice_id", "text", "status", "position"}
    r = client.post(f"/api/jobs/{queued['job_id']}/cancel")
    assert r.status_code == 200 and r.json()["cancelled"] is True and r.json()["job"]["status"] == "cancelled"
    assert client.get(f"/api/jobs/{queued['job_id']}").json()["status"] == "cancelled"
    assert client.get("/api/jobs/nope").status_code == 404
    assert client.post("/api/jobs/nope/cancel").status_code == 404
    fake_render.release.set()
    w = client.app.state.worker
    assert w.wait(running["job_id"], WAIT).status == "done"
    got = client.get(f"/api/jobs/{running['job_id']}").json()
    assert got["status"] == "done" and got["render_id"] == running["render_id"]
    assert client.post(f"/api/jobs/{running['job_id']}/cancel").json()["cancelled"] is False
