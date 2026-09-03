"""The store is where a render becomes a served file: these tests pin the
schema, the idempotent bank ids that let phrases.json be re-imported without
losing pins, and the render row/file round trip. Everything runs against a
temp DATA_DIR selected through the BAG_DATA env, as in the container."""
from __future__ import annotations

import hashlib
import importlib
import wave
from contextlib import closing
from pathlib import Path

import pytest

from app import canon, config, render, store
from app.gate import TakeScore
from app.render import RenderResult

SR = 24_000
TABLES = {"voices", "lines", "renders", "jobs", "scenes", "playlists", "session", "speaker"}


@pytest.fixture
def data_dir(tmp_path, monkeypatch):
    monkeypatch.setenv("BAG_DATA", str(tmp_path))
    importlib.reload(config)
    store.init_db()
    yield tmp_path
    monkeypatch.undo()
    importlib.reload(config)


def make_result(text: str = "Ten nie.", take_no: int = 0) -> RenderResult:
    c = canon.canonicalize(text)
    pcm = bytes(SR * 2)          # 1 s of silence is enough for a file round trip
    score = TakeScore(seed=7, sim=0.93, sane=True, reason="ok")
    return RenderResult(render_id=hashlib.sha256(f"{text}|{take_no}".encode()).hexdigest(), voice_id="bag",
                        text=text, canon=c, seed=7, take_no=take_no, raw_pcm=pcm, pcm=pcm, sr=SR, sim=0.93,
                        cer=None, verified=False, gate="pass", scores=[score],
                        timings={"total": 1.2, "rounds": 1}, voice_version=1,
                        recipe_version=render.RECIPE_VERSION, master_version="m" * 64, lufs=-18.4)


def test_init_db_creates_the_schema_in_wal_mode(data_dir):
    assert (data_dir / "app.db").exists()
    with closing(store.db()) as con:
        names = {row["name"] for row in con.execute("SELECT name FROM sqlite_master WHERE type='table'")}
        mode = con.execute("PRAGMA journal_mode").fetchone()[0]
    assert TABLES <= names
    assert mode == "wal"


def test_upsert_line_is_idempotent_and_keeps_the_pin(data_dir):
    args = ("bag", "sk", "Urážka partie", "Vy nie ste družina, vy ste kolektívna diagnóza.", "bank")
    lid = store.upsert_line(*args)
    assert lid == hashlib.sha1("sk|bag|Urážka partie|Vy nie ste družina, vy ste kolektívna diagnóza.".encode()).hexdigest()
    store.set_active_render(lid, "r1")
    assert store.upsert_line(*args) == lid
    with closing(store.db()) as con:
        rows = con.execute("SELECT active_render_id, source FROM lines WHERE id=?", (lid,)).fetchall()
    assert len(rows) == 1 and rows[0]["active_render_id"] == "r1" and rows[0]["source"] == "bank"


def test_render_files_and_row_round_trip(data_dir):
    r = make_result()
    raw_path, path = store.write_render_files(r)
    assert Path(raw_path).name == f"{r.render_id}.raw.wav"
    assert Path(path).name == f"{r.render_id}.{render.master.VERSION}.wav"
    assert Path(path).parent == config.renders_dir("bag")
    with wave.open(path, "rb") as w:
        assert (w.getnchannels(), w.getsampwidth(), w.getframerate(), w.getnframes()) == (1, 2, SR, SR)

    lid = store.upsert_line("bag", "sk", "Ten nie.", "Ten nie.", "bank")
    store.put_render(r, lid, raw_path, path)
    store.set_active_render(lid, r.render_id)
    row = store.get_render(r.render_id)
    assert row is not None
    assert (row["line_id"], row["voice_id"], row["voice_version"], row["take_no"], row["seed"]) == (lid, "bag", 1, 0, 7)
    assert (row["recipe_version"], row["master_version"]) == (render.RECIPE_VERSION, "m" * 64)
    assert row["sim"] == 0.93 and row["cer"] is None and row["verified"] is False and row["gate"] == "pass"
    assert row["dur_s"] == 1.0 and row["lufs"] == -18.4
    assert (row["raw_path"], row["path"]) == (raw_path, path)

    store.put_render(r, lid, raw_path, path)     # same id again replaces, never duplicates
    with closing(store.db()) as con:
        assert con.execute("SELECT COUNT(*) FROM renders").fetchone()[0] == 1
        assert con.execute("SELECT active_render_id FROM lines WHERE id=?", (lid,)).fetchone()[0] == r.render_id


def test_get_render_missing_is_none(data_dir):
    assert store.get_render("nope") is None
