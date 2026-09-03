"""The board is what the DM sees: these tests pin the bank import (every line,
once, re-importable without losing pins or favourites), the tile order and
categories, the default favourites row and the Ten nie. key, and the tile
state derived from the pinned render. The API layer is exercised through
FastAPI's TestClient with the worker replaced by a fake; no GPU, no network."""
from __future__ import annotations

import hashlib
import importlib
import json
import shutil
import sys
import types
from contextlib import closing
from pathlib import Path
from types import SimpleNamespace

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from app import board, canon, config, store
from app.api import board as board_api
from app.gate import TakeScore
from app.render import RenderResult

REPO = Path(__file__).resolve().parents[2]
PHRASES = REPO / "data" / "phrases.json"
SR = 24_000
BAG_SK = ["Pozdrav kámoša", "Urážka partie", "Chvastanie po záchrane", "Odmietnutie predmetu",
          "Bojový pokrik", "Sarkastická poznámka", "Namrzené povzbudenie", "Ten nie."]


@pytest.fixture
def data_dir(tmp_path, monkeypatch):
    """Point the whole app at a temp DATA_DIR by env, the way the container does."""
    monkeypatch.setenv("BAG_DATA", str(tmp_path))
    importlib.reload(config)
    yield tmp_path
    monkeypatch.undo()
    importlib.reload(config)


@pytest.fixture
def fresh_db(data_dir) -> Path:
    store.init_db()
    return data_dir


@pytest.fixture(scope="module")
def bank_template(tmp_path_factory) -> Path:
    """The bank imported once per module: 540 single-connection upserts cost
    seconds, so tests that only read the bank get a private copy of this file."""
    root = tmp_path_factory.mktemp("bank")
    with pytest.MonkeyPatch.context() as mp:
        mp.setenv("BAG_DATA", str(root))
        importlib.reload(config)
        store.init_db()
        board.import_bank(PHRASES)
        with closing(store.db()) as con:
            con.execute("PRAGMA wal_checkpoint(TRUNCATE)")
    importlib.reload(config)
    return root / "app.db"


@pytest.fixture
def bank(data_dir, bank_template) -> dict:
    shutil.copyfile(bank_template, data_dir / "app.db")
    return json.loads(PHRASES.read_text(encoding="utf-8"))


def make_result(text: str, take_no: int = 0, gate: str = "pass", verified: bool = True) -> RenderResult:
    c = canon.canonicalize(text)
    pcm = bytes(SR * 2)
    score = TakeScore(seed=7, sim=0.93, sane=True, reason="ok")
    return RenderResult(render_id=hashlib.sha256(f"{text}|{take_no}".encode()).hexdigest(), voice_id="bag",
                        text=text, canon=c, seed=7, take_no=take_no, raw_pcm=pcm, pcm=pcm, sr=SR, sim=0.93,
                        cer=None, verified=verified, gate=gate, scores=[score], timings={"total": 1.0},
                        voice_version=1, recipe_version="r" * 64, master_version="m" * 64, lufs=-18.0)


def pin_render(line_id: str, text: str, **kw) -> str:
    r = make_result(text, **kw)
    store.put_render(r, line_id, "raw.wav", "mastered.wav")
    store.set_active_render(line_id, r.render_id)
    return r.render_id


def row_count() -> int:
    with closing(store.db()) as con:
        return con.execute("SELECT COUNT(*) FROM lines").fetchone()[0]


def test_import_bank_counts_every_line_once_per_voice_and_is_idempotent(fresh_db):
    bank = json.loads(PHRASES.read_text(encoding="utf-8"))
    expected = sum(len(texts) * (2 if key == "npc" else 1)
                   for banks in bank.values() for key, cats in banks.items() for texts in cats.values())
    assert board.import_bank(PHRASES) == expected
    assert row_count() == expected
    assert board.import_bank(PHRASES) == expected
    assert row_count() == expected
    assert len(board.board("bag", "sk")["lines"]) == 80
    with closing(store.db()) as con:
        npc = con.execute("SELECT voice_id, COUNT(*) n FROM lines WHERE lang='sk' AND category='Pozdrav' "
                          "GROUP BY voice_id ORDER BY voice_id").fetchall()
        sources = {row[0] for row in con.execute("SELECT DISTINCT source FROM lines")}
    assert [(r["voice_id"], r["n"]) for r in npc] == [("female", 10), ("male", 10)]
    assert sources == {"bank"}


def test_reimport_keeps_pins_favourites_and_slots(bank):
    lines = board.board("bag", "sk")["lines"]
    moved, pinned = lines[-1], lines[5]
    board.set_line(moved["id"], slot=8)
    board.set_line(lines[0]["id"], favourite=False)
    render_id = pin_render(pinned["id"], pinned["text"])
    board.import_bank(PHRASES)
    after = {line["id"]: line for line in board.board("bag", "sk")["lines"]}
    assert after[moved["id"]]["slot"] == 8 and after[moved["id"]]["favourite"] is True
    assert after[lines[0]["id"]]["favourite"] is False and after[lines[0]["id"]]["slot"] is None
    assert after[pinned["id"]]["render_id"] == render_id


def test_categories_in_bank_order(bank):
    assert board.categories("bag", "sk") == BAG_SK
    assert board.categories("bag", "en") == list(bank["en"]["bag"])
    assert board.categories("shopkeep", "sk") == list(bank["sk"]["shopkeep"])
    assert board.categories("male", "sk") == board.categories("female", "sk") == list(bank["sk"]["npc"])
    assert board.categories("nobody", "sk") == []


def test_lines_keep_bank_order_within_categories(bank):
    texts = [line["text"] for line in board.board("bag", "sk")["lines"]]
    assert texts == [text for cat in BAG_SK for text in bank["sk"]["bag"][cat]]


def test_favourites_default_to_first_line_of_each_category_and_ten_nie_on_t(bank):
    b = board.board("bag", "sk")
    firsts = [store.line_id("bag", "sk", cat, bank["sk"]["bag"][cat][0]) for cat in BAG_SK]
    assert b["favourites"] == firsts[:7] + [None]
    assert b["ten_nie"] == firsts[7]
    by_id = {line["id"]: line for line in b["lines"]}
    assert [(by_id[i]["favourite"], by_id[i]["slot"]) for i in firsts[:7]] == [(True, n) for n in range(1, 8)]
    assert (by_id[firsts[7]]["favourite"], by_id[firsts[7]]["slot"]) == (False, None)
    assert sum(line["favourite"] for line in b["lines"]) == 7
    en = board.board("bag", "en")
    assert en["ten_nie"] == store.line_id("bag", "en", "Not that one.", bank["en"]["bag"]["Not that one."][0])
    assert en["favourites"][6] is not None and en["favourites"][7] is None


def test_favourites_default_for_other_voices_and_none_for_the_t_key(bank):
    npc = board.board("male", "sk")
    assert npc["favourites"][:6] == [store.line_id("male", "sk", cat, texts[0])
                                     for cat, texts in bank["sk"]["npc"].items()]
    assert npc["favourites"][6:] == [None, None] and npc["ten_nie"] is None


def test_status_derives_from_the_pinned_render(bank):
    lines = board.board("bag", "sk")["lines"]
    assert {line["status"] for line in lines} == {"pending"}
    ready, amber, red, dangling = lines[:4]
    r_ready = pin_render(ready["id"], ready["text"])
    pin_render(amber["id"], amber["text"], verified=False)
    pin_render(red["id"], red["text"], gate="failed")
    store.set_active_render(dangling["id"], "gone")
    by_id = {line["id"]: line for line in board.board("bag", "sk")["lines"]}
    assert (by_id[ready["id"]]["status"], by_id[ready["id"]]["render_id"]) == ("ready", r_ready)
    assert by_id[amber["id"]]["status"] == "unverified"
    assert by_id[red["id"]]["status"] == "gate-failed"
    assert (by_id[dangling["id"]]["status"], by_id[dangling["id"]]["render_id"]) == ("pending", None)
    assert board.get_line(ready["id"]) == by_id[ready["id"]]


def test_set_line_slot_evicts_holder_and_unfavourite_frees_the_slot(bank):
    b = board.board("bag", "sk")
    holder, newcomer, other = b["favourites"][0], b["lines"][20]["id"], b["favourites"][2]
    assert board.set_line(newcomer, slot=1)["slot"] == 1
    by_id = {line["id"]: line for line in board.board("bag", "sk")["lines"]}
    assert (by_id[holder]["favourite"], by_id[holder]["slot"]) == (True, None)
    assert (by_id[newcomer]["favourite"], by_id[newcomer]["slot"]) == (True, 1)
    assert board.set_line(newcomer, slot=0) == {**by_id[newcomer], "slot": None}
    assert board.set_line(other, favourite=False)["slot"] is None
    assert board.set_line(other, favourite=True)["slot"] is None
    assert board.set_line(other, slot=3, favourite=False) == {**by_id[other], "favourite": False, "slot": None}
    assert board.board("bag", "sk")["favourites"][:3] == [None, b["favourites"][1], None]


def test_set_line_category_and_errors(bank):
    line = board.board("bag", "sk")["lines"][15]
    assert board.set_line(line["id"], category="  Nová   kategória ")["category"] == "Nová kategória"
    # a category sits where its earliest line sits: row 15 is inside the second bank category
    assert board.categories("bag", "sk") == BAG_SK[:2] + ["Nová kategória"] + BAG_SK[2:]
    with pytest.raises(ValueError):
        board.set_line(line["id"], slot=9)
    with pytest.raises(ValueError):
        board.set_line(line["id"], category=" ")
    with pytest.raises(board.LineNotFound):
        board.set_line("nope", favourite=True)
    with pytest.raises(board.LineNotFound):
        board.get_line("nope")


def test_add_line_appends_an_improv_tile_and_refuses_what_cannot_render(bank):
    line = board.add_line("bag", "sk", "Urážka partie", "  Ty si   hviezda. ")
    assert line == {"id": store.line_id("bag", "sk", "Urážka partie", "Ty si hviezda."), "text": "Ty si hviezda.",
                    "category": "Urážka partie", "status": "pending", "render_id": None,
                    "favourite": False, "slot": None}
    assert board.add_line("bag", "sk", "Urážka partie", "Ty si hviezda.")["id"] == line["id"]
    b = board.board("bag", "sk")
    assert b["lines"][-1] == line and b["categories"] == BAG_SK
    new_cat = board.add_line("bag", "sk", "Vlastné", "Prvá vlastná.")
    assert board.categories("bag", "sk") == BAG_SK + ["Vlastné"]
    with closing(store.db()) as con:
        assert con.execute("SELECT source FROM lines WHERE id=?", (new_cat["id"],)).fetchone()[0] == "improv"
    with pytest.raises(canon.TooLong):
        board.add_line("bag", "sk", "Vlastné", "slovo " * 60)
    with pytest.raises(ValueError):
        board.add_line("bag", "sk", "Vlastné", "   ")


def test_prerender_plan_favourites_then_ten_nie_then_the_rest(bank):
    b = board.board("bag", "sk")
    starred = b["lines"][15]["id"]
    board.set_line(starred, favourite=True)
    board.set_line(b["lines"][31]["id"], slot=8)          # row 31 holds no default slot; row 30 does
    plan = board.prerender_plan("bag", "sk")
    favourites = board.board("bag", "sk")["favourites"]
    assert None not in favourites and plan[:8] == favourites
    assert plan[8:10] == [starred, b["ten_nie"]]
    assert len(plan) == len(set(plan)) == len(b["lines"])
    assert plan[10:] == [line["id"] for line in b["lines"] if line["id"] not in plan[:10]]


def test_next_take_follows_the_pinned_render(bank):
    line = board.board("bag", "sk")["lines"][0]
    assert board.next_take(line["id"]) == {"voice_id": "bag", "text": line["text"], "take_no": 0}
    pin_render(line["id"], line["text"], take_no=2)
    assert board.next_take(line["id"])["take_no"] == 3
    with pytest.raises(board.LineNotFound):
        board.next_take("nope")


def test_ensure_columns_migrates_a_pre_m2_lines_table(fresh_db):
    with closing(store.db()) as con, con:
        con.execute("ALTER TABLE lines DROP COLUMN favourite")
        con.execute("ALTER TABLE lines DROP COLUMN slot")
    board.ensure_columns()
    board.ensure_columns()
    with closing(store.db()) as con:
        cols = {row["name"]: row for row in con.execute("PRAGMA table_info(lines)")}
    assert cols["favourite"]["dflt_value"] == "0" and cols["favourite"]["notnull"] == 1
    assert cols["slot"]["type"] == "INTEGER"
    board.import_bank(PHRASES)
    assert board.board("bag", "sk")["favourites"][0] is not None


class FakeWorker:
    def __init__(self) -> None:
        self.submitted: list = []

    def submit(self, job):
        self.submitted.append(job)
        return job


@pytest.fixture
def client(bank, monkeypatch):
    """The board router alone, with the worker owner's ``app.jobs`` replaced by
    a recorder so the regenerate wiring is checked without a worker thread."""
    jobs = types.ModuleType("app.jobs")
    jobs.new_job = lambda kind, priority, voice_id, text, line_id=None, take_no=0: SimpleNamespace(
        id="job-1", kind=kind, priority=priority, voice_id=voice_id, text=text, line_id=line_id, take_no=take_no)
    monkeypatch.setitem(sys.modules, "app.jobs", jobs)
    api = FastAPI()
    api.include_router(board_api.router)
    api.state.worker = FakeWorker()
    return TestClient(api)


def test_api_board_lines_and_patch(client):
    b = client.get("/api/board", params={"voice": "bag", "lang": "sk"}).json()
    assert b == board.board("bag", "sk") and b["categories"] == BAG_SK
    assert client.get("/api/board").json() == b

    created = client.post("/api/lines", json={"voice": "bag", "lang": "sk", "category": "Vlastné", "text": "Nová."})
    assert created.status_code == 200 and created.json()["status"] == "pending"
    too_long = client.post("/api/lines", json={"voice": "bag", "lang": "sk", "category": "V", "text": "a " * 200})
    assert (too_long.status_code, too_long.json()["detail"]) == (400, "too_long")
    assert client.post("/api/lines", json={"voice": "bag", "lang": "sk", "category": "V", "text": " "}).status_code == 400

    patched = client.patch(f"/api/lines/{created.json()['id']}", json={"slot": 8})
    assert patched.status_code == 200 and (patched.json()["slot"], patched.json()["favourite"]) == (8, True)
    assert client.get("/api/board").json()["favourites"][7] == created.json()["id"]
    assert client.patch(f"/api/lines/{created.json()['id']}", json={"slot": 9}).status_code == 422
    assert client.patch("/api/lines/nope", json={"favourite": True}).status_code == 404


def test_api_regenerate_submits_a_live_job_with_the_next_take(client):
    line = board.board("bag", "sk")["lines"][3]
    pin_render(line["id"], line["text"], take_no=1)
    res = client.post(f"/api/lines/{line['id']}/regenerate")
    assert res.status_code == 200 and res.json() == {"job_id": "job-1"}
    job = client.app.state.worker.submitted[0]
    assert (job.kind, job.priority, job.voice_id, job.text, job.line_id, job.take_no) == (
        "regenerate", "live", "bag", line["text"], line["id"], 2)
    assert client.post("/api/lines/nope/regenerate").status_code == 404
