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
import yaml
from fastapi import FastAPI
from fastapi.testclient import TestClient

from app import board, canon, config, delivery, store
from app.api import board as board_api
from app.gate import TakeScore
from app.render import RenderResult

REPO = Path(__file__).resolve().parents[2]
PHRASES = REPO / "data" / "phrases.json"
SR = 24_000
BAG_SK = ["Pozdrav kámoša", "Urážka partie", "Chvastanie po záchrane", "Odmietnutie predmetu",
          "Bojový pokrik", "Sarkastická poznámka", "Namrzené povzbudenie", "Ten nie."]
# What the Lab writes into voice.yaml once it has measured a spice (M3 contract).
ARMED = {"vzdych": {"sim_drop": 0.011, "min_sim": 0.918, "n": 20, "armed_at": "2026-09-04T10:00:00Z"},
         "smiech": {"sim_drop": 0.017, "min_sim": 0.907, "n": 20, "armed_at": "2026-09-04T10:00:00Z"}}
TONED = "Toto poviem inak."
PLAIN = "Toto poviem normálne."
SIGH = "Toto <|sfx:sigh|>poviem inak."


def arm(voice_id: str = "bag", armed: dict | None = None) -> None:
    """Write the minimal voice.yaml ``app.delivery`` reads. Written per test
    rather than shipped in a fixture file, because half these tests are about
    what happens when the Lab changes its mind and disarms a spice."""
    voice_dir = config.voice_dir(voice_id)
    voice_dir.mkdir(parents=True, exist_ok=True)
    (voice_dir / "voice.yaml").write_text(
        yaml.safe_dump({"id": voice_id, "label": voice_id, "lang": "sk",
                        "armed_spices": ARMED if armed is None else armed},
                       sort_keys=False, allow_unicode=True), encoding="utf-8")


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
    assert len(board.board("bag", "sk")["lines"]) == sum(len(t) for t in bank["sk"]["bag"].values())
    with closing(store.db()) as con:
        npc = con.execute("SELECT voice_id, COUNT(*) n FROM lines WHERE lang='sk' AND category='Pozdrav' "
                          "GROUP BY voice_id ORDER BY voice_id").fetchall()
        sources = {row[0] for row in con.execute("SELECT DISTINCT source FROM lines")}
    greetings = len(bank["sk"]["npc"]["Pozdrav"])   # counted from the bank, not typed
    assert [(r["voice_id"], r["n"]) for r in npc] == [("female", greetings), ("male", greetings)]
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


def test_every_voice_gets_its_own_signature_line_on_the_t_key(bank):
    """Bag's "Ten nie." is Bag's; the merchant refuses to sell and an NPC tells
    you where to go. Each voice's signature category is kept off the slots."""
    npc_categories = list(bank["sk"]["npc"])
    signature = board.signature_category("male", "sk")
    slotted = [cat for cat in npc_categories if cat != signature]
    npc = board.board("male", "sk")
    assert npc["favourites"][:len(slotted)] == [store.line_id("male", "sk", cat, bank["sk"]["npc"][cat][0])
                                                for cat in slotted]
    assert npc["ten_nie"] == store.line_id("male", "sk", signature, bank["sk"]["npc"][signature][0])
    # Both NPC voices share the NPC bank, so each owns its own copy of the line.
    female = board.board("female", "sk")["ten_nie"]
    assert female == store.line_id("female", "sk", signature, bank["sk"]["npc"][signature][0])
    assert female != npc["ten_nie"]
    shop = board.board("shopkeep", "sk")
    shop_signature = board.signature_category("shopkeep", "sk")
    assert shop["ten_nie"] == store.line_id("shopkeep", "sk", shop_signature,
                                            bank["sk"]["shopkeep"][shop_signature][0])
    assert board.signature_category("unknown-voice", "sk") is None


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
    # A category sits where its earliest line sits, so move a line that is inside
    # the second bank category but is not its first: the second category keeps its
    # place and the new one lands right after it. Picked from the data, not by row
    # number, so growing the bank cannot silently retarget this test.
    second = [line for line in board.board("bag", "sk")["lines"] if line["category"] == BAG_SK[1]]
    line = second[1]
    assert board.set_line(line["id"], category="  Nová   kategória ")["category"] == "Nová kategória"
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
                    "favourite": False, "slot": None, "delivery": None, "delivery_armed": False}
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


def test_add_line_defaults_category_when_omitted_and_stays_idempotent(bank):
    saved = board.add_line("bag", "sk", None, "  Toto si zapamätaj. ")
    assert saved["category"] == board.SAVED_CATEGORY["sk"] == "Moje"
    again = board.add_line("bag", "sk", None, "Toto si zapamätaj.")
    assert again["id"] == saved["id"]
    assert board.categories("bag", "sk")[-1] == "Moje"
    en = board.add_line("bag", "en", None, "Remember this.")
    assert en["category"] == board.SAVED_CATEGORY["en"] == "Mine"
    with closing(store.db()) as con:
        assert con.execute("SELECT source FROM lines WHERE id=?", (saved["id"],)).fetchone()[0] == "improv"


def test_delete_line_removes_a_saved_tile_but_refuses_a_bank_line(bank):
    saved = board.add_line("bag", "sk", None, "Vlastná veta na zmazanie.")
    render_id = pin_render(saved["id"], saved["text"])
    board.delete_line(saved["id"])
    with pytest.raises(board.LineNotFound):
        board.get_line(saved["id"])
    assert store.get_render(render_id) is not None  # the render outlives the deleted tile

    bank_line = board.board("bag", "sk")["lines"][0]
    with pytest.raises(board.BankLine):
        board.delete_line(bank_line["id"])
    assert board.get_line(bank_line["id"]) == bank_line
    with pytest.raises(board.LineNotFound):
        board.delete_line("nope")


def test_set_signature_gives_a_later_voice_its_own_t_line(data_dir):
    assert board.signature_category("grump", "sk") is None
    voice_dir = config.voice_dir("grump")
    voice_dir.mkdir(parents=True)
    (voice_dir / "voice.yaml").write_text(
        yaml.safe_dump({"id": "grump", "label": "Grump"}, sort_keys=False), encoding="utf-8")

    board.set_signature("grump", "sk", "Odmietnutie")
    board.set_signature("grump", "en", "Refusal")
    assert board.signature_category("grump", "sk") == "Odmietnutie"
    assert board.signature_category("grump", "en") == "Refusal"
    data = yaml.safe_load((voice_dir / "voice.yaml").read_text(encoding="utf-8"))
    assert data["label"] == "Grump"  # existing keys survive the rewrite
    assert data["signature_category"] == {"sk": "Odmietnutie", "en": "Refusal"}
    # the four built-in voices keep their hard-coded signature, untouched by the fallback
    assert board.signature_category("bag", "sk") == "Ten nie."


def test_set_signature_creates_voice_yaml_when_none_exists_yet(data_dir):
    assert not config.voice_dir("newbie").exists()
    board.set_signature("newbie", "sk", "Kategória")
    assert board.signature_category("newbie", "sk") == "Kategória"


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
        assert "delivery" not in {row["name"] for row in con.execute("PRAGMA table_info(lines)")}
    board.ensure_columns()
    board.ensure_columns()
    with closing(store.db()) as con:
        cols = {row["name"]: row for row in con.execute("PRAGMA table_info(lines)")}
    assert cols["favourite"]["dflt_value"] == "0" and cols["favourite"]["notnull"] == 1
    assert cols["slot"]["type"] == "INTEGER"
    # M5: nullable, no default, because every line saved before M5 is a bare line.
    assert (cols["delivery"]["type"], cols["delivery"]["notnull"], cols["delivery"]["dflt_value"]) == ("TEXT", 0, None)
    board.import_bank(PHRASES)
    assert board.board("bag", "sk")["favourites"][0] is not None
    assert {line["delivery"] for line in board.board("bag", "sk")["lines"]} == {None}   # the bank ships bare


def test_add_line_stores_a_tone_and_a_second_save_retones_the_one_tile(bank):
    """The ask: a favourite must remember the tone the DM picked. The id hashes
    the text and not the tone, so re-saving is a re-tone, never a second tile."""
    arm()
    toned = board.add_line("bag", "sk", None, TONED, delivery="vzdych")
    assert (toned["delivery"], toned["delivery_armed"]) == ("vzdych", True)
    assert toned["id"] == store.line_id("bag", "sk", board.SAVED_CATEGORY["sk"], TONED)

    retoned = board.add_line("bag", "sk", None, TONED, delivery="smiech")
    assert retoned["id"] == toned["id"] and retoned["delivery"] == "smiech"
    assert board.add_line("bag", "sk", None, TONED)["delivery"] == "smiech"   # omitted leaves the tone alone
    assert board.add_line("bag", "sk", None, TONED, delivery="bare")["delivery"] is None
    assert board.add_line("bag", "sk", None, TONED, delivery="vzdych")["delivery"] == "vzdych"
    assert board.add_line("bag", "sk", None, TONED, delivery="")["delivery"] is None

    lines = board.board("bag", "sk")["lines"]
    assert [line["text"] for line in lines].count(TONED) == 1
    assert lines[-1]["id"] == toned["id"]


def test_add_line_refuses_a_tone_the_lab_never_measured(bank):
    arm(armed={"vzdych": ARMED["vzdych"]})
    with pytest.raises(delivery.NotArmed):
        board.add_line("bag", "sk", None, TONED, delivery="smiech")      # a real spice, not armed here
    with pytest.raises(delivery.NotArmed):
        board.add_line("bag", "sk", None, TONED, delivery="sepot")       # not a spice at all
    # refused before the row is written, so no bare tile is left behind
    assert not any(line["text"] == TONED for line in board.board("bag", "sk")["lines"])
    assert isinstance(delivery.NotArmed("x"), ValueError)                # what the API turns into a 400


def test_add_line_refuses_a_tone_for_a_voice_with_no_voice_yaml(bank):
    assert board.voice_for("bag") is None
    with pytest.raises(delivery.NotArmed):
        board.add_line("bag", "sk", None, TONED, delivery="vzdych")
    # clearing must still work, or a tile could get stuck on a tone forever
    assert board.add_line("bag", "sk", None, TONED, delivery="bare")["delivery"] is None


def test_set_line_retones_a_tile_and_bare_clears_it(bank):
    arm()
    line = board.board("bag", "sk")["lines"][0]
    assert board.set_line(line["id"], delivery="vzdych")["delivery"] == "vzdych"
    assert board.get_line(line["id"])["delivery_armed"] is True
    assert board.set_line(line["id"], favourite=True)["delivery"] == "vzdych"     # None leaves it alone
    assert board.set_line(line["id"], delivery="smiech", slot=2)["delivery"] == "smiech"
    assert board.get_line(line["id"])["slot"] == 2
    assert board.set_line(line["id"], delivery="bare")["delivery"] is None
    assert board.set_line(line["id"], delivery="")["delivery"] is None

    with pytest.raises(delivery.NotArmed):
        board.set_line(line["id"], delivery="krik", category="Ina")
    # the refused patch wrote nothing at all, not even the category beside it
    assert board.get_line(line["id"])["category"] == line["category"]
    with pytest.raises(board.LineNotFound):
        board.set_line("nope", delivery="vzdych")


def test_a_real_re_tone_unpins_the_tile_so_the_next_tap_is_heard_in_it(bank):
    """A ready tile plays its pinned take straight from the cache, and the
    worker only adopts a render when nothing is pinned. So a re-tone that keeps
    the pin would leave the DM tapping a tile that answers in the old tone for
    good; dropping the pin puts it back to pending, which is what M5 promises.
    The render row itself survives, and an unchanged tone keeps its pin."""
    arm()
    line = board.add_line("bag", "sk", None, TONED)
    rid = pin_render(line["id"], TONED)
    assert (board.get_line(line["id"])["status"], board.get_line(line["id"])["render_id"]) == ("ready", rid)

    retoned = board.set_line(line["id"], delivery="vzdych")
    assert (retoned["status"], retoned["render_id"]) == ("pending", None)
    with closing(store.db()) as con:            # the take is still in the store, only the pin went
        assert con.execute("SELECT 1 FROM renders WHERE id=?", (rid,)).fetchone() is not None

    again = pin_render(line["id"], SIGH)
    assert board.set_line(line["id"], delivery="vzdych")["render_id"] == again   # unchanged tone keeps the pin
    assert board.set_line(line["id"], favourite=True)["render_id"] == again      # and so does an untouched tone
    assert board.add_line("bag", "sk", None, TONED, delivery="vzdych")["render_id"] == again
    assert board.add_line("bag", "sk", None, TONED, delivery="smiech")["render_id"] is None

    pin_render(line["id"], TONED, take_no=1)
    assert board.set_line(line["id"], delivery="bare")["render_id"] is None      # clearing is a change too


def test_a_disarmed_tone_stays_on_the_tile_but_reads_as_unarmed(bank):
    """The Lab re-measures and drops a spice; a tile saved with it keeps showing
    what the DM chose (the UI greys it), instead of losing it or failing."""
    arm()
    line = board.add_line("bag", "sk", None, TONED, delivery="vzdych")
    arm(armed={"smiech": ARMED["smiech"]})
    after = board.get_line(line["id"])
    assert (after["delivery"], after["delivery_armed"]) == ("vzdych", False)
    assert after == next(row for row in board.board("bag", "sk")["lines"] if row["id"] == line["id"])
    config.voice_dir("bag").joinpath("voice.yaml").unlink()
    assert board.get_line(line["id"])["delivery_armed"] is False       # no voice.yaml, nothing armed, no crash


def test_line_text_applies_an_armed_tone_and_falls_back_to_the_bare_line(bank):
    arm()
    voice = board.voice_for("bag")
    toned = board.add_line("bag", "sk", None, TONED, delivery="vzdych")
    bare = board.add_line("bag", "sk", None, PLAIN)

    assert board.line_text(toned, voice) == SIGH
    assert board.line_text(bare, voice) == PLAIN
    assert board.line_text(toned, None) == TONED                       # unreadable voice -> the plain line
    arm(armed={})
    assert board.line_text(toned, board.voice_for("bag")) == TONED     # disarmed -> the plain line, never an error
    assert board.voice_for("bag") is not None


def test_line_text_is_idempotent_and_leaves_a_typed_token_alone(bank):
    """``delivery.apply`` owns both rules; this pins that ``line_text`` really
    routes through it, because the pre-render and the tap must produce the same
    text byte for byte or they warm and ask for different cache keys."""
    arm()
    voice = board.voice_for("bag")
    toned = board.add_line("bag", "sk", None, TONED, delivery="vzdych")
    typed = "Toto <|sfx:laughter|>poviem inak."
    assert board.line_text({**toned, "text": SIGH}, voice) == SIGH
    assert board.line_text({**toned, "text": typed}, voice) == typed


def test_a_tone_moves_neither_the_tile_nor_the_prerender_plan(bank):
    """The tone belongs to the line, not to the board's shape: order, slots and
    the T tile may not shift when a tile is re-toned."""
    arm()
    before = board.board("bag", "sk")
    plan = board.prerender_plan("bag", "sk")
    board.set_line(before["favourites"][0], delivery="vzdych")
    board.set_line(before["ten_nie"], delivery="smiech")
    after = board.board("bag", "sk")
    assert board.prerender_plan("bag", "sk") == plan
    assert [line["id"] for line in after["lines"]] == [line["id"] for line in before["lines"]]
    assert (after["favourites"], after["ten_nie"], after["categories"]) == (
        before["favourites"], before["ten_nie"], before["categories"])


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

    saved = client.post("/api/lines", json={"voice": "bag", "lang": "sk", "text": "Ulož toto."})
    assert saved.status_code == 200 and saved.json()["category"] == board.SAVED_CATEGORY["sk"]


def test_api_delete_line_refuses_a_bank_line_and_removes_a_saved_one(client):
    created = client.post("/api/lines", json={"voice": "bag", "lang": "sk", "category": "Vlastné", "text": "Zmiznúť."})
    line_id = created.json()["id"]
    bank_id = board.board("bag", "sk")["lines"][0]["id"]

    assert client.delete(f"/api/lines/{bank_id}").status_code == 409
    assert client.get("/api/board").json()["lines"][0]["id"] == bank_id  # the bank line survived

    assert client.delete(f"/api/lines/{line_id}").json() == {"ok": True}
    assert not any(line["id"] == line_id for line in client.get("/api/board").json()["lines"])
    assert client.delete(f"/api/lines/{line_id}").status_code == 404
    assert client.delete("/api/lines/nope").status_code == 404


def test_api_regenerate_submits_a_live_job_with_the_next_take(client):
    line = board.board("bag", "sk")["lines"][3]
    pin_render(line["id"], line["text"], take_no=1)
    res = client.post(f"/api/lines/{line['id']}/regenerate")
    assert res.status_code == 200 and res.json() == {"job_id": "job-1"}
    job = client.app.state.worker.submitted[0]
    assert (job.kind, job.priority, job.voice_id, job.text, job.line_id, job.take_no) == (
        "regenerate", "live", "bag", line["text"], line["id"], 2)
    assert client.post("/api/lines/nope/regenerate").status_code == 404


def test_api_lines_carry_a_tone_and_refuse_an_unarmed_one(client):
    arm()
    created = client.post("/api/lines", json={"voice": "bag", "lang": "sk", "text": TONED, "delivery": "vzdych"})
    assert created.status_code == 200
    assert (created.json()["delivery"], created.json()["delivery_armed"]) == ("vzdych", True)
    line_id = created.json()["id"]
    assert next(line for line in client.get("/api/board").json()["lines"]
                if line["id"] == line_id)["delivery"] == "vzdych"

    unarmed = client.post("/api/lines", json={"voice": "bag", "lang": "sk", "text": PLAIN, "delivery": "krik"})
    assert unarmed.status_code == 400 and "krik" in unarmed.json()["detail"]
    unknown = client.post("/api/lines", json={"voice": "bag", "lang": "sk", "text": PLAIN, "delivery": "sepot"})
    assert unknown.status_code == 400
    assert not any(line["text"] == PLAIN for line in client.get("/api/board").json()["lines"])

    retoned = client.patch(f"/api/lines/{line_id}", json={"delivery": "smiech"})
    assert retoned.status_code == 200 and retoned.json()["delivery"] == "smiech"
    assert client.patch(f"/api/lines/{line_id}", json={"delivery": "krik"}).status_code == 400
    assert client.patch(f"/api/lines/{line_id}", json={"favourite": True}).json()["delivery"] == "smiech"
    assert client.patch(f"/api/lines/{line_id}", json={"delivery": "bare"}).json()["delivery"] is None
    assert client.patch("/api/lines/nope", json={"delivery": "vzdych"}).status_code == 404


def test_api_regenerate_keeps_the_tiles_tone(client):
    arm()
    line_id = client.post("/api/lines",
                          json={"voice": "bag", "lang": "sk", "text": TONED, "delivery": "vzdych"}).json()["id"]
    assert client.post(f"/api/lines/{line_id}/regenerate").status_code == 200
    assert client.app.state.worker.submitted[-1].text == SIGH

    # the Lab disarms it: the tile still regenerates, just flat -- silence at
    # the table is worse than a flat line
    arm(armed={})
    assert client.post(f"/api/lines/{line_id}/regenerate").status_code == 200
    assert client.app.state.worker.submitted[-1].text == TONED
