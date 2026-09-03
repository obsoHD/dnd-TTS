"""The player is the one thing that decides what the speaker plays, so these
tests pin the promises the table relies on: strict FIFO, never two
``play.start`` without an ``end``/``stop`` between them (two rapid taps never
overlap), Repeat/Next/Stop as the DM expects them, and a speaker claim that a
stale disconnect cannot revoke. The routers are exercised through
``TestClient`` with the store and the board mocked: no database beyond one
temp file, no network, no GPU box."""
from __future__ import annotations

import importlib
import random
from typing import Iterator

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from app import config, store
from app.api import play as play_api
from app.api import remote as remote_api
from app.player import RENDER_URL, Player

Event = tuple[str, dict]


@pytest.fixture
def events() -> list[Event]:
    return []


@pytest.fixture
def player(events: list[Event]) -> Player:
    return Player(lambda name, data: events.append((name, data)))


def names(events: list[Event]) -> list[str]:
    return [name for name, _ in events]


def starts(events: list[Event]) -> list[str]:
    return [data["render_id"] for name, data in events if name == "play.start"]


def assert_no_double_start(events: list[Event]) -> None:
    """The contract's guard: between two ``play.start`` there is an end or a stop."""
    armed = False
    for name, _ in events:
        if name == "play.start":
            assert not armed, f"two play.start without end/stop: {names(events)}"
            armed = True
        elif name in ("play.end", "play.stop"):
            armed = False


# --- Player -----------------------------------------------------------------

def test_enqueue_when_idle_starts_immediately(player: Player, events: list[Event]) -> None:
    out = player.enqueue("r1", "Ten nie.")
    assert out == {"render_id": "r1", "url": RENDER_URL.format("r1"), "label": "Ten nie.", "position": 0}
    assert events == [("play.start", {"render_id": "r1", "url": "/api/renders/r1.wav", "label": "Ten nie."})]
    assert player.state()["now"]["render_id"] == "r1"
    assert player.state()["queue"] == []


def test_fifo_and_ended_advances_one_at_a_time(player: Player, events: list[Event]) -> None:
    assert player.enqueue("a", "A")["position"] == 0
    assert player.enqueue("b", "B")["position"] == 1
    assert player.enqueue("c", "C")["position"] == 2
    assert starts(events) == ["a"]
    assert [i["render_id"] for i in player.state()["queue"]] == ["b", "c"]

    player.ended("a")
    assert names(events)[-2:] == ["play.end", "play.start"]
    assert events[-2] == ("play.end", {"render_id": "a"})
    assert player.state()["now"]["render_id"] == "b"

    player.ended("b")
    player.ended("c")
    assert starts(events) == ["a", "b", "c"]
    assert player.state()["now"] is None
    assert player.state()["queue"] == []
    assert_no_double_start(events)


def test_stale_ended_is_ignored(player: Player, events: list[Event]) -> None:
    player.enqueue("a", "A")
    player.enqueue("b", "B")
    player.ended("zzz")                       # a report for something not playing
    player.ended("b")                         # queued, not playing yet
    assert names(events) == ["play.start"]
    assert player.state()["now"]["render_id"] == "a"
    player.ended("")                          # idle-safe as well
    player.stop()
    player.ended("a")                         # trails the stop: must not start "b"
    assert names(events) == ["play.start", "play.stop"]


def test_stop_clears_queue_keeps_last_for_repeat(player: Player, events: list[Event]) -> None:
    player.enqueue("a", "A")
    player.enqueue("b", "B")
    player.stop()
    assert events[-1] == ("play.stop", {})
    state = player.state()
    assert state["now"] is None and state["queue"] == []
    assert state["last"]["render_id"] == "a"
    player.stop()                             # idle stop still acknowledges
    assert names(events) == ["play.start", "play.stop", "play.stop"]


def test_repeat_replays_last_when_idle(player: Player, events: list[Event]) -> None:
    player.repeat()                           # nothing played yet: no-op, no event
    assert events == []
    player.enqueue("a", "A")
    player.ended("a")
    player.repeat()
    assert starts(events) == ["a", "a"]
    assert player.state()["now"]["label"] == "A"
    assert_no_double_start(events)


def test_repeat_while_playing_goes_to_the_front(player: Player, events: list[Event]) -> None:
    player.enqueue("a", "A")
    player.enqueue("b", "B")
    player.repeat()
    assert [i["render_id"] for i in player.state()["queue"]] == ["a", "b"]
    assert starts(events) == ["a"]
    player.ended("a")
    assert player.state()["now"]["render_id"] == "a"
    player.ended("a")
    assert player.state()["now"]["render_id"] == "b"
    assert_no_double_start(events)


def test_repeat_after_stop_replays_the_line_cut_short(player: Player, events: list[Event]) -> None:
    player.enqueue("a", "A")
    player.stop()
    player.repeat()
    assert names(events) == ["play.start", "play.stop", "play.start"]
    assert player.state()["now"]["render_id"] == "a"


def test_next_pops_the_pending_item_after_a_stop(player: Player, events: list[Event]) -> None:
    player.enqueue("a", "A")
    player.enqueue("b", "B")
    player.enqueue("c", "C")
    player.next()
    assert names(events) == ["play.start", "play.stop", "play.start"]
    assert player.state()["now"]["render_id"] == "b"
    assert [i["render_id"] for i in player.state()["queue"]] == ["c"]
    player.next()
    assert player.state()["now"]["render_id"] == "c"
    player.next()                             # nothing pending: nothing to jump to
    assert player.state()["now"]["render_id"] == "c"
    assert names(events).count("play.stop") == 2
    assert_no_double_start(events)


def test_state_is_a_snapshot(player: Player) -> None:
    player.enqueue("a", "A")
    player.enqueue("b", "B")
    snap = player.state()
    snap["now"]["render_id"] = "hacked"
    snap["queue"].clear()
    assert player.state()["now"]["render_id"] == "a"
    assert len(player.state()["queue"]) == 1


def test_speaker_claim_and_release(player: Player, events: list[Event]) -> None:
    assert player.state()["speaker"] is False
    player.claim_speaker("couch")
    assert events == [("speaker.presence", {"connected": True})]
    assert player.state()["speaker"] is True
    player.release_speaker("tablet")          # not the speaker: ignored
    assert player.state()["speaker"] is True
    player.claim_speaker("kitchen")           # last claim wins
    player.release_speaker("couch")           # stale disconnect of the old speaker
    assert player.state()["speaker"] is True
    player.release_speaker("kitchen")
    assert events[-1] == ("speaker.presence", {"connected": False})
    assert player.state()["speaker"] is False
    assert names(events).count("speaker.presence") == 3


def test_guard_holds_under_random_operations(player: Player, events: list[Event]) -> None:
    """Whatever the table does in whatever order, the wire never carries two
    ``play.start`` in a row and ``now`` always matches the last start."""
    rng = random.Random(4)
    ids = [f"r{n}" for n in range(6)]
    for _ in range(500):
        op = rng.choice(["enqueue", "enqueue", "ended", "ended", "stop", "repeat", "next"])
        if op == "enqueue":
            player.enqueue(rng.choice(ids), "x")
        elif op == "ended":
            player.ended(rng.choice(ids))
        else:
            getattr(player, op)()
        state = player.state()
        last_start = next((d for n, d in reversed(events) if n == "play.start"), None)
        if state["now"] is not None:
            assert state["now"] == last_start
    assert_no_double_start(events)
    assert names(events).count("play.start") > 50


# --- routers ------------------------------------------------------------------

RENDERS = {"r1": {"id": "r1", "line_id": "L1"}, "r2": {"id": "r2", "line_id": None}}
TEXTS = {"L1": "Ten nie."}


@pytest.fixture
def api(events: list[Event], monkeypatch: pytest.MonkeyPatch) -> Iterator[TestClient]:
    app = FastAPI()
    app.state.player = Player(lambda name, data: events.append((name, data)))
    app.include_router(play_api.router)
    app.include_router(remote_api.router)
    monkeypatch.setattr(store, "get_render", RENDERS.get)
    monkeypatch.setattr(play_api, "_line_text", lambda line_id: TEXTS.get(line_id, ""))
    monkeypatch.setattr(remote_api, "REMOTE_KEY", "hush")
    monkeypatch.setattr(remote_api, "_board", lambda voice, lang: BOARDS[(voice, lang)])
    with TestClient(app) as client:
        yield client


def line(line_id: str, text: str, render_id: str | None) -> dict:
    return {"id": line_id, "text": text, "category": "Ten nie.", "status": "ready" if render_id else "pending",
            "render_id": render_id, "favourite": True, "slot": None}


BOARDS = {
    ("bag", "sk"): {
        "categories": ["Ten nie."],
        "lines": [line("L1", "Ten nie.", "r1"), line("L2", "Zdravím, kámo.", "r2"),
                  line("L3", "Ešte nie.", None)],
        "favourites": ["L2", "L3"],            # slot1 ready, slot2 pending, slots 3..8 empty
        "ten_nie": "L1",
    },
    ("bag", "en"): {"categories": [], "lines": [], "favourites": [], "ten_nie": None},
}


def test_api_play_enqueues_with_label(api: TestClient, events: list[Event]) -> None:
    r = api.post("/api/play", json={"render_id": "r1", "label": "Nope."})
    assert r.status_code == 200
    assert r.json() == {"render_id": "r1", "url": "/api/renders/r1.wav", "label": "Nope.", "position": 0}
    assert events == [("play.start", {"render_id": "r1", "url": "/api/renders/r1.wav", "label": "Nope."})]


def test_api_play_label_falls_back_to_the_line_text(api: TestClient, events: list[Event]) -> None:
    api.post("/api/play", json={"render_id": "r1"})
    api.post("/api/play", json={"render_id": "r2", "label": ""})
    assert events[0][1]["label"] == "Ten nie."
    assert api.get("/api/player").json()["queue"][0]["label"] == ""


def test_api_play_rejects_unknown_render(api: TestClient, events: list[Event]) -> None:
    assert api.post("/api/play", json={"render_id": "nope"}).status_code == 404
    assert api.post("/api/play", json={}).status_code == 422
    assert events == []


def test_api_transport_and_player_state(api: TestClient, events: list[Event]) -> None:
    api.post("/api/play", json={"render_id": "r1"})
    api.post("/api/play", json={"render_id": "r2"})
    assert api.post("/api/repeat").json()["queue"][0]["render_id"] == "r1"
    nxt = api.post("/api/next").json()
    assert nxt["now"]["render_id"] == "r1" and [i["render_id"] for i in nxt["queue"]] == ["r2"]
    stopped = api.post("/api/stop").json()
    assert stopped["now"] is None and stopped["queue"] == []
    assert api.get("/api/player").json() == stopped
    assert names(events) == ["play.start", "play.stop", "play.start", "play.stop"]
    assert_no_double_start(events)


def test_api_speaker_claim(api: TestClient, events: list[Event]) -> None:
    r = api.post("/api/speaker/claim", json={"client_id": "couch"})
    assert r.status_code == 200 and r.json()["speaker"] is True
    assert events == [("speaker.presence", {"connected": True})]
    assert api.post("/api/speaker/claim", json={}).status_code == 422


@pytest.mark.parametrize("query", ["", "?k=", "?k=bag", "?k=HUSH"])
def test_remote_rejects_a_bad_key(api: TestClient, events: list[Event], query: str) -> None:
    assert api.get(f"/remote/stop{query}").status_code == 403
    assert api.get(f"/remote/bogus{query}").status_code == 403   # the key is checked first
    assert events == []


def test_remote_slot_plays_the_favourite(api: TestClient, events: list[Event]) -> None:
    r = api.get("/remote/slot1?k=hush")
    assert r.status_code == 200
    assert r.json()["action"] == "slot1"
    assert r.json()["state"]["now"] == {"render_id": "r2", "url": "/api/renders/r2.wav", "label": "Zdravím, kámo."}
    assert events == [("play.start", {"render_id": "r2", "url": "/api/renders/r2.wav", "label": "Zdravím, kámo."})]


def test_remote_tennie_plays_the_pinned_line(api: TestClient, events: list[Event]) -> None:
    assert api.get("/remote/tennie?k=hush").json()["state"]["now"]["label"] == "Ten nie."
    assert starts(events) == ["r1"]


def test_remote_reports_empty_and_unrendered_slots(api: TestClient, events: list[Event]) -> None:
    assert api.get("/remote/slot2?k=hush").status_code == 409       # favourite without a render
    assert api.get("/remote/slot8?k=hush").status_code == 404       # nothing in the slot
    assert api.get("/remote/slot1?k=hush&lang=en").status_code == 404
    assert api.get("/remote/tennie?k=hush&lang=en").status_code == 404
    assert api.get("/remote/slot9?k=hush").status_code == 404       # not an action
    assert api.get("/remote/play?k=hush").status_code == 404
    assert events == []


def test_remote_transport_actions(api: TestClient, events: list[Event]) -> None:
    api.get("/remote/slot1?k=hush")
    api.get("/remote/tennie?k=hush")
    assert api.get("/remote/repeat?k=hush").json()["state"]["queue"][0]["render_id"] == "r2"
    assert api.get("/remote/next?k=hush").json()["state"]["now"]["render_id"] == "r2"
    stopped = api.get("/remote/stop?k=hush").json()
    assert stopped == {"action": "stop", "state": {"now": None, "queue": [], "last": stopped["state"]["last"],
                                                    "speaker": False}}
    assert names(events) == ["play.start", "play.stop", "play.start", "play.stop"]
    assert_no_double_start(events)


def test_remote_key_defaults_to_bag(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("BAG_REMOTE_KEY", raising=False)
    assert importlib.reload(remote_api).REMOTE_KEY == "bag"
    monkeypatch.setenv("BAG_REMOTE_KEY", "pedal")
    assert importlib.reload(remote_api).REMOTE_KEY == "pedal"
    monkeypatch.undo()
    importlib.reload(remote_api)


# --- the one real database touch ---------------------------------------------

@pytest.fixture
def data_dir(tmp_path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("BAG_DATA", str(tmp_path))
    importlib.reload(config)
    store.init_db()
    yield tmp_path
    monkeypatch.undo()
    importlib.reload(config)


def test_line_text_reads_the_bank(data_dir) -> None:
    lid = store.upsert_line("bag", "sk", "Ten nie.", "Ten nie.", "bank")
    assert play_api._line_text(lid) == "Ten nie."
    assert play_api._line_text("missing") == ""
    assert play_api._line_text(None) == ""
