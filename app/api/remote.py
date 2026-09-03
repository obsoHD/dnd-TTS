"""``GET /remote/{action}?k=KEY`` — the LAN pedal and anything else that can
only open a URL (REBUILD.md §4 "LAN remote / pedal").

GET because pedal firmware and phone shortcuts cannot POST a body; the
shared key keeps a guest's phone on the same Wi-Fi from stopping the Bag
mid-sentence. Slots and ``tennie`` resolve through ``board()`` so the remote
plays exactly the tile the Play page shows in that slot, for the Bag unless
``voice``/``lang`` say otherwise (the remote has no idea which voice is
active on a tablet).
"""
from __future__ import annotations

import os
import secrets

from fastapi import APIRouter, HTTPException, Query, Request

from app.player import Player

# Read once, like app/config.py: a test overrides the module attribute.
REMOTE_KEY = os.environ.get("BAG_REMOTE_KEY", "bag")
SLOTS = {f"slot{n}": n for n in range(1, 9)}
TRANSPORT = {"next": Player.next, "repeat": Player.repeat, "stop": Player.stop}

router = APIRouter(prefix="/remote")


def _board(voice: str, lang: str) -> dict:
    """Imported at call time: ``app.board`` belongs to another owner, and the
    remote must stay importable (and testable) without it."""
    from app.board import board

    return board(voice, lang)


def _slot_line_id(board: dict, action: str) -> str | None:
    """``favourites`` lists line ids by slot 1..8 (missing/None = empty slot);
    ``tennie`` is pinned to its own key, not a slot."""
    if action == "tennie":
        return board["ten_nie"]
    favourites = board["favourites"]
    index = SLOTS[action] - 1
    return favourites[index] if index < len(favourites) else None


def _playable(board: dict, action: str) -> dict:
    """The line behind a slot, or the HTTP reason it cannot be played right
    now: the remote never triggers a render, it only plays what is ready."""
    line_id = _slot_line_id(board, action)
    if line_id is None:
        raise HTTPException(404, f"{action} is empty")
    line = next((l for l in board["lines"] if l["id"] == line_id), None)
    if line is None or not line.get("render_id"):
        raise HTTPException(409, f"{action} is not rendered yet")
    return line


@router.get("/{action}")
def remote(action: str, request: Request, k: str = Query(""),
           voice: str = "bag", lang: str = "sk") -> dict:
    if not secrets.compare_digest(k, REMOTE_KEY):
        raise HTTPException(403, "bad key")
    player: Player = request.app.state.player
    if action in SLOTS or action == "tennie":
        line = _playable(_board(voice, lang), action)
        player.enqueue(line["render_id"], line["text"])
    elif action in TRANSPORT:
        TRANSPORT[action](player)
    else:
        raise HTTPException(404, "unknown action")
    return {"action": action, "state": player.state()}
