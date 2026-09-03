"""Playback control: the table talks to the ``Player`` through these routes
and hears the outcome on ``/ws`` (M2-contracts.md API).

The ``Player`` lives on ``app.state.player`` (``main.py`` creates it after
the hub), so this module holds no state and a test can mount the router on
a bare app. Handlers are plain ``def``: the SQLite lookups run on the
threadpool, never on the event loop, and the ``Player`` lock covers the rest.
"""
from __future__ import annotations

from contextlib import closing

from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel

from app import store
from app.player import Player

router = APIRouter(prefix="/api")


class PlayRequest(BaseModel):
    render_id: str
    label: str | None = None


class ClaimRequest(BaseModel):
    client_id: str


def _player(request: Request) -> Player:
    return request.app.state.player


def _line_text(line_id: str | None) -> str:
    """Fallback label so the speaker's now-playing never goes blank when the
    caller sent only a render id (a bank render always knows its line)."""
    if line_id is None:
        return ""
    with closing(store.db()) as con:
        row = con.execute("SELECT text FROM lines WHERE id=?", (line_id,)).fetchone()
    return row["text"] if row else ""


@router.post("/play")
def play(body: PlayRequest, request: Request) -> dict:
    """Queue a finished render. A render the store does not know cannot be
    served by ``/api/renders``, so it is refused here instead of stalling the
    queue on a 404 inside the speaker."""
    row = store.get_render(body.render_id)
    if row is None:
        raise HTTPException(404, "unknown render")
    label = body.label or _line_text(row["line_id"])
    return _player(request).enqueue(body.render_id, label)


@router.post("/stop")
def stop(request: Request) -> dict:
    _player(request).stop()
    return _player(request).state()


@router.post("/repeat")
def repeat(request: Request) -> dict:
    _player(request).repeat()
    return _player(request).state()


@router.post("/next")
def next_item(request: Request) -> dict:
    _player(request).next()
    return _player(request).state()


@router.get("/player")
def player_state(request: Request) -> dict:
    return _player(request).state()


@router.post("/speaker/claim")
def claim_speaker(body: ClaimRequest, request: Request) -> dict:
    """HTTP twin of the WebSocket ``{"type":"claim"}`` so a speaker page can
    claim before its socket is up and a curl can hand audio to a device."""
    _player(request).claim_speaker(body.client_id)
    return _player(request).state()
