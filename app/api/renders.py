"""Serving renders and pinning them to lines (contract: ``/api/renders``).

A render id hashes everything that can change the bytes (``app.render``), so
``/api/renders/{id}.wav`` may be cached forever: the header says ``immutable``
and a tablet that played a line once never fetches it again. Pinning makes a
render the line's active one; the worker only sets ``active_render_id`` when
the line has none, so a pin survives later takes and re-imports of the bank.
"""
from __future__ import annotations

from contextlib import closing
from pathlib import Path

from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel

from app import store

router = APIRouter(prefix="/api/renders")

IMMUTABLE = "public, max-age=31536000, immutable"


class PinBody(BaseModel):
    line_id: str


def _render(render_id: str) -> dict:
    row = store.get_render(render_id)
    if row is None:
        raise HTTPException(status_code=404, detail="unknown render")
    return row


def _line(line_id: str) -> dict:
    with closing(store.db()) as con:
        row = con.execute("SELECT id, voice_id FROM lines WHERE id=?", (line_id,)).fetchone()
    if row is None:
        raise HTTPException(status_code=404, detail="unknown line")
    return dict(row)


@router.get("/{render_id}.wav")
def render_wav(render_id: str) -> FileResponse:
    """Registered before ``/{render_id}`` because that pattern also matches ``x.wav``."""
    path = Path(_render(render_id)["path"])
    if not path.is_file():
        raise HTTPException(status_code=404, detail="render file missing")
    return FileResponse(path, media_type="audio/wav", headers={"Cache-Control": IMMUTABLE})


@router.get("/{render_id}")
def render_row(render_id: str) -> dict:
    return _render(render_id)


@router.post("/{render_id}/pin")
def pin_render(render_id: str, body: PinBody) -> dict:
    """A render can only be pinned to a line of its own voice: the ids would
    accept it, but the tile would then play another character."""
    render = _render(render_id)
    line = _line(body.line_id)
    if line["voice_id"] != render["voice_id"]:
        raise HTTPException(status_code=409, detail="render belongs to another voice")
    store.set_active_render(body.line_id, render_id)
    return {"line_id": body.line_id, "render_id": render_id}
