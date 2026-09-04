"""HTTP face of the board: the Play page reads it once per voice and language
and patches single tiles. Every handler is a thin translation from JSON to
``app.board`` and from its errors to status codes; no board logic lives here.

Contract: docs/M2-contracts.md, section ``API``.
"""
from __future__ import annotations

from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel, Field

from app import board
from app.canon import TooLong

router = APIRouter()


class NewLine(BaseModel):
    voice: str
    lang: str
    text: str
    category: str | None = None  # omitted -> board.SAVED_CATEGORY[lang]
    # The tone the tile keeps (an ``app.delivery`` id, ``"bare"`` for none).
    # Omitted leaves an existing tile's tone alone; unknown or unarmed -> 400.
    delivery: str | None = None


class LinePatch(BaseModel):
    favourite: bool | None = None
    slot: int | None = Field(default=None, ge=0, le=board.SLOTS,
                             description="1-8 puts the line on that key; 0 takes it off the row")
    category: str | None = None
    delivery: str | None = Field(default=None, description='an armed delivery id; "bare" clears the tone')


@router.get("/api/board")
def get_board(voice: str = "bag", lang: str = "sk") -> dict:
    return board.board(voice, lang)


@router.post("/api/lines")
def post_line(body: NewLine) -> dict:
    try:
        return board.add_line(body.voice, body.lang, body.category, body.text, delivery=body.delivery)
    except TooLong as e:
        raise HTTPException(400, "too_long") from e
    except ValueError as e:
        raise HTTPException(400, str(e)) from e


@router.delete("/api/lines/{line_id}")
def delete_line(line_id: str) -> dict:
    try:
        board.delete_line(line_id)
    except board.LineNotFound as e:
        raise HTTPException(404, "line not found") from e
    except board.BankLine as e:
        raise HTTPException(409, "bank line") from e
    return {"ok": True}


@router.patch("/api/lines/{line_id}")
def patch_line(line_id: str, body: LinePatch) -> dict:
    try:
        return board.set_line(line_id, favourite=body.favourite, slot=body.slot, category=body.category,
                              delivery=body.delivery)
    except board.LineNotFound as e:
        raise HTTPException(404, "line not found") from e
    except ValueError as e:
        raise HTTPException(400, str(e)) from e


@router.post("/api/lines/{line_id}/regenerate")
def regenerate(line_id: str, request: Request) -> dict:
    """A fresh take at ``live`` priority: the DM is waiting at the table."""
    from app.jobs import new_job  # the worker owner's module; bound here so the board imports without it

    try:
        take = board.next_take(line_id)
        line = board.get_line(line_id)
    except board.LineNotFound as e:
        raise HTTPException(404, "line not found") from e
    # ``line_text`` and not ``take["text"]``: a regenerated take must sound like
    # the tile the DM tapped, so it carries the tile's own tone (and falls back
    # to the bare line when that tone is no longer armed).
    text = board.line_text(line, board.voice_for(take["voice_id"]))
    job = new_job("regenerate", "live", take["voice_id"], text, line_id=line_id, take_no=take["take_no"])
    return {"job_id": request.app.state.worker.submit(job).id}
