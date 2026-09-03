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
    category: str
    text: str


class LinePatch(BaseModel):
    favourite: bool | None = None
    slot: int | None = Field(default=None, ge=0, le=board.SLOTS,
                             description="1-8 puts the line on that key; 0 takes it off the row")
    category: str | None = None


@router.get("/api/board")
def get_board(voice: str = "bag", lang: str = "sk") -> dict:
    return board.board(voice, lang)


@router.post("/api/lines")
def post_line(body: NewLine) -> dict:
    try:
        return board.add_line(body.voice, body.lang, body.category, body.text)
    except TooLong as e:
        raise HTTPException(400, "too_long") from e
    except ValueError as e:
        raise HTTPException(400, str(e)) from e


@router.patch("/api/lines/{line_id}")
def patch_line(line_id: str, body: LinePatch) -> dict:
    try:
        return board.set_line(line_id, favourite=body.favourite, slot=body.slot, category=body.category)
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
    except board.LineNotFound as e:
        raise HTTPException(404, "line not found") from e
    job = new_job("regenerate", "live", take["voice_id"], take["text"], line_id=line_id, take_no=take["take_no"])
    return {"job_id": request.app.state.worker.submit(job).id}
