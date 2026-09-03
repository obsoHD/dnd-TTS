"""The Writer's two endpoints: ``POST /api/fix`` and ``GET /api/deliveries``.

Contract: docs/M3-contracts.md, section ``app/api/write.py``. Both are thin: the
correction lives in ``app.llm`` and the delivery list in ``app.delivery``, so
neither of those modules needs a FastAPI import and both stay testable without a
client. Register this router in ``app/main.py`` the way the other peer routers
are registered (``PEER_ROUTERS``).

Nothing here renders. ``/api/fix`` hands text back to the improv bar and the DM
decides whether to speak it; ``/api/deliveries`` only reports what the Lab has
already measured for this voice.
"""
from __future__ import annotations

import importlib
from types import ModuleType

from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel, Field

from app import llm
from app.voices import Voice

router = APIRouter()
BRAIN_DOWN = "mozog nie je pripravený"


class FixRequest(BaseModel):
    voice: str
    text: str = Field(min_length=1)
    lang: str = "sk"


class FixResponse(BaseModel):
    text: str
    original: str
    changed: bool
    note: str


def _delivery() -> ModuleType:
    """``app.delivery`` belongs to the delivery builder and is imported at call
    time, following ``app.main._peer``: while that file is missing, ``/api/fix``
    must keep working and the error must name whose file it is instead of a bare
    ModuleNotFoundError from inside a request handler."""
    try:
        return importlib.import_module("app.delivery")
    except ModuleNotFoundError as e:
        if e.name != "app.delivery":
            raise
        raise RuntimeError("app/delivery.py is missing; it is owned by the delivery builder "
                           "(docs/M3-contracts.md)") from e


def _voice(request: Request, voice_id: str) -> Voice:
    v = request.app.state.voices.get(voice_id)
    if v is None:
        raise HTTPException(status_code=404, detail=f"unknown voice {voice_id!r}")
    return v


@router.post("/api/fix", response_model=FixResponse)
def post_fix(body: FixRequest, request: Request) -> FixResponse:
    """The pencil button. A brain that is not resident is a 503 with the Slovak
    the banner shows verbatim; every other failure comes back 200 with
    ``changed: false`` and a note, because the DM's line is still speakable."""
    try:
        return FixResponse(**llm.fix(body.text, _voice(request, body.voice), lang=body.lang))
    except llm.BrainNotReady as e:
        raise HTTPException(status_code=503, detail=BRAIN_DOWN) from e


@router.get("/api/deliveries")
def get_deliveries(request: Request, voice: str = "bag") -> list[dict]:
    """What the delivery pill offers for this voice: bare first, then every
    spice, with ``armed`` telling the UI which ones the Lab has measured."""
    return _delivery().available(_voice(request, voice))
