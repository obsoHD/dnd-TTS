"""``POST /api/say``: the table's one way to ask for a line.

It never waits for the render (REBUILD.md §6): the request is resolved to its
cache key, a job is queued, and the caller gets the job id, the render id and
whether the file already exists. ``cached: true`` lets the UI play at once
(second ``say`` of the same text in under 50 ms); otherwise it listens for
``job.done`` on ``/ws``. The worker is ``request.app.state.worker``.
"""
from __future__ import annotations

from typing import Literal

from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel, Field

from app import canon, worker
from app.jobs import new_job

router = APIRouter()
KIND = "say"


class SayRequest(BaseModel):
    voice: str
    text: str = Field(min_length=1)
    # Part of the contract's shape; not read. The voice's own language drives
    # canonicalisation (as in render_line), so the cache key can never fork.
    lang: str = "sk"
    priority: Literal["live", "prep", "batch"] = "live"
    take_no: int = Field(default=0, ge=0)
    # A board tile passes its line so the render becomes the line's active one.
    line_id: str | None = None


class SayResponse(BaseModel):
    job_id: str
    render_id: str
    cached: bool
    position: int


def _plan(req: SayRequest) -> worker.Plan:
    """Bad input is the caller's error (400/404), never a failed job."""
    try:
        return worker.plan(req.voice, req.text, req.take_no)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=f"unknown voice {req.voice!r}") from exc
    except (canon.TooLong, canon.BannedToken) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@router.post("/api/say", response_model=SayResponse)
def say(req: SayRequest, request: Request) -> SayResponse:
    if not req.text.strip():
        raise HTTPException(status_code=400, detail="empty text")
    p = _plan(req)
    job = new_job(KIND, req.priority, req.voice, req.text, line_id=req.line_id, take_no=req.take_no)
    job.render_id = p.render_id            # known now, so job.queued carries it
    w: worker.RenderWorker = request.app.state.worker
    w.submit(job)
    return SayResponse(job_id=job.id, render_id=p.render_id,
                       cached=worker.cached_render(p.render_id) is not None, position=w.position(job.id))
