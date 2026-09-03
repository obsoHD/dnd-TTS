"""Job inspection and control: ``GET /api/jobs/{id}``, ``POST /api/jobs/{id}/cancel``,
``GET /api/queue``. Thin by design: the worker owns every rule, these only
translate its answers to HTTP. The worker is ``request.app.state.worker``.
"""
from __future__ import annotations

from fastapi import APIRouter, HTTPException, Request

from app.worker import RenderWorker

router = APIRouter()


def _worker(request: Request) -> RenderWorker:
    return request.app.state.worker


def _job_or_404(w: RenderWorker, job_id: str) -> dict:
    job = w.get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail=f"unknown job {job_id!r}")
    return job.as_dict()


@router.get("/api/jobs/{job_id}")
def get_job(job_id: str, request: Request) -> dict:
    return _job_or_404(_worker(request), job_id)


@router.post("/api/jobs/{job_id}/cancel")
def cancel_job(job_id: str, request: Request) -> dict:
    """``cancelled`` is False for a job that already finished. A running job is
    stopped cooperatively, so its status flips once the render lets go."""
    w = _worker(request)
    cancelled = w.cancel(job_id)
    return {"cancelled": cancelled, "job": _job_or_404(w, job_id)}


@router.get("/api/queue")
def queue(request: Request) -> list[dict]:
    return _worker(request).queue_snapshot()
