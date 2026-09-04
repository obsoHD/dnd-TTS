"""Liveness, readiness and Prometheus metrics (contract: ``/healthz``, ``/readyz``, ``/metrics``).

``/readyz`` is what the status dots poll, so it must answer within one probe
timeout no matter how many services are down: the three network probes run
side by side and never raise. It always answers 200 because the dict *is* the
answer; a 503 would make every dashboard fetch look like an outage.

Metrics are module-level collectors (one registry per process). Other modules
do not touch the collectors; they use the helpers:

* ``observe_event(event, data)`` – feed it every worker event (``app.main`` does
  this for the whole app), and cache hits/misses, gate rejects and per-stage
  durations fall out of the event stream without the worker knowing Prometheus.
* ``stage_timer(stage)`` – a ``with`` block for stages that have no
  ``job.progress`` event of their own (an STT or LLM call in M3). Do not wrap a
  stage the worker already reports through ``job.progress``: it would count twice.
* ``bind_worker(worker, priorities)`` – gauges the queue straight from the worker
  on every scrape instead of tracking deltas, which would drift on a restart.
"""
from __future__ import annotations

import time
import warnings
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Iterable
from urllib.parse import urlsplit

import requests
from fastapi import APIRouter, Request, Response
from prometheus_client import CONTENT_TYPE_LATEST, Counter, Gauge, Histogram, generate_latest
from urllib3.exceptions import InsecureRequestWarning

from app import config, tts_client
from app.api.voices import is_locked
from app.voices import Voice

router = APIRouter()

PROBE_TIMEOUT_S = 3.0
LLM_RESIDENT_FRACTION = 0.9     # below this ollama has spilled the model to CPU: too slow for the table
STAGE_BUCKETS = (0.05, 0.1, 0.25, 0.5, 1.0, 2.0, 3.0, 5.0, 10.0, 30.0)

QUEUE_DEPTH = Gauge("bag_queue_depth", "Jobs queued or running, by priority", ["priority"])
CACHE_HITS = Counter("bag_cache_hits_total", "Jobs answered from an existing gate-passed render")
CACHE_MISSES = Counter("bag_cache_misses_total", "Jobs that went to the TTS")
GATE_REJECTS = Counter("bag_gate_rejects_total", "Finished renders whose best take still failed the gate", ["reason"])
STAGE_SECONDS = Histogram("bag_stage_seconds", "Wall time per pipeline stage", ["stage"], buckets=STAGE_BUCKETS)

_open_stage: dict[str, tuple[str, float]] = {}      # job_id -> (stage, monotonic start)


def stage_timer(stage: str):
    """``with stage_timer("asr"): ...`` records the block's wall time under ``stage``."""
    return STAGE_SECONDS.labels(stage=stage).time()


def observe_event(event: str, data: dict) -> None:
    """Metrics from the worker's own events. A stage lasts from its
    ``job.progress`` to the next progress/done/failed of the same job. A
    ``job.queued`` (pre-empted and re-queued) or ``job.cancelled`` for a job
    with an open stage aborted that stage, so it is discarded, not measured."""
    job_id = str(data.get("job_id"))
    if event == "job.progress":
        _close_stage(job_id)
        _open_stage[job_id] = (str(data.get("stage")), time.monotonic())
    elif event in ("job.done", "job.failed"):
        _close_stage(job_id)
    elif event in ("job.queued", "job.cancelled"):
        _open_stage.pop(job_id, None)
    if event == "job.done":
        (CACHE_HITS if data.get("cached") else CACHE_MISSES).inc()
        if data.get("gate") == "failed":
            GATE_REJECTS.labels(reason="gate").inc()


def _close_stage(job_id: str) -> None:
    opened = _open_stage.pop(job_id, None)
    if opened is not None:
        STAGE_SECONDS.labels(stage=opened[0]).observe(time.monotonic() - opened[1])


def bind_worker(worker: Any, priorities: Iterable[str]) -> None:
    for priority in priorities:
        QUEUE_DEPTH.labels(priority=priority).set_function(
            lambda p=priority: sum(1 for job in worker.queue_snapshot() if job.get("priority") == p))


def stt_up() -> bool:
    """``/health`` next to the STT endpoint (``.../stt`` -> ``.../health``). The
    whisper box is self-signed, hence ``verify=False``; any non-5xx answer means
    the process is up, which is all the dot claims."""
    url = urlsplit(config.STT_URL)._replace(path="/health", query="", fragment="").geturl()
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", InsecureRequestWarning)
            return requests.get(url, timeout=PROBE_TIMEOUT_S, verify=False).status_code < 500
    except requests.RequestException:
        return False


def llm_residency() -> str:
    """``resident`` when ollama holds ``LLM_MODEL`` (almost) entirely in VRAM,
    ``loading`` while a warm-up is in flight, else ``absent``. A partially
    offloaded 27B answers, but not at table speed."""
    from app import llm      # imported here so health stays cheap for the probes above

    if llm.warming():
        return "loading"
    try:
        models = requests.get(config.LLM_URL + "/api/ps", timeout=PROBE_TIMEOUT_S).json().get("models", [])
    except (requests.RequestException, ValueError):
        return "absent"
    for m in models:
        if (m.get("name") or m.get("model")) != config.LLM_MODEL:
            continue
        if m.get("size_vram", 0) / max(1, m.get("size") or 1) >= LLM_RESIDENT_FRACTION:
            return "resident"
    return "absent"


def bank_ready(voices: dict[str, Voice]) -> bool:
    """Every line of each locked voice's prerender plan has a render (contract).
    Unlocked voices cannot render at all, so they do not hold the bank back."""
    from app import board  # board-owned module; imported here so a missing file fails in main's clear check, not here

    for v in voices.values():
        if not is_locked(v):
            continue
        rendered = {line["id"] for line in board.board(v.id, v.lang)["lines"] if line.get("render_id")}
        if any(line_id not in rendered for line_id in board.prerender_plan(v.id, v.lang)):
            return False
    return True


def readiness(state: Any) -> dict:
    """The ``/readyz`` document, from ``app.state`` (see ``app/main.py``)."""
    with ThreadPoolExecutor(max_workers=3) as pool:
        tts, stt, llm = pool.submit(tts_client.health), pool.submit(stt_up), pool.submit(llm_residency)
    return {
        "tts": tts.result(),
        "tts_warm": bool(state.tts_warm),
        "stt": stt.result(),
        "llm": llm.result(),
        "speaker": bool(state.player.state().get("speaker")),
        "bank_ready": bank_ready(state.voices),
        "queue_depth": len(state.worker.queue_snapshot()),
    }


@router.get("/healthz")
def healthz() -> dict:
    return {"ok": True}


@router.get("/readyz")
def readyz(request: Request) -> dict:
    """Plain ``def`` on purpose: FastAPI runs it on a worker thread, so the
    blocking probes never stall the event loop that serves the WebSocket."""
    return readiness(request.app.state)


@router.get("/metrics")
def metrics() -> Response:
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)
