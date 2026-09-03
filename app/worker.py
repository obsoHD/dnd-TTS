"""The one render worker (REBUILD.md §6): a persisted priority heap, one job at
a time, and a live line that jumps the queue.

Two rules shape everything here. Two jobs never run at once, because the TTS
already batches the takes of one job and a second would only slow the first.
And a ``live`` job (the DM just hit Speak) must start within 500 ms even while
the bank pre-renders, so a running ``batch``/``prep`` job is asked to stop at
its next checkpoint (a ``threading.Event`` threaded down to the streaming
read), put back at the head of its priority band, and run again later.
Nothing it produced is lost, because nothing is stored before a job finishes.

Every state change and every event happens under one lock, so listeners see
``job.queued`` before ``job.started`` before ``job.done`` in that order no
matter which thread caused the transition.

Wiring (``app/main.py``): ``store.init_db()`` first, then construct the worker,
``start()`` it and put it on ``app.state.worker`` where the routers read it.
``stop()`` re-queues a running job so the next start resumes it.
"""
from __future__ import annotations

import heapq
import logging
import threading
import time
from contextlib import closing
from dataclasses import dataclass
from typing import Callable

from app import jobs, store
from app.canon import Canon, canonicalize
from app.jobs import PRIORITY, TERMINAL, Job
from app.render import render_id, render_line
from app.tts_client import Cancelled
from app.voices import Voice, load_voice

log = logging.getLogger(__name__)
TEXT_PREVIEW = 60            # enough of a line to recognise it in the queue view
STOP_TIMEOUT_S = 5.0

OnEvent = Callable[[str, dict], None]


@dataclass
class Plan:
    """What a request resolves to before any GPU work: the voice, the canonical
    text and the cache key."""

    voice: Voice
    canon: Canon
    render_id: str


def plan(voice_id: str, text: str, take_no: int) -> Plan:
    """The voice's own language drives canonicalisation, exactly as
    ``render_line`` does, so the key computed here is the key the render carries."""
    voice = load_voice(voice_id)
    c = canonicalize(text, lang=voice.lang, banned=set(voice.banned_tokens))
    return Plan(voice=voice, canon=c, render_id=render_id(voice, c.text, take_no))


def cached_render(rid: str) -> dict | None:
    """The stored render when it can be served as-is. A ``failed`` row is kept
    for Regenerate to improve on, but it never short-circuits: the table asked
    for the line, not for the best of a bad batch."""
    row = store.get_render(rid)
    return row if row is not None and row["gate"] == "pass" else None


def _line_active_render(line_id: str) -> str | None:
    with closing(store.db()) as con:
        row = con.execute("SELECT active_render_id FROM lines WHERE id=?", (line_id,)).fetchone()
    return row["active_render_id"] if row is not None else None


def adopt_render(line_id: str | None, rid: str) -> None:
    """A line's first render becomes its active one; after that only an explicit
    Pin replaces it, so a Regenerate never swaps a take the DM already liked."""
    if line_id is not None and _line_active_render(line_id) is None:
        store.set_active_render(line_id, rid)


class RenderWorker:
    def __init__(self, on_event: OnEvent):
        jobs.ensure_columns()
        self._on_event = on_event
        # Re-entrant: events go out under the lock, and a listener may look back at the queue.
        self._cv = threading.Condition(threading.RLock())
        self._heap: list[tuple[int, float, int, str]] = []
        self._seq = 0                          # FIFO within equal (priority, created)
        self._jobs: dict[str, Job] = {}
        self._running: Job | None = None
        self._cancel = threading.Event()       # the running job's cooperative stop
        self._requeue: bool | None = None      # None: nobody interrupted the running job
        self._stopping = False
        self._thread = threading.Thread(target=self._run, name="render-worker", daemon=True)

    # -- public --------------------------------------------------------------

    def start(self) -> None:
        with self._cv:
            self._recover()
            self._emit_queue()
        self._thread.start()

    def stop(self, timeout: float = STOP_TIMEOUT_S) -> None:
        """End the loop. A running job is interrupted and re-queued rather than
        dropped, so it survives to the next start."""
        with self._cv:
            self._stopping = True
            if self._running is not None:
                self._interrupt(requeue=True)
            self._cv.notify_all()
        if self._thread.is_alive():
            self._thread.join(timeout)

    def submit(self, job: Job) -> Job:
        with self._cv:
            self._jobs[job.id] = job
            self._set(job, "queued")
            self._push(job)
            if job.priority == "live" and self._running is not None and self._running.priority != "live":
                self._interrupt(requeue=True)
            self._emit("job.queued", {**self._brief(job), "position": self.position(job.id)})
            self._emit_queue()
            self._cv.notify_all()
        return job

    def cancel(self, job_id: str) -> bool:
        """True when the job was still pending. A queued job is dropped here; a
        running one is stopped cooperatively and the loop reports the outcome."""
        with self._cv:
            job = self._jobs.get(job_id)
            if job is None or job.status not in ("queued", "running"):
                return False
            if job.status == "running":
                self._interrupt(requeue=False)
                return True
            self._set(job, "cancelled", done=jobs.now())
            self._emit("job.cancelled", self._brief(job))
            self._emit_queue()
            self._cv.notify_all()
            return True

    def get(self, job_id: str) -> Job | None:
        """This process's jobs from memory; older ones from the table."""
        with self._cv:
            job = self._jobs.get(job_id)
        return job if job is not None else jobs.load(job_id)

    def position(self, job_id: str) -> int:
        """1-based place among queued jobs; 0 when running or already finished."""
        with self._cv:
            for pos, job in enumerate(self._queued(), start=1):
                if job.id == job_id:
                    return pos
        return 0

    def queue_snapshot(self) -> list[dict]:
        with self._cv:
            items = [(self._running, 0)] if self._running is not None else []
            items += [(job, pos) for pos, job in enumerate(self._queued(), start=1)]
        return [self._entry(job, pos) for job, pos in items]

    def wait(self, job_id: str, timeout: float) -> Job | None:
        """Block until the job is done/failed/cancelled; None on timeout or an
        unknown id. A re-queued job is still pending, so this keeps waiting."""
        deadline = time.monotonic() + timeout
        with self._cv:
            while True:
                job = self._jobs.get(job_id)
                if job is None:
                    return None
                if job.status in TERMINAL:
                    return job
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    return None
                self._cv.wait(remaining)

    # -- queue bookkeeping (call under the lock) ----------------------------

    def _queued(self) -> list[Job]:
        """Queued jobs in pop order. Cancelled entries stay in the heap until
        popped (cheaper than rebuilding it) and are filtered out here."""
        in_order = [self._jobs[entry[3]] for entry in sorted(self._heap)]
        return [job for job in in_order if job.status == "queued"]

    def _push(self, job: Job) -> None:
        self._seq += 1
        heapq.heappush(self._heap, (PRIORITY[job.priority], job.created, self._seq, job.id))

    def _pop(self) -> Job | None:
        while self._heap:
            job = self._jobs[heapq.heappop(self._heap)[3]]
            if job.status == "queued":
                return job
        return None

    def _set(self, job: Job, status: str, **fields) -> None:
        """One transition, persisted at once so memory and the table never disagree."""
        job.status = status
        for name, value in fields.items():
            setattr(job, name, value)
        jobs.save(job)

    def _interrupt(self, requeue: bool) -> None:
        """Stop the running job at its next checkpoint. A user cancel (drop)
        beats a preemption (re-queue) whichever arrived first."""
        self._requeue = requeue if self._requeue is None else (self._requeue and requeue)
        self._cancel.set()

    def _recover(self) -> None:
        """Jobs the previous process never finished go back on the heap in their
        original order. One caught ``running`` restarts from scratch: its takes
        were never stored. Jobs already submitted to this instance are skipped."""
        for job in jobs.pending():
            if job.id in self._jobs:
                continue
            self._jobs[job.id] = job
            self._set(job, "queued", started=None)
            self._push(job)

    # -- events (call under the lock) ----------------------------------------

    def _emit(self, name: str, data: dict) -> None:
        """A listener's failure is logged, never propagated: a UI hiccup must
        not stop the render loop."""
        try:
            self._on_event(name, data)
        except Exception:  # noqa: BLE001
            log.exception("event listener failed on %s", name)

    def _emit_queue(self) -> None:
        queued = self._queued()
        by_priority = {name: sum(1 for job in queued if job.priority == name) for name in PRIORITY}
        running = self._running.id if self._running is not None else None
        self._emit("queue.changed", {"depth": len(queued), "running": running, "by_priority": by_priority})

    def _progress(self, job: Job, stage: str) -> None:
        with self._cv:
            self._emit("job.progress", {"job_id": job.id, "stage": stage})

    @staticmethod
    def _brief(job: Job) -> dict:
        return {"job_id": job.id, "kind": job.kind, "priority": job.priority, "voice_id": job.voice_id,
                "line_id": job.line_id, "render_id": job.render_id}

    @staticmethod
    def _entry(job: Job, position: int) -> dict:
        return {"id": job.id, "kind": job.kind, "priority": job.priority, "voice_id": job.voice_id,
                "text": job.text[:TEXT_PREVIEW], "status": job.status, "position": position,
                "line_id": job.line_id, "render_id": job.render_id}

    # -- the loop (worker thread) --------------------------------------------

    def _run(self) -> None:
        while (claimed := self._claim()) is not None:
            self._execute(*claimed)

    def _claim(self) -> tuple[Job, threading.Event] | None:
        """Block for the next queued job; None once ``stop`` was called."""
        with self._cv:
            while not self._stopping:
                job = self._pop()
                if job is None:
                    self._cv.wait()
                    continue
                self._running = job
                self._cancel = threading.Event()
                self._requeue = None
                self._set(job, "running", started=jobs.now())
                self._emit("job.started", self._brief(job))
                self._emit_queue()
                return job, self._cancel
            return None

    def _execute(self, job: Job, cancel: threading.Event) -> None:
        try:
            payload = self._render(job, cancel)
        except Cancelled:
            self._settle_interrupted(job)
            return
        except Exception as exc:  # noqa: BLE001 - a dead worker takes the whole table down
            log.exception("job %s failed", job.id)
            error = f"{type(exc).__name__}: {exc}"
            self._settle(job, "failed", "job.failed", {**self._brief(job), "error": error}, error=error)
            return
        self._settle(job, "done", "job.done", payload)

    def _render(self, job: Job, cancel: threading.Event) -> dict:
        """Run one job to its ``job.done`` payload. The cache is checked here,
        not at submit time, so a queued twin of a line that just finished is
        served from disk too."""
        p = plan(job.voice_id, job.text, job.take_no)
        job.render_id = p.render_id
        row = cached_render(p.render_id)
        if row is not None:
            adopt_render(job.line_id, p.render_id)
            return self._done_payload(job, cached=True, sim=row["sim"], cer=row["cer"],
                                      verified=row["verified"], gate=row["gate"], dur_s=row["dur_s"])
        self._progress(job, "render")
        result = render_line(p.voice, job.text, job.take_no, cancel=cancel)
        self._progress(job, "store")
        raw_path, path = store.write_render_files(result)
        store.put_render(result, job.line_id, raw_path, path)
        adopt_render(job.line_id, result.render_id)
        return self._done_payload(job, cached=False, sim=result.sim, cer=result.cer, verified=result.verified,
                                  gate=result.gate, dur_s=store.duration_s(result.pcm, result.sr))

    def _done_payload(self, job: Job, **stats) -> dict:
        return {**self._brief(job), **stats}

    def _settle(self, job: Job, status: str, event: str, data: dict, **fields) -> None:
        with self._cv:
            self._running = None
            self._set(job, status, done=jobs.now(), **fields)
            self._emit(event, data)
            self._emit_queue()
            self._cv.notify_all()

    def _settle_interrupted(self, job: Job) -> None:
        """After a cooperative stop: back on the heap (preemption, shutdown) or
        dropped (user cancel), as decided by whoever interrupted it."""
        with self._cv:
            self._running = None
            if self._requeue:
                self._set(job, "queued", started=None)
                self._push(job)
                self._emit("job.queued", {**self._brief(job), "position": self.position(job.id)})
            else:
                self._set(job, "cancelled", done=jobs.now())
                self._emit("job.cancelled", self._brief(job))
            self._emit_queue()
            self._cv.notify_all()
