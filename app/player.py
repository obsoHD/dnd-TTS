"""Server-side playback state for the table (M2-contracts.md, REBUILD.md §4).

The browser "speaker" only executes what this queue decides, so two rapid
taps can never overlap: exactly one item is ``now`` at any time, and a
second ``play.start`` is impossible until the speaker reports ``ended`` or
the DM stops. State lives in memory on purpose: a speaker claim dies with
its socket, and a stalled queue is one ``Esc`` away. Every mutation and the
event it emits happen under one lock, because the API handlers run on
threadpool threads and the ordering of ``play.start``/``play.stop`` on the
wire is the whole guarantee.
"""
from __future__ import annotations

import threading
from collections import deque
from typing import Callable

RENDER_URL = "/api/renders/{}.wav"     # served by app/api/renders.py (service owner)

OnEvent = Callable[[str, dict], None]


def _item(render_id: str, label: str) -> dict:
    return {"render_id": render_id, "url": RENDER_URL.format(render_id), "label": label}


class Player:
    def __init__(self, on_event: OnEvent) -> None:
        self._emit = on_event
        self._lock = threading.RLock()
        self._now: dict | None = None
        self._last: dict | None = None
        self._queue: deque[dict] = deque()
        self._speaker: str | None = None

    def enqueue(self, render_id: str, label: str) -> dict:
        """FIFO: play now when idle, otherwise wait behind everything queued.
        Returns the item with its ``position`` (0 = playing now)."""
        item = _item(render_id, label)
        with self._lock:
            if self._now is None:
                self._start(item)
                return {**item, "position": 0}
            self._queue.append(item)
            return {**item, "position": len(self._queue)}

    def stop(self) -> None:
        """Esc: drop everything pending and silence the speaker. ``play.stop``
        goes out even when idle so every caller gets the same acknowledgement."""
        with self._lock:
            self._queue.clear()
            self._now = None
            self._emit("play.stop", {})

    def repeat(self) -> None:
        """R: the line that started most recently plays again before anything
        queued; after a stop that is the line just cut short."""
        with self._lock:
            if self._last is None:
                return
            item = dict(self._last)
            if self._now is None:
                self._start(item)
            else:
                self._queue.appendleft(item)

    def next(self) -> None:
        """Space/pedal: jump to the next pending item, cutting the current one
        short. A ``play.stop`` precedes the new start so the guard holds; with
        nothing pending there is nothing to jump to and the call is a no-op."""
        with self._lock:
            if not self._queue:
                return
            self._now = None
            self._emit("play.stop", {})
            self._start(self._queue.popleft())

    def ended(self, render_id: str) -> None:
        """The speaker finished ``render_id``: advance. A report for anything
        but the current item is stale (it trails a stop or a skip) and ignored."""
        with self._lock:
            if self._now is None or self._now["render_id"] != render_id:
                return
            self._now = None
            self._emit("play.end", {"render_id": render_id})
            if self._queue:
                self._start(self._queue.popleft())

    def state(self) -> dict:
        with self._lock:
            return {"now": dict(self._now) if self._now else None,
                    "queue": [dict(i) for i in self._queue],
                    "last": dict(self._last) if self._last else None,
                    "speaker": self._speaker is not None}

    def claim_speaker(self, client_id: str) -> None:
        """Last claim wins: the DM walked to another device and wants sound there."""
        with self._lock:
            self._speaker = client_id
            self._emit("speaker.presence", {"connected": True})

    def release_speaker(self, client_id: str) -> None:
        """Only the current speaker's release counts: a stale disconnect from a
        page that was taken over must not report the new speaker gone."""
        with self._lock:
            if self._speaker != client_id:
                return
            self._speaker = None
            self._emit("speaker.presence", {"connected": False})

    def _start(self, item: dict) -> None:
        """The one place ``play.start`` is emitted; callers hold the lock and
        have cleared ``_now`` first, which is what makes the guard hold."""
        self._now = item
        self._last = item
        self._emit("play.start", dict(item))
