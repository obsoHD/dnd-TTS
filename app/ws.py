"""WebSocket hub: one fan-out for every browser at the table (docs/M2-contracts.md, ``app/ws.py``).

Wire format, both directions, is one JSON object per frame.
Server -> client: ``{"type": "<event>", "data": {...}}`` using the contract's
event names (``status`` first on connect, then ``job.*``, ``queue.changed``,
``play.*``, ``speaker.presence``).
Client -> server: ``{"type": "ended", "render_id": "..."}`` when the speaker
finished a file and ``{"type": "claim"}`` to become the speaker.

Why the hub has its own sender task: events are born on the render worker
thread and inside request handlers, but a WebSocket may only be written from
the event loop, one frame at a time. ``broadcast`` therefore hands the
serialised frame to the loop (``call_soon_threadsafe``) and a single pump task
writes it to every socket, so no two coroutines ever write one socket at the
same time and every client sees events in the order they happened.

Why a per-client backlog: the connect snapshot takes real time (readiness
probes) while the worker keeps emitting. Frames that arrive for a socket before
its ``status`` has gone out are held and replayed right after it, so the first
frame a client ever sees is ``status`` and nothing that happened meanwhile is
lost (a replayed ``job.done`` the snapshot already reflects is harmless).
"""
from __future__ import annotations

import asyncio
import json
import logging
from dataclasses import dataclass, field
from typing import Any
from uuid import uuid4

from fastapi import APIRouter, Query, WebSocket, WebSocketDisconnect

from app.api import health

log = logging.getLogger("bag.ws")
router = APIRouter()

Frame = tuple[WebSocket | None, str]        # (one socket, or None for everyone) -> serialised JSON


@dataclass
class _Client:
    client_id: str
    ready: bool = False                     # its snapshot has been sent; until then broadcasts pile up here
    backlog: list[str] = field(default_factory=list)


class Hub:
    def __init__(self) -> None:
        # Only the loop thread touches ``_clients`` (connect/disconnect/pump), so no lock.
        self._clients: dict[WebSocket, _Client] = {}
        self._loop: asyncio.AbstractEventLoop | None = None
        self._queue: asyncio.Queue[Frame] | None = None
        self._pump: asyncio.Task[None] | None = None

    def bind(self, loop: asyncio.AbstractEventLoop) -> None:
        """Attach to the serving loop and start the pump. The lifespan calls it
        before the worker starts so boot-time events already have a loop to go
        to; ``connect`` calls it lazily as well so a bare Hub works on its own."""
        if self._loop is not None:
            return
        self._loop = loop
        self._queue = asyncio.Queue()
        self._pump = loop.create_task(self._drain())

    def close(self) -> None:
        if self._pump is not None:
            self._pump.cancel()

    async def connect(self, ws: WebSocket, client_id: str) -> None:
        self.bind(asyncio.get_running_loop())
        await ws.accept()
        self._clients[ws] = _Client(client_id)

    def disconnect(self, ws: WebSocket) -> None:
        """Idempotent: the pump drops a socket whose send failed and the
        endpoint's ``finally`` drops it again."""
        self._clients.pop(ws, None)

    def broadcast(self, event: str, data: dict) -> None:
        """Thread-safe: safe from the worker thread, a request handler or the loop itself."""
        self._post(None, event, data)

    def send(self, ws: WebSocket, event: str, data: dict) -> None:
        """One client only: the ``status`` snapshot. It marks the client ready
        and flushes whatever was broadcast while the snapshot was being built."""
        self._post(ws, event, data)

    def _post(self, target: WebSocket | None, event: str, data: dict) -> None:
        loop, queue = self._loop, self._queue
        if loop is None or queue is None or loop.is_closed():
            return                                   # no loop means nobody is connected
        frame = json.dumps({"type": event, "data": data}, ensure_ascii=False)
        try:
            loop.call_soon_threadsafe(queue.put_nowait, (target, frame))
        except RuntimeError:                         # loop torn down under a still-running worker thread
            log.debug("dropped %s after loop shutdown", event)

    async def _drain(self) -> None:
        assert self._queue is not None
        while True:
            target, frame = await self._queue.get()
            if target is None:
                await self._fan_out(frame)
                continue
            client = self._clients.get(target)
            if client is None:
                continue                             # gone before its snapshot was ready
            client.ready = True
            backlog, client.backlog = client.backlog, []
            await self._deliver({target: [frame, *backlog]})

    async def _fan_out(self, frame: str) -> None:
        batches: dict[WebSocket, list[str]] = {}
        for ws, client in self._clients.items():
            if client.ready:
                batches[ws] = [frame]
            else:
                client.backlog.append(frame)
        await self._deliver(batches)

    async def _deliver(self, batches: dict[WebSocket, list[str]]) -> None:
        sockets = list(batches)
        results = await asyncio.gather(*(self._write(ws, batches[ws]) for ws in sockets), return_exceptions=True)
        for ws, result in zip(sockets, results):
            if isinstance(result, BaseException):
                self.disconnect(ws)                  # dead socket; its receive loop finishes the cleanup

    @staticmethod
    async def _write(ws: WebSocket, frames: list[str]) -> None:
        for frame in frames:
            await ws.send_text(frame)


def _parse(text: str) -> dict[str, Any]:
    """A malformed client frame is ignored, never fatal: a flaky tablet must not
    drop its own connection by sending garbage."""
    try:
        msg = json.loads(text)
    except ValueError:
        return {}
    return msg if isinstance(msg, dict) else {}


def _status(state: Any) -> dict:
    """The connect snapshot: readiness plus everything the page would otherwise
    have to fetch before it can draw (player state and the queue)."""
    return {**health.readiness(state), "player": state.player.state(), "queue": state.worker.queue_snapshot()}


@router.websocket("/ws")
async def websocket_endpoint(ws: WebSocket, client: str = Query(""), role: str = Query("play")) -> None:
    state = ws.app.state
    hub: Hub = state.hub
    client_id = client or uuid4().hex
    claimed = role == "speaker"                      # the speaker page exists to be the speaker
    await hub.connect(ws, client_id)
    hub.send(ws, "status", await asyncio.to_thread(_status, state))
    try:
        while True:
            msg = _parse(await ws.receive_text())
            if msg.get("type") == "claim":
                state.player.claim_speaker(client_id)
                claimed = True
            elif msg.get("type") == "ended" and isinstance(msg.get("render_id"), str):
                state.player.ended(msg["render_id"])
    except WebSocketDisconnect:
        pass
    finally:
        hub.disconnect(ws)
        if claimed:
            state.player.release_speaker(client_id)  # a vanished speaker must not leave audio parked on it
