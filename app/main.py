"""App factory and boot sequence for the Bag render service (M2).

Run: ``uvicorn app.main:app --host 0.0.0.0 --port 8020``.

Shared objects live on ``app.state`` so every router, whoever built it, finds
the same instances (reach them as ``request.app.state.<name>``):

    hub       app.ws.Hub               fan-out to browsers; ``broadcast(event, data)`` from any thread
    player    app.player.Player        server-side playback FIFO and the speaker claim
    worker    app.worker.RenderWorker  the one render thread; ``submit`` / ``cancel`` / ``queue_snapshot``
    voices    dict[str, Voice]         voices loaded at boot, keyed by id (``DATA_DIR/voices/<id>/voice.yaml``)
    tts_warm  bool                     True once a job came back from the TTS itself (not the cache) since boot

Boot order is the contract's: ``init_db`` -> ``import_bank`` -> load voices ->
``Hub`` -> ``Player`` -> ``RenderWorker.start()`` -> queue the prerender plan of
every locked voice at ``batch`` priority. The worker and the player share one
``on_event`` callback that feeds the metrics and then the hub, so nothing else
has to know about either.
"""
from __future__ import annotations

import asyncio
import importlib
import logging
from contextlib import asynccontextmanager
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType
from typing import AsyncIterator, Callable

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from app import config, llm, store, ws
from app.api import health, renders
from app.api import voices as voices_api
from app.ws import Hub

log = logging.getLogger("bag.main")

WEB_DIR = Path(__file__).resolve().parent.parent / "web"
PRERENDER_KIND = "bank"
# Routers other builders own, by module name -> owner named in the error when the file is missing.
# ``write`` (M3: POST /api/fix, GET /api/deliveries) rides the same mechanism; without it the
# improv bar's pencil and delivery pill answer 404 in the assembled app. ``creator`` (M4) is the
# same story for the whole Voice Creator API behind the ``/creator`` page.
PEER_ROUTERS = {"say": "worker", "jobs": "worker", "board": "board", "play": "player", "remote": "player",
                "write": "writer", "creator": "creator"}


@dataclass(frozen=True)
class Peers:
    """The modules the boot sequence drives but does not own."""

    jobs: ModuleType
    worker: ModuleType
    board: ModuleType
    player: ModuleType


def _peer(name: str, owner: str) -> ModuleType:
    """Import a module another builder owns. A missing file is a milestone
    status, not a bug here, so the error names whose file it is instead of a
    bare ModuleNotFoundError; an import error *inside* the module propagates as is."""
    try:
        return importlib.import_module(name)
    except ModuleNotFoundError as e:
        if e.name != name:
            raise
        raise RuntimeError(f"{name.replace('.', '/')}.py is missing; it is owned by the {owner} builder "
                           "(docs/M2-contracts.md)") from e


def _peers() -> Peers:
    return Peers(jobs=_peer("app.jobs", "worker"), worker=_peer("app.worker", "worker"),
                 board=_peer("app.board", "board"), player=_peer("app.player", "player"))


def _migrate(module: ModuleType) -> None:
    """Run ``ensure_columns()`` when the module defines one. The contract names
    the migration in both ``jobs.py`` and ``board.py``, but the M1 schema already
    carries the board's columns, so its owner may legitimately ship none."""
    migrate = getattr(module, "ensure_columns", None)
    if migrate is not None:
        migrate()


def _emitter(app: FastAPI) -> Callable[[str, dict], None]:
    """The single ``on_event`` for the worker and the player: telemetry first,
    then the browsers. Runs on the worker thread and in request handlers alike;
    both callees are thread-safe."""
    def emit(event: str, data: dict) -> None:
        if event == "job.done" and not data.get("cached"):
            app.state.tts_warm = True
        health.observe_event(event, data)
        app.state.hub.broadcast(event, data)
    return emit


def _enqueue_prerender(app: FastAPI, peers: Peers) -> int:
    """Queue each locked voice's plan (favourites first: that is the plan's
    order) at ``batch`` priority, skipping lines that are already ``ready`` so a
    warm restart does not flood the queue with instant cache hits. Unlocked
    voices have no trusted reference and would fail every render."""
    queued = 0
    for v in app.state.voices.values():
        if not voices_api.is_locked(v):
            continue
        lines = {line["id"]: line for line in peers.board.board(v.id, v.lang)["lines"]}
        for line_id in peers.board.prerender_plan(v.id, v.lang):
            line = lines.get(line_id)
            if line is None or line["status"] == "ready":
                continue
            # board.line_text, not line["text"]: a tile saved with a tone warms
            # the cache key that tone hashes to, which is the one the tap asks
            # for. A tone the Lab has since disarmed falls back to the bare line.
            app.state.worker.submit(peers.jobs.new_job(
                PRERENDER_KIND, "batch", v.id, peers.board.line_text(line, v), line_id=line_id))
            queued += 1
    return queued


SEEDED_BANK = Path(__file__).resolve().parent.parent / "data" / "phrases.json"


def _seed_bank() -> None:
    """Copy the shipped phrase bank into the data volume once.

    WHY once: the file in DATA_DIR is the operator's copy (editable, backed up
    with the renders); the one next to the code is only the shipped default, so
    re-copying on every boot would silently discard their edits. Missing on both
    sides is not fatal: import_bank reports zero lines and the app still serves.
    """
    live = config.DATA_DIR / "phrases.json"
    if live.exists() or not SEEDED_BANK.exists() or live == SEEDED_BANK:
        return
    live.parent.mkdir(parents=True, exist_ok=True)
    live.write_bytes(SEEDED_BANK.read_bytes())
    log.info("seeded %s from the image", live)


async def _startup(app: FastAPI, peers: Peers) -> None:
    store.init_db()
    _migrate(peers.jobs)
    _migrate(peers.board)
    _seed_bank()
    bank = peers.board.import_bank()
    app.state.voices = voices_api.load_all()
    hub = Hub()
    hub.bind(asyncio.get_running_loop())
    app.state.hub = hub
    app.state.tts_warm = False
    emit = _emitter(app)
    app.state.player = peers.player.Player(emit)
    worker = peers.worker.RenderWorker(emit)
    worker.start()
    app.state.worker = worker
    health.bind_worker(worker, peers.jobs.PRIORITY)
    llm.warm_in_background()      # a cold 27B is ~90 s; nobody should meet that mid-scene
    queued = _enqueue_prerender(app, peers)
    log.info("bank: %d lines; voices: %s; prerender queued: %d",
             bank, ", ".join(app.state.voices) or "none", queued)


def _shutdown(app: FastAPI) -> None:
    """Stop the worker when it offers ``stop()`` (the contract names none, the
    worker ships one): an in-flight render is cancelled instead of surviving
    the app, which matters when tests boot several apps in one process. The
    hub goes last so a job cancelled here can still report it."""
    stop = getattr(app.state.worker, "stop", None)
    if stop is not None:
        stop()
    app.state.hub.close()


def _page(name: str) -> FileResponse:
    """A page the web builder has not delivered yet is a 404 that says so; the
    API keeps working meanwhile. ``no-cache`` so an edited page shows on reload."""
    path = WEB_DIR / name
    if not path.is_file():
        raise HTTPException(status_code=404, detail=f"{path} is missing (web builder)")
    return FileResponse(path, media_type="text/html", headers={"Cache-Control": "no-cache"})


def create_app() -> FastAPI:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")
    peers = _peers()                                  # fail at import, not at first request

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        await _startup(app, peers)
        try:
            yield
        finally:
            _shutdown(app)

    app = FastAPI(title="Bag", lifespan=lifespan)
    app.include_router(health.router)
    app.include_router(voices_api.router)
    app.include_router(renders.router)
    app.include_router(ws.router)
    for name, owner in PEER_ROUTERS.items():
        app.include_router(_peer(f"app.api.{name}", owner).router)
    # check_dir=False: a missing web/ (web builder still at work) yields 404s, not a dead API.
    app.mount("/static", StaticFiles(directory=WEB_DIR, check_dir=False), name="static")

    @app.get("/", include_in_schema=False)
    def index() -> FileResponse:
        return _page("index.html")

    @app.get("/speaker", include_in_schema=False)
    def speaker() -> FileResponse:
        return _page("speaker.html")

    @app.get("/creator", include_in_schema=False)
    def creator() -> FileResponse:
        return _page("creator.html")

    return app


app = create_app()
