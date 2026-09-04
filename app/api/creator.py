"""HTTP face of the Voice Creator (contract: docs/M4-contracts.md, section 4).

Every handler is a thin translation from JSON or multipart into ``app.creator``
and from its errors into status codes; no creator logic lives here. The one
thing this file owns outright is the commit's shape: locking, calibrating and
banking a voice is minutes of GPU work, so the request returns immediately and
the DM watches ``creator.progress`` arrive on the ``/ws`` the Play page already
holds open.

Register this router in ``app/main.py`` the way the other peer routers are:
``PEER_ROUTERS`` gains ``"creator": "creator"``. Until it does, ``/creator`` and
every route below answer 404 in the assembled app.
"""
from __future__ import annotations

import logging
import threading
from pathlib import Path

from fastapi import APIRouter, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

from app import creator, llm, voices

log = logging.getLogger(__name__)
router = APIRouter(prefix="/api/creator")

BRAIN_DOWN = "mozog nie je pripravený"      # the same Slovak the banner shows verbatim (M3)
BRAIN_GARBLED = "mozog vrátil nepoužiteľnú odpoveď"
# Browsers label a wav from a phone as ``audio/*``, but a file dragged from a
# desktop often arrives as ``application/octet-stream``; the suffix is the
# fallback so a legitimate clip is not refused over a header.
AUDIO_SUFFIXES = (".wav", ".mp3", ".m4a", ".aac", ".ogg", ".opus", ".flac", ".webm", ".wma")

_commits: set[str] = set()          # draft ids with a commit thread in flight
_commits_lock = threading.Lock()


class Window(BaseModel):
    start_s: float = Field(ge=0.0)
    end_s: float = Field(gt=0.0)


class DraftPatch(BaseModel):
    """Step 2 and 3's edits. ``description`` is the Generate button: it is the
    only field whose arrival runs the Writer, so a plain rename never costs the
    DM a regenerated character."""

    transcript: str | None = None
    description: str | None = None
    label: str | None = None
    voice_id: str | None = None
    categories: list[str] | None = None
    # Not in the contract's PATCH list, but step 4 makes every line editable and
    # removable and nothing else carries the edited board back to the server.
    phrases: dict[str, list[str]] | None = None


class PhrasesRequest(BaseModel):
    per_category: int = Field(default=10, ge=1, le=llm.MAX_LINES)


def _view(draft: creator.Draft) -> dict:
    """The draft plus what the page would otherwise have to work out itself."""
    return {**draft.as_dict(), "thin": creator.thin_categories(draft),
            "signature_category": draft.categories[-1] if draft.categories else None}


def _fail(exc: Exception) -> HTTPException:
    """One place where a creator error becomes a status code.

    503 is reserved for "come back when the box is ready" (the brain is not
    resident, whisper did not answer); 502 says the brain answered nonsense, so
    pressing Generate again is worth trying; everything else is the DM's input
    and comes back 400 with the reason in Slovak so the page can print it.
    """
    if isinstance(exc, creator.DraftNotFound):
        return HTTPException(404, "koncept sa nenašiel")
    if isinstance(exc, llm.BrainNotReady):
        return HTTPException(503, BRAIN_DOWN)
    if isinstance(exc, llm.WriterFailed):
        return HTTPException(502, BRAIN_GARBLED)
    if isinstance(exc, creator.SttDown):
        return HTTPException(503, str(exc))
    return HTTPException(400, str(exc))


def _run(call, *args, **kw) -> dict:
    """Call into ``app.creator`` and translate whatever it raises."""
    try:
        return _view(call(*args, **kw))
    except (creator.DraftNotFound, creator.CreatorError, llm.BrainNotReady, llm.WriterFailed,
            ValueError) as exc:
        raise _fail(exc) from exc


def _on_event(app):
    """The worker's own event sink when the service exposes one, else the hub it
    ends in. Either way ``creator.progress`` reaches the browsers on the same
    socket as ``job.done``, which is the point."""
    emit = getattr(app.state, "on_event", None)
    return emit if callable(emit) else app.state.hub.broadcast


# -- the collection ----------------------------------------------------------
# Declared before ``/{draft_id}``: FastAPI matches in order, and "drafts" is a
# perfectly good draft id as far as a path parameter is concerned.


@router.get("/drafts")
def list_drafts() -> list[dict]:
    return [_view(draft) for draft in creator.drafts()]


@router.post("/drafts")
async def post_draft(file: UploadFile = File(...), label: str = Form(...),
                     lang: str = Form("sk")) -> dict:
    """Take the clip and open a draft. The upload is read whole because the cap
    is 25 MB and the window has not been chosen yet, so there is nothing to
    stream it into."""
    if not _is_audio(file):
        raise HTTPException(400, "nahraj zvukový súbor")
    data = await file.read()
    return _run(creator.create, data, file.filename or "upload", label, lang)


def _is_audio(file: UploadFile) -> bool:
    content_type = (file.content_type or "").split(";")[0].strip().lower()
    if content_type.startswith("audio/"):
        return True
    return (file.filename or "").lower().endswith(AUDIO_SUFFIXES)


# -- one draft ---------------------------------------------------------------


@router.get("/{draft_id}")
def get_draft(draft_id: str) -> dict:
    return _run(creator.get, draft_id)


@router.delete("/{draft_id}")
def delete_draft(draft_id: str) -> dict:
    """Idempotent: a draft the DM already discarded on another tablet is not an
    error, it is the state they asked for."""
    try:
        creator.discard(draft_id)
    except creator.DraftNotFound:
        pass
    return {"ok": True}


@router.patch("/{draft_id}")
def patch_draft(draft_id: str, body: DraftPatch) -> dict:
    """Store the DM's edits. ``description`` runs the Writer first, so explicit
    ``categories`` or ``phrases`` in the same request still win over what it
    generated -- the DM's own words are never overwritten by the model's."""
    if body.description is not None:
        _run(creator.describe, draft_id, body.description)
    _run(creator.update, draft_id, transcript=body.transcript, label=body.label,
         voice_id=body.voice_id, categories=body.categories)
    if body.phrases is not None:
        return _run(creator.set_phrases, draft_id, body.phrases)
    return _run(creator.get, draft_id)


@router.post("/{draft_id}/clip")
def post_clip(draft_id: str, body: Window) -> dict:
    return _run(creator.clip, draft_id, body.start_s, body.end_s)


@router.get("/{draft_id}/clip.wav")
def get_clip(draft_id: str) -> FileResponse:
    """The trimmed clip, for the audition player on step 1. ``no-cache`` because
    re-trimming writes a new clip to the same path."""
    draft = _run(creator.get, draft_id)
    path = draft.get("clip_path") or ""
    if not path or not Path(path).is_file():
        raise HTTPException(404, "klip ešte nie je orezaný")
    return FileResponse(path, media_type="audio/wav", headers={"Cache-Control": "no-cache"})


@router.post("/{draft_id}/transcribe")
def post_transcribe(draft_id: str) -> dict:
    return _run(creator.transcribe, draft_id)


@router.post("/{draft_id}/phrases")
def post_phrases(draft_id: str, body: PhrasesRequest) -> dict:
    return _run(creator.write_phrases, draft_id, body.per_category)


@router.post("/{draft_id}/commit")
def post_commit(draft_id: str, request: Request) -> dict:
    """Start the lock -> calibrate -> bank -> queue run and return at once.

    ``job_id`` is the draft's own id: a commit is one per draft, and every
    ``creator.progress`` frame carries ``draft_id``, so that is the handle the
    page already matches on. A second press while one is running is 409 rather
    than a second lock of the same clip.
    """
    draft = _run(creator.get, draft_id)
    try:
        creator.check_voice_id(draft["voice_id"])
    except ValueError as exc:                    # a bad id now, not three GPU minutes later
        raise _fail(exc) from exc
    with _commits_lock:
        if draft_id in _commits:
            raise HTTPException(409, "tento koncept sa práve ukladá")
        _commits.add(draft_id)
    threading.Thread(target=_commit, args=(request.app, draft_id), daemon=True,
                     name=f"creator-commit-{draft_id[:8]}").start()
    return {"job_id": draft_id}


def _commit(app, draft_id: str) -> None:
    """The commit thread. It must not raise: the draft records its own failure
    (``status: failed``) and the exception is logged, because there is no
    request left to answer by the time anything here can go wrong."""
    try:
        result = creator.commit(draft_id, worker=app.state.worker, on_event=_on_event(app))
        # The roster is loaded once at boot, so without this the DM would have to
        # restart the service to see the voice they just made.
        app.state.voices[result["voice_id"]] = voices.load_voice(result["voice_id"])
        log.info("creator: %s committed (%d lines, %d queued)",
                 result["voice_id"], result["lines"], result["queued"])
    except Exception:  # noqa: BLE001 - a dead thread must still release the draft
        log.exception("creator: commit of draft %s failed", draft_id)
    finally:
        with _commits_lock:
            _commits.discard(draft_id)
