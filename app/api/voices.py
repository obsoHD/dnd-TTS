"""Voices as the table sees them (contract: ``GET /api/voices``, ``GET /api/voices/{id}``).

The roster needs a handful of facts per voice; the Lab needs the whole
``voice.yaml``. Neither needs the persona: it is prompt text for the LLM and
several KB per voice, so it is stripped from the detail view. Voices are the
snapshot loaded at boot (``app.state.voices``); the M1 CLI edits ``voice.yaml``
on disk, and a restart picks that up.
"""
from __future__ import annotations

from dataclasses import asdict

from fastapi import APIRouter, HTTPException, Request

from app import config, voices
from app.voices import Voice

router = APIRouter(prefix="/api/voices")


def load_all() -> dict[str, Voice]:
    """Every ``voices/<id>/voice.yaml`` under DATA_DIR, keyed by id. A broken
    file raises: a voice that cannot load must stop boot, not vanish from the roster."""
    root = config.DATA_DIR / "voices"
    loaded = (voices.load_voice(path.parent.name) for path in sorted(root.glob("*/voice.yaml")))
    return {v.id: v for v in loaded}


def is_locked(v: Voice) -> bool:
    """``lock_reference`` records the clip's sha256; until then there is no
    reference the gate can trust, so the voice cannot render."""
    return bool(v.ref_sha256)


def is_calibrated(v: Voice) -> bool:
    return bool(v.gate.get("calibrated_at"))


def summary(v: Voice) -> dict:
    return {"id": v.id, "label": v.label, "lang": v.lang, "version": v.version,
            "locked": is_locked(v), "calibrated": is_calibrated(v), "energy": v.master.get("energy")}


def detail(v: Voice) -> dict:
    """voice.yaml as JSON minus the persona text."""
    data = asdict(v)
    data.pop("persona", None)
    return data


@router.get("")
def list_voices(request: Request) -> list[dict]:
    return [summary(v) for v in request.app.state.voices.values()]


@router.get("/{voice_id}")
def get_voice(voice_id: str, request: Request) -> dict:
    v = request.app.state.voices.get(voice_id)
    if v is None:
        raise HTTPException(status_code=404, detail="unknown voice")
    return detail(v)
