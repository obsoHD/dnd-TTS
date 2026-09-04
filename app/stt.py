"""Whisper in one place: the gate's verifier and the Voice Creator's transcriber.

Two callers, one endpoint, one failure contract. The gate posts a finished take
to whisper to catch truncation and garbage, and it has to answer inside the
table's budget: a slow verifier must never delay a line, so the gate passes its
own short timeout and reads ``None`` as "unverified", never as "bad". The
creator posts a 30 s reference clip and can afford to wait a minute, because
nobody is mid-scene while a new voice is being built.

That is the whole difference between the two uses, so it is the only thing the
caller passes. Everything else -- the WAV wrapper, the self-signed certificate,
the "``None`` means the STT could not answer" rule -- lives here once, so a
change to the STT (a moved URL, a new field name) is a change to one function
instead of two that drifted apart.

Contract: docs/M4-contracts.md, section ``app/stt.py``.
"""
from __future__ import annotations

import warnings

import requests
import urllib3

from app import store

# The creator's budget: a 30 s clip on a busy box takes tens of seconds. The
# gate does not use this default; it passes its own 2 s table budget.
DEFAULT_TIMEOUT_S = 60.0


def _url() -> str:
    """Resolved at call time so this module imports without the service config
    and so a test's ``BAG_STT_URL`` override is honoured after a reload."""
    from app.config import STT_URL

    return STT_URL


def transcribe(pcm: bytes, sr: int, lang: str, timeout: float = DEFAULT_TIMEOUT_S) -> str | None:
    """What whisper heard in this audio, or ``None`` when it could not answer.

    ``None`` covers every reachability failure -- refused socket, HTTP error,
    a body that is not the JSON the endpoint promises, and the timeout -- because
    no caller can do anything different about them: the gate marks the take
    unverified and the creator tells the DM to type the transcript by hand.
    The endpoint is self-signed by design, hence ``verify=False`` with its
    warning silenced for this call only.
    """
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", urllib3.exceptions.InsecureRequestWarning)
            r = requests.post(_url(), files={"audio": ("take.wav", store.wav_bytes(pcm, sr), "audio/wav")},
                              data={"lang": lang}, timeout=timeout, verify=False)
        r.raise_for_status()
        return str((r.json() or {}).get("text", "")).strip()
    except (requests.RequestException, ValueError):
        return None
