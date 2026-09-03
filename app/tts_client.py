"""Client for the Higgs TTS server (SGLang-Omni, OpenAI-style ``/v1/audio/speech``).

The server streams raw int16 PCM. Every take is collected whole before it is
returned because the gate (``app.gate``) scores complete takes only; nothing
downstream plays partial audio. All audio here is int16 mono PCM bytes plus a
sample rate, per ``docs/M1-contracts.md``.
"""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass

import requests

DEFAULT_SR = 24_000          # Higgs v3 output rate; the x-sample-rate header wins when present
HEALTH_TIMEOUT_S = 3.0
_SPEECH_PATH = "/v1/audio/speech"
_HEALTH_PATH = "/health"


@dataclass
class Sampler:
    """Sampling knobs, frozen per voice. Defaults come from the Slovak CER A/B in the contract."""

    temperature: float = 0.75
    top_k: int = 40
    top_p: float = 0.95


def _tts_url() -> str:
    """Resolved at call time so this module imports without the service config."""
    from app.config import TTS_URL

    return TTS_URL


def _body(text: str, ref_path: str, ref_text: str, seed: int, sampler: Sampler,
          max_new_tokens: int) -> dict:
    """The request body, sampling always explicit: the server's own defaults are
    T=1.0 with no top_k/top_p, which is far wilder than the calibrated voice."""
    return {
        "model": "/model",
        "stream": True,
        "response_format": "pcm",
        "input": text,
        "references": [{"audio_path": ref_path, "text": ref_text}],
        "temperature": sampler.temperature,
        "top_k": sampler.top_k,
        "top_p": sampler.top_p,
        "seed": seed,
        "max_new_tokens": max_new_tokens,
        "voice": "default",
    }


def synth(text: str, ref_path: str, ref_text: str, seed: int, sampler: Sampler,
          max_new_tokens: int, timeout: float = 120) -> tuple[bytes, int]:
    """One take. ``ref_path`` is the reference as the TTS container sees it
    (``/refs/...``); the caller owns that mapping. Raises on HTTP/transport errors
    so a broken take is never mistaken for a silent one."""
    body = _body(text, ref_path, ref_text, seed, sampler, max_new_tokens)
    with requests.post(_tts_url() + _SPEECH_PATH, json=body, stream=True, timeout=timeout) as r:
        r.raise_for_status()
        sr = int(r.headers.get("x-sample-rate") or DEFAULT_SR)
        pcm = b"".join(r.iter_content(chunk_size=4096))
    return pcm, sr


def synth_many(text: str, ref_path: str, ref_text: str, seeds: list[int], sampler: Sampler,
               max_new_tokens: int) -> list[tuple[int, bytes, int]]:
    """All seeds at once: the server batches concurrent decodes, so N takes cost
    about one (2 takes measured at 1.07 s). Results keep the order of ``seeds``.
    A failed take raises rather than silently shrinking the batch; the render
    layer owns retries."""
    if not seeds:
        return []
    with ThreadPoolExecutor(max_workers=len(seeds)) as pool:
        futures = [pool.submit(synth, text, ref_path, ref_text, seed, sampler, max_new_tokens)
                   for seed in seeds]
        return [(seed, *future.result()) for seed, future in zip(seeds, futures)]


def health() -> bool:
    """True when the TTS answers its health endpoint; any transport failure is
    simply "down" so the status dot never raises."""
    try:
        return requests.get(_tts_url() + _HEALTH_PATH, timeout=HEALTH_TIMEOUT_S).ok
    except requests.RequestException:
        return False
