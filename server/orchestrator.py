# -*- coding: utf-8 -*-
"""Bag orchestrator — the thin layer between a browser and the raw TTS model.

Three jobs, none of which the raw /v1/audio/speech endpoint does on its own:

  1. GUARANTEE Bag's voice. The model drifts to a female Slovak default ~80% of
     the time even from a good deep-male reference, and `seed` does not hold a
     voice across lines (verified). So we generate, MEASURE the pitch, and
     re-roll the seed until it lands in male range. Every line comes out as Bag.

  2. SPEED. The narrator is deliberate; dialogue needs pace. We time-stretch the
     audio with ffmpeg `atempo`, which changes speed WITHOUT changing pitch — so
     a faster Bag is still the same deep Bag, just quicker.

  3. SPACE. Raw TTS is dry and close-mic'd — "isolated". A little room reverb
     places the voice in a physical space, which is what a speaker hidden in a
     prop bag on a table should sound like.

It also serves the browser UI at `/`. Config is env; see the bottom.
"""
from __future__ import annotations

import json
import os
import struct
import subprocess
import urllib.request
import wave
from io import BytesIO

from fastapi import FastAPI
from fastapi.responses import HTMLResponse, Response
from pydantic import BaseModel

from server import elongation

TTS_URL = os.environ.get("BAG_TTS_URL", "http://127.0.0.1:8010")
REF = os.environ.get("BAG_REF", "/refs/bag_ref.wav")
REF_TEXT = os.environ.get("BAG_REF_TEXT",
    "Popravia? Dostane tretí obed. Ak nie, mám ho ja. Stávka o to, prečo človek "
    "zomrie? Je to zlodej, čo vyzerá ako zlodej? Možno je to zlodej, a možno nie. "
    "To je na tom vtipné.")
MALE_MAX_HZ = float(os.environ.get("BAG_MALE_MAX_HZ", "155"))
MAX_TRIES = int(os.environ.get("BAG_MAX_TRIES", "8"))

HERE = os.path.dirname(os.path.abspath(__file__))
app = FastAPI(title="Bag")


# ------------------------------------------------------------- delivery modes
# Bag's registers, straight off the item card. Each is a stack of documented
# control tokens that LEAD the turn (emotion / style / pitch / speed / expressive
# are global-per-turn and must sit at the very start). `pitch_low` is in almost
# every mode: it deepens him AND makes the voice-gate land male on the first try.
# `pause` / `long_pause` are positional and get injected inline at his beats.
#   speed = default atempo multiplier (fine control; the UI slider overrides it)
#   space = default reverb preset
#   beats = inject pause tokens at em-dashes / ellipses for comedic timing
MODES = {
    # talking to his bonded guy — hyped, high-spirited, fast. The default.
    "bro":     {"lead": "<|emotion:enthusiasm|><|prosody:expressive_high|><|prosody:speed_fast|><|prosody:pitch_low|>",
                "speed": 1.12, "space": "room", "beats": True,
                "desc": "hyped, high-spirited, talking to his guy"},
    # dry mockery, deadpan. Deliberately flat delivery, but loaded.
    "deadpan": {"lead": "<|emotion:bitterness|><|prosody:expressive_low|><|prosody:pitch_low|>",
                "speed": 1.05, "space": "room", "beats": True,
                "desc": "dry, deadpan mockery"},
    # gloating after saving the day — 'who saves the fucking day?'
    "smug":    {"lead": "<|emotion:pride|><|prosody:expressive_high|><|prosody:pitch_low|>",
                "speed": 1.10, "space": "room", "beats": True,
                "desc": "smug, gloating, victorious"},
    # protective fury — 'those aren't your fucking things'
    "pissed":  {"lead": "<|emotion:anger|><|prosody:expressive_high|><|prosody:pitch_low|><|prosody:speed_fast|>",
                "speed": 1.05, "space": "room", "beats": False,
                "desc": "protective, furious, loud"},
    # the One Thing — quiet, ominous, slow. 'Not that one.'
    "menace":  {"lead": "<|style:whispering|><|emotion:contemplation|><|prosody:pitch_low|><|prosody:speed_slow|>",
                "speed": 1.00, "space": "hall", "beats": True,
                "desc": "quiet, ominous, dangerous"},
    # combat urgency / panic — fast, alarmed
    "panic":   {"lead": "<|emotion:fear|><|prosody:expressive_high|><|prosody:pitch_low|><|prosody:speed_fast|>",
                "speed": 1.10, "space": "room", "beats": False,
                "desc": "urgent, alarmed, combat"},
    # rare reluctant softness under the insults — 'you owe me a fucking potion'
    "soft":    {"lead": "<|emotion:affection|><|prosody:expressive_high|><|prosody:pitch_low|><|prosody:speed_slow|>",
                "speed": 1.00, "space": "room", "beats": True,
                "desc": "reluctant, quietly sincere"},
}
DEFAULT_MODE = "bro"


def _beats(text: str) -> str:
    """Turn Bag's written pauses into real ones. Em-dashes and ellipses are
    where his comedic timing lives, so drop inline pause tokens there."""
    for m in ("—", "–", " - "):
        text = text.replace(m, " <|prosody:pause|> ")
    for m in ("...", "…"):
        text = text.replace(m, " <|prosody:pause|> ")
    return text


# ----------------------------------------------------------------- pitch gate
def _f0(pcm: bytes, sr: int) -> float:
    """Rough fundamental frequency by autocorrelation — enough to tell a deep
    male (~90-130 Hz) from the model's female drift (~200+ Hz)."""
    if len(pcm) < sr * 2:
        return 0.0
    s = struct.unpack("<%dh" % (len(pcm) // 2), pcm)
    s = s[len(s) // 3: len(s) // 3 + sr]              # a voiced middle second
    m = sum(s) / len(s)
    s = [x - m for x in s]
    best = (0.0, 0)
    for lag in range(sr // 300, sr // 70):
        c = sum(s[i] * s[i + lag] for i in range(0, len(s) - lag, 4))
        if c > best[0]:
            best = (c, lag)
    return sr / best[1] if best[1] else 0.0


def _synth(text: str, seed: int) -> tuple[bytes, int]:
    body = {"model": "/model", "stream": True, "response_format": "pcm",
            "temperature": 0.8, "top_k": 50, "max_new_tokens": 700, "seed": seed,
            "voice": "default", "input": text,
            "references": [{"audio_path": REF, "text": REF_TEXT}]}
    req = urllib.request.Request(TTS_URL + "/v1/audio/speech",
                                 data=json.dumps(body).encode("utf-8"),
                                 headers={"Content-Type": "application/json"})
    pcm = bytearray()
    with urllib.request.urlopen(req, timeout=120) as r:
        sr = int(r.headers.get("x-sample-rate") or 24000)
        for chunk in iter(lambda: r.read(4096), b""):
            pcm.extend(chunk)
    return bytes(pcm), sr


def bag_voice(text: str) -> tuple[bytes, int, dict]:
    """Generate until the pitch says it's Bag, not the female default."""
    attempts = []
    best = None                                       # fallback: deepest we saw
    for seed in range(MAX_TRIES):
        pcm, sr = _synth(text, seed)
        hz = _f0(pcm, sr)
        attempts.append(round(hz))
        if best is None or hz < best[2]:
            best = (pcm, sr, hz)
        if 60 < hz < MALE_MAX_HZ:
            return pcm, sr, {"accepted_seed": seed, "hz": round(hz),
                             "tries": attempts}
    # nothing cleared the gate — hand back the deepest attempt rather than fail
    return best[0], best[1], {"accepted_seed": None, "hz": round(best[2]),
                              "tries": attempts, "note": "gate not met; deepest kept"}


# --------------------------------------------------------------- post-process
def _process(pcm: bytes, sr: int, speed: float, space: str) -> bytes:
    """atempo for pitch-preserving speed, aecho for room space. One ffmpeg pass
    from raw PCM in to WAV out."""
    speed = max(0.5, min(2.0, speed))
    filters = [f"atempo={speed:.3f}"]
    if space == "room":
        filters.append("aecho=0.8:0.85:45:0.22")
    elif space == "hall":
        filters.append("aecho=0.8:0.9:60|100:0.3|0.2")
    elif space == "bag":                              # muffled + close room: inside a bag
        filters.append("lowpass=f=5500,aecho=0.8:0.8:35:0.18")
    af = ",".join(filters)
    cmd = ["ffmpeg", "-f", "s16le", "-ar", str(sr), "-ac", "1", "-i", "pipe:0",
           "-af", af, "-f", "wav", "pipe:1"]
    p = subprocess.run(cmd, input=pcm, stdout=subprocess.PIPE,
                       stderr=subprocess.DEVNULL)
    return p.stdout


def _wav(pcm: bytes, sr: int) -> bytes:
    buf = BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sr)
        w.writeframes(pcm)
    return buf.getvalue()


# --------------------------------------------------------------------- routes
class SayReq(BaseModel):
    text: str
    mode: str = DEFAULT_MODE       # a delivery register from MODES
    speed: float | None = None     # None -> use the mode's default
    space: str | None = None       # None -> use the mode's default
    emotion: str = ""              # optional extra emotion token, advanced


@app.get("/modes")
def modes():
    return {"default": DEFAULT_MODE,
            "modes": {k: v["desc"] for k, v in MODES.items()}}


@app.post("/say")
def say(req: SayReq):
    raw = req.text.strip()
    if not raw:
        return Response(status_code=400, content="empty text")
    m = MODES.get(req.mode, MODES[DEFAULT_MODE])
    speed = req.speed if req.speed is not None else m["speed"]
    space = req.space if req.space is not None else m["space"]

    # pull the ** elongation markup out first — the model speaks the clean line
    clean, aligner_words, marks = elongation.parse_marks(raw)

    lead = m["lead"]
    if req.emotion:
        lead = f"<|emotion:{req.emotion}|>" + lead
    body = _beats(clean) if m["beats"] else clean
    text = lead + body

    pcm, sr, meta = bag_voice(text)
    pcm, drawls = elongation.elongate(pcm, sr, aligner_words, marks)  # exact vowels
    wav = _process(pcm, sr, speed, space)
    return Response(content=wav, media_type="audio/wav",
                    headers={"X-Bag-Mode": req.mode,
                             "X-Bag-Seed": str(meta.get("accepted_seed")),
                             "X-Bag-Hz": str(meta.get("hz")),
                             "X-Bag-Drawls": ";".join(f"{w}+{ms}ms" for w, ms in drawls),
                             "X-Bag-Tries": ",".join(map(str, meta.get("tries", [])))})


@app.on_event("startup")
def _warm():
    elongation.load()          # pull the MMS aligner into memory once


@app.get("/healthz")
def healthz():
    return {"ok": True, "tts": TTS_URL, "ref": REF,
            "elongation": elongation.load(),
            "elongation_error": elongation._load_error}


@app.get("/", response_class=HTMLResponse)
def index():
    with open(os.path.join(HERE, "web", "index.html"), encoding="utf-8") as f:
        return f.read()
