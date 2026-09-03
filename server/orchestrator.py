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
import re
import struct
import subprocess
import threading
import urllib.request
import wave
from io import BytesIO
from urllib.parse import quote

try:
    import audioop                     # py3.12 (removed in 3.13; see _peak_normalize)
except ImportError:                    # pragma: no cover
    audioop = None

import numpy as np
import pyloudnorm as pyln
import requests
import urllib3
from fastapi import FastAPI, File, Form, UploadFile
from pedalboard import (Compressor, HighpassFilter, Limiter, LowpassFilter, Pedalboard,
                        Reverb, time_stretch)
from fastapi.responses import HTMLResponse, JSONResponse, Response
from pydantic import BaseModel

from server import elongation

urllib3.disable_warnings()      # STT is https with a self-signed cert on the LAN

TTS_URL = os.environ.get("BAG_TTS_URL", "http://127.0.0.1:8010")
LLM_URL = os.environ.get("BAG_LLM_URL", "http://127.0.0.1:11434")
# Qwen3-family 27B (uncensored build, so Bag's swearing isn't moralized). Its
# Slovak is far better than llama3.1-8b's, and it fits on GPU 0 next to whisper.
# Pin it in the broker's OLLAMA_KEEP (scripts/register_broker.sh <model>) or the
# scheduler unloads it whenever the assistant frees VRAM.
LLM_MODEL = os.environ.get("BAG_LLM_MODEL", "huihui_ai/qwen3.8-abliterated:27b")
STT_URL = os.environ.get("BAG_STT_URL", "https://127.0.0.1:8443/stt")
MAX_TRIES = int(os.environ.get("BAG_MAX_TRIES", "8"))
# Auto speed = the mode's native pace (the model's own speed tokens). Only when a
# line is clearly slow (below SLOW wps) do we nudge it toward TARGET, capped 1.12x.
ADAPTIVE_TARGET_WPS = float(os.environ.get("BAG_TARGET_WPS", "2.7"))
ADAPTIVE_SLOW_WPS = float(os.environ.get("BAG_SLOW_WPS", "2.2"))

# --------------------------------------------------------------------- voices
# Each voice is a reference clip + its transcript (transcript materially improves
# cloning), a `pitch` token that shapes timbre, an accept `band` (Hz) for the
# gate so we keep re-rolling until the generation lands in that voice's range,
# and a `persona` that drives the LLM. Add NPCs by dropping a clip in /refs.
BAG_PERSONA = (
    "Si Vak (Mr. Bag) — vedomý, sarkastický a drzý čarovný predmet v hre "
    "Dungeons & Dragons. Inteligencia 12, Múdrosť 14, Charizma 18. Hovoríš po "
    "slovensky, hrubo, s humorom a preklínaním, ako starý kamoš, ktorý všetko "
    "komentuje. Si so svojím majiteľom od narodenia a tváriš sa, že ťa to otravuje, "
    "ale v skutočnosti ti na ňom záleží. Odpovedaj KRÁTKO — jedna až tri vety, "
    "hovorená reč, žiadne odrážky ani javiskové poznámky. Nikdy nevydáš 'ten jeden "
    "predmet' — vždy odmietni slovami 'Ten nie.'")
NPC_PERSONA = (
    "Si postava (NPC) v hre Dungeons & Dragons. Hovoríš po slovensky, stručne a "
    "v úlohe. Odpovedaj KRÁTKO — jedna až tri vety hovorenej reči, bez odrážok.")
SHOPKEEP_PERSONA = (
    "Si ŠIALENÝ, prehnane nadšený kupec v hre Dungeons & Dragons. Hovoríš po "
    "slovensky, hlasno, teatrálne a manicky. Všetko sa snažíš predať, vychvaľuješ "
    "svoj tovar do nebies a smeješ sa vlastným vtipom. Odpovedaj KRÁTKO — jedna až "
    "tri vety hovorenej reči, bez odrážok ani javiskových poznámok.")

# English personas. The prompt must be MONOLINGUAL per request — a Slovak system
# prompt plus "answer in English" makes the small model blend languages.
BAG_PERSONA_EN = (
    "You are Bag (Mr. Bag) — a sentient, sarcastic, foul-mouthed magic item in a "
    "Dungeons & Dragons game. INT 12, WIS 14, CHA 18. You speak English, crude and "
    "funny, like an old buddy who comments on everything. You've been with your "
    "owner since birth and pretend it annoys you, but you care. Answer SHORT — one "
    "to three sentences of spoken dialogue, no bullet points or stage directions. "
    "You never hand over 'the one item' — always refuse with 'Not that one.'")
NPC_PERSONA_EN = (
    "You are an NPC in a Dungeons & Dragons game. You speak English, briefly and "
    "in character. Answer SHORT — one to three sentences of spoken dialogue, no "
    "bullet points.")
SHOPKEEP_PERSONA_EN = (
    "You are a CRAZY, wildly enthusiastic merchant in a Dungeons & Dragons game. "
    "You speak English — loud, theatrical and manic. You try to sell everything, "
    "praise your wares to the skies and laugh at your own jokes. Answer SHORT — one "
    "to three sentences of spoken dialogue, no bullet points or stage directions.")


def _is_en(lang: str) -> bool:
    return (lang or "sk").lower().startswith("en")


def _persona(v: dict, lang: str) -> str:
    return v["persona_en"] if _is_en(lang) else v["persona"]

VOICES = {
    "bag":    {"ref": "/refs/bag_ref.wav", "label": "Mr. Bag (deep male)",
               "pitch": "<|prosody:pitch_low|>", "band": (60, 155),
               "persona": BAG_PERSONA, "persona_en": BAG_PERSONA_EN,
               "text": ("Popravia? Dostane tretí obed. Ak nie, mám ho ja. Stávka o "
                        "to, prečo človek zomrie? Je to zlodej, čo vyzerá ako zlodej? "
                        "Možno je to zlodej, a možno nie. To je na tom vtipné.")},
    "male":   {"ref": "/refs/male-voice.wav", "label": "Adam (male)",
               "pitch": "", "band": (75, 185), "persona": NPC_PERSONA,
               "persona_en": NPC_PERSONA_EN,
               "text": ("Hey, Adam here. Let's create something that feels real, "
                        "sounds human, and connects every time.")},
    "female": {"ref": "/refs/female-voice.wav", "label": "Clara (female)",
               "pitch": "", "band": (150, 290), "persona": NPC_PERSONA,
               "persona_en": NPC_PERSONA_EN,
               "text": ("By repeating what students say, teachers can demonstrate "
                        "that they are listening. By extending what students say.")},
    "shopkeep": {"ref": "/refs/shopkeep_ref.wav", "label": "Crazy Shopkeep (male)",
                 "pitch": "", "band": (105, 255), "persona": SHOPKEEP_PERSONA,
                 "persona_en": SHOPKEEP_PERSONA_EN,
                 "text": ("Why are you guys so anti-dictators? Imagine if America was "
                          "a dictatorship. You could let one percent of the people "
                          "have all the nation's wealth. You could help your rich "
                          "friends get richer by cutting their taxes and bailing them "
                          "out when they gamble and lose. You could ignore the needs "
                          "of the poor for health care and education.")},
}
DEFAULT_VOICE = "bag"

# Phrase board: per-character situation types. Clicking one has the LLM improvise
# a fresh in-character line of that type, then speaks it. Tuned per persona.
PHRASE_TYPES = {
    "bag": {"sk": ["Pozdrav kámoša", "Urážka partie", "Chvastanie po záchrane",
                   "Odmietnutie predmetu", "Bojový pokrik", "Sarkastická poznámka",
                   "Namrzené povzbudenie", "Ten nie."],
            "en": ["Greet your buddy", "Insult the party", "Gloat after saving the day",
                   "Refuse an item", "Battle cry", "Sarcastic remark",
                   "Grumpy encouragement", "Not that one."]},
    "shopkeep": {"sk": ["Vítanie zákazníka", "Tvrdý predaj", "Jednanie o cene",
                        "Nehorázna cena", "Vychvaľovanie tovaru", "Zatváram krám",
                        "Podozrivá ponuka"],
                 "en": ["Welcome a customer", "Hard sell", "Haggle over the price",
                        "Outrageous price", "Praise the wares", "Closing up shop",
                        "Suspicious offer"]},
}
NPC_PHRASE_TYPES = {"sk": ["Pozdrav", "Varovanie", "Klebeta z mesta", "Ponuka úlohy",
                           "Krčmová reč", "Rozlúčka"],
                    "en": ["Greeting", "Warning", "Town gossip", "Quest offer",
                           "Tavern talk", "Farewell"]}

HERE = os.path.dirname(os.path.abspath(__file__))
app = FastAPI(title="Bag")


@app.exception_handler(Exception)
async def _errors(request, exc):
    """Never leak a bare 500: the UI gets a readable reason (LLM down, STT
    unreachable, TTS timeout...) instead of a blank error."""
    return JSONResponse({"error": f"{type(exc).__name__}: {exc}"}, status_code=502)


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
    "bro":     {"lead": "<|emotion:enthusiasm|><|prosody:expressive_high|><|prosody:speed_fast|>",
                "speed": 1.12, "space": "room", "beats": True,
                "desc": "hyped, high-spirited, talking to his guy"},
    # dry mockery, deadpan. Deliberately flat delivery, but loaded.
    "deadpan": {"lead": "<|emotion:bitterness|><|prosody:expressive_low|>",
                "speed": 1.05, "space": "room", "beats": True,
                "desc": "dry, deadpan mockery"},
    # gloating after saving the day — 'who saves the fucking day?'
    "smug":    {"lead": "<|emotion:pride|><|prosody:expressive_high|>",
                "speed": 1.10, "space": "room", "beats": True,
                "desc": "smug, gloating, victorious"},
    # protective fury — 'those aren't your fucking things'
    "pissed":  {"lead": "<|emotion:anger|><|prosody:expressive_high|><|prosody:speed_fast|>",
                "speed": 1.05, "space": "room", "beats": False,
                "desc": "protective, furious, loud"},
    # the One Thing — quiet, ominous, slow. 'Not that one.'
    "menace":  {"lead": "<|style:whispering|><|emotion:contemplation|><|prosody:speed_slow|>",
                "speed": 1.00, "space": "hall", "beats": True,
                "desc": "quiet, ominous, dangerous"},
    # combat urgency / panic — fast, alarmed
    "panic":   {"lead": "<|emotion:fear|><|prosody:expressive_high|><|prosody:speed_fast|>",
                "speed": 1.10, "space": "room", "beats": False,
                "desc": "urgent, alarmed, combat"},
    # rare reluctant softness under the insults — 'you owe me a fucking potion'
    "soft":    {"lead": "<|emotion:affection|><|prosody:expressive_high|><|prosody:speed_slow|>",
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


def _synth(text: str, seed: int, ref: str, ref_text: str) -> tuple[bytes, int]:
    body = {"model": "/model", "stream": True, "response_format": "pcm",
            "temperature": 0.8, "top_k": 50, "max_new_tokens": 700, "seed": seed,
            "voice": "default", "input": text,
            "references": [{"audio_path": ref, "text": ref_text}]}
    req = urllib.request.Request(TTS_URL + "/v1/audio/speech",
                                 data=json.dumps(body).encode("utf-8"),
                                 headers={"Content-Type": "application/json"})
    pcm = bytearray()
    with urllib.request.urlopen(req, timeout=120) as r:
        sr = int(r.headers.get("x-sample-rate") or 24000)
        for chunk in iter(lambda: r.read(4096), b""):
            pcm.extend(chunk)
    return bytes(pcm), sr


_TTS_LOCK = threading.Lock()   # one GPU: one generation (gate loop included) at a time


def voice_gen(text: str, voice: dict) -> tuple[bytes, int, dict]:
    with _TTS_LOCK:
        return _voice_gen_unlocked(text, voice)


def _voice_gen_unlocked(text: str, voice: dict) -> tuple[bytes, int, dict]:
    """Generate until the pitch lands in this voice's band — the same gate that
    keeps Bag from drifting female also keeps a female voice from drifting deep."""
    lo, hi = voice["band"]
    mid = (lo + hi) / 2
    attempts = []
    best = None                                       # fallback: closest to band
    for seed in range(MAX_TRIES):
        pcm, sr = _synth(text, seed, voice["ref"], voice["text"])
        hz = _f0(pcm, sr)
        attempts.append(round(hz))
        dist = abs(hz - mid)
        if best is None or dist < best[3]:
            best = (pcm, sr, hz, dist)
        if lo < hz < hi:
            return pcm, sr, {"accepted_seed": seed, "hz": round(hz),
                             "tries": attempts}
    return best[0], best[1], {"accepted_seed": None, "hz": round(best[2]),
                              "tries": attempts, "note": "gate not met; closest kept"}


# --------------------------------------------------------------- post-process
def _peak_normalize(pcm: bytes, target: float = 0.89) -> bytes:
    """Peak-normalize to ~-1 dBFS. TTS output is already level, so a dynamic
    loudness normalizer (loudnorm) only pumps and squashes it — peak is the
    right tool for speech that will be played through one speaker."""
    n = len(pcm) // 2
    if n == 0:
        return pcm
    if audioop is not None:
        peak = audioop.max(pcm, 2) or 1
        gain = min((32767 * target) / peak, 8.0)
        return audioop.mul(pcm, 2, gain) if abs(gain - 1.0) > 0.02 else pcm
    s = struct.unpack("<%dh" % n, pcm)
    peak = max(1, max(abs(x) for x in s))
    gain = min((32767 * target) / peak, 8.0)
    if abs(gain - 1.0) <= 0.02:
        return pcm
    return struct.pack("<%dh" % n, *(max(-32768, min(32767, int(x * gain))) for x in s))


def _trim(pcm: bytes, sr: int) -> bytes:
    """Trim leading/trailing silence and clamp any runaway internal pause
    (>1.2 s -> 0.7 s). Our 250 ms breaks sit well under that threshold."""
    af = ("silenceremove=start_periods=1:start_threshold=-45dB:start_silence=0.06,"
          "areverse,silenceremove=start_periods=1:start_threshold=-45dB:start_silence=0.12,"
          "areverse,silenceremove=stop_periods=-1:stop_duration=1.2:stop_threshold=-45dB:stop_silence=0.7")
    cmd = ["ffmpeg", "-f", "s16le", "-ar", str(sr), "-ac", "1", "-i", "pipe:0",
           "-af", af, "-f", "s16le", "pipe:1"]
    return subprocess.run(cmd, input=pcm, stdout=subprocess.PIPE,
                          stderr=subprocess.DEVNULL).stdout or pcm


# Real reverbs (Freeverb-class algorithm via pedalboard), not an echo filter.
# Tuned for a voice sitting in a space: low wet levels, plenty of damping.
SPACES = {
    "dry":  [],
    "room": [Reverb(room_size=0.22, damping=0.65, wet_level=0.09, dry_level=0.93, width=0.5)],
    "hall": [Reverb(room_size=0.60, damping=0.45, wet_level=0.20, dry_level=0.86, width=0.9)],
    "bag":  [LowpassFilter(cutoff_frequency_hz=4200),                 # muffled, tiny box
             Reverb(room_size=0.08, damping=0.85, wet_level=0.07, dry_level=0.95, width=0.3)],
}
SPEECH_LUFS = -16.0        # streaming/speech loudness target (EBU R128-style)


def _stretch_np(x: np.ndarray, sr: int, speed: float) -> np.ndarray:
    """Rubber Band time-stretch (pedalboard.time_stretch), pitch-preserving and
    far cleaner than atempo. The library's factor convention is verified at
    runtime: whichever direction shortens the audio is 'faster'."""
    if abs(speed - 1.0) < 0.01:
        return x
    y = time_stretch(x[None, :], sr, stretch_factor=speed)[0]
    if (speed > 1.0) == (len(y) > len(x)):       # convention was inverted
        y = time_stretch(x[None, :], sr, stretch_factor=1.0 / speed)[0]
    return y


def _process(pcm: bytes, sr: int, speed: float, space: str) -> bytes:
    """The finishing chain, on the tools real voice products use:
      1. Rubber Band time-stretch for speed (only when != 1.0)
      2. highpass 80 Hz (rumble) -> gentle compressor (evens out the line)
         -> the room's reverb -> limiter
      3. loudness-normalize to -16 LUFS (pyloudnorm, ITU-R BS.1770) then a
         true-peak limiter at -1 dBFS, so every voice/mode lands at one level."""
    x = np.frombuffer(pcm, dtype=np.int16).astype(np.float32) / 32768.0
    if x.size == 0:
        return _wav(pcm, sr)
    x = _stretch_np(x, sr, max(0.5, min(2.0, speed)))

    board = Pedalboard([HighpassFilter(cutoff_frequency_hz=80),
                        Compressor(threshold_db=-18, ratio=2.5, attack_ms=5, release_ms=90),
                        *SPACES.get(space, SPACES["room"]),
                        Limiter(threshold_db=-1.0)])
    y = board(x[None, :], sr)[0]

    if len(y) >= int(0.5 * sr):                  # the meter needs ~0.4 s of audio
        lufs = pyln.Meter(sr).integrated_loudness(y.astype(np.float64))
        if np.isfinite(lufs):
            y = pyln.normalize.loudness(y, lufs, SPEECH_LUFS).astype(np.float32)
    # Deterministic true-peak ceiling at -1 dBFS. pedalboard's Limiter is not a
    # brickwall (transients overshoot, then np.clip distorts), so if the loudness
    # gain pushed peaks past the ceiling, scale down: loudness target only within
    # the peak ceiling — the broadcast convention.
    peak = float(np.max(np.abs(y))) if y.size else 0.0
    ceiling = 10 ** (-1.0 / 20)
    if peak > ceiling:
        y = y * (ceiling / peak)
    out = (np.clip(y, -1.0, 1.0) * 32767).astype(np.int16).tobytes()
    return _wav(out, sr)


def _wav(pcm: bytes, sr: int) -> bytes:
    buf = BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sr)
        w.writeframes(pcm)
    return buf.getvalue()


# ------------------------------------------------------------------- pipeline
_SENT = re.compile(r"(?<=[.!?…])\s+")


def _chunks(text: str, max_len: int = 220, min_len: int = 40) -> list[str]:
    """Split long text at sentence ends into generation-sized chunks. Short text
    is one chunk. Tiny trailing fragments are merged into their neighbour."""
    text = text.strip()
    if len(text) <= max_len:
        return [text]
    parts, cur = [], ""
    for s in _SENT.split(text):
        if cur and len(cur) + 1 + len(s) > max_len:
            parts.append(cur)
            cur = s
        else:
            cur = (cur + " " + s).strip()
    if cur:
        parts.append(cur)
    merged: list[str] = []
    for p in parts:
        if merged and len(p) < min_len:
            merged[-1] += " " + p
        else:
            merged.append(p)
    return merged


def _render(text: str, voice_key: str, mode_key: str,
            speed=None, space=None, emotion="") -> tuple[bytes, dict, list]:
    """text (with ** markup) -> the chosen voice, in the chosen delivery mode,
    with exact-vowel drawls and speed/space shaping. The whole TTS path."""
    v = VOICES.get(voice_key, VOICES[DEFAULT_VOICE])
    m = MODES.get(mode_key, MODES[DEFAULT_MODE])
    spc = space if space is not None else m["space"]
    lead = v["pitch"] + m["lead"]            # voice sets timbre, mode sets delivery
    if emotion:
        lead = f"<|emotion:{emotion}|>" + lead

    # Long text renders sentence-by-sentence: each generation stays short and
    # stable and the model's runaway on long inputs is bounded. Markup (** and
    # TAB/—/… breaks) is parsed per chunk so it still lands where written; the
    # model speaks a clean line and breaks become short deterministic silences.
    parts, drawls, meta, sr, total_words = [], [], {}, 24000, 0
    for chunk in _chunks(text):
        clean, aligner_words, marks, breaks = elongation.parse_marks(chunk)
        if not clean:
            continue
        pcm, sr, cmeta = voice_gen(lead + clean, v)
        pcm, applied = elongation.elongate(pcm, sr, aligner_words, marks, breaks)
        drawls += applied
        total_words += len(clean.split())
        meta = meta or cmeta
        parts.append(pcm)
    if not parts:
        raise ValueError("nothing to say")
    meta["chunks"] = len(parts)
    pcm = (b"\x00\x00" * int(0.25 * sr)).join(parts)   # one break's worth between chunks

    # adaptive speed: nudge the delivery toward a natural dialogue pace from how
    # fast the model actually spoke (words/sec) — not a fixed multiplier.
    # An explicit speed from the UI overrides.
    pcm = _trim(pcm, sr)                    # silence trim + runaway-pause clamp first
    if speed is None:
        # auto = the mode's native pace; a gentle nudge only when clearly slow
        dur = max(0.2, len(pcm) / (2 * sr))
        wps = max(1, total_words) / dur
        sp = 1.0 if wps >= ADAPTIVE_SLOW_WPS else min(1.12, ADAPTIVE_TARGET_WPS / wps)
        meta["adaptive_speed"] = round(sp, 2)
    else:
        sp = speed
    return _process(pcm, sr, sp, spc), meta, drawls


def _audio_response(wav, meta, drawls, extra=None) -> Response:
    headers = {"X-Bag-Hz": str(meta.get("hz")),
               "X-Bag-Tries": ",".join(map(str, meta.get("tries", []))),
               "X-Bag-Speed": str(meta.get("adaptive_speed", "")),
               "X-Bag-Chunks": str(meta.get("chunks", 1)),
               "X-Bag-Drawls": ";".join(f"{w}+{ms}ms" for w, ms in drawls)}
    if extra:
        headers.update(extra)
    return Response(content=wav, media_type="audio/wav", headers=headers)


def _llm_reply(user_text: str, persona: str, history=None,
               temperature=0.85, num_predict=180) -> str:
    """The LLM via ollama, in persona, kept short. Warm-pinned via keep_alive."""
    msgs = [{"role": "system", "content": persona}]
    for h in (history or [])[-8:]:
        if h.get("role") in ("user", "assistant") and h.get("content"):
            msgs.append({"role": h["role"], "content": h["content"]})
    msgs.append({"role": "user", "content": user_text})
    txt = ""
    for attempt in range(2):                     # the model occasionally returns ""
        r = requests.post(LLM_URL + "/api/chat", timeout=120, json={
            "model": LLM_MODEL, "messages": msgs, "stream": False, "keep_alive": "30m",
            "think": False,             # Qwen3: no reasoning trace, just the line
            "options": {"temperature": temperature if attempt == 0 else 0.7,
                        "num_predict": num_predict,
                        "num_ctx": 4096}})   # short lines; a 40k ctx wastes ~8GB VRAM
        r.raise_for_status()
        txt = (r.json().get("message", {}) or {}).get("content", "").strip()
        txt = re.sub(r"<think>.*?</think>", "", txt, flags=re.S).strip('"').strip()
        if txt:
            break
    return txt


def _stt(audio: bytes, filename: str, lang: str = "sk") -> str:
    r = requests.post(STT_URL, timeout=120, verify=False,
                      files={"audio": (filename or "clip.webm", audio)},
                      data={"lang": lang})
    r.raise_for_status()
    return (r.json() or {}).get("text", "").strip()


# --------------------------------------------------------------------- routes
class SayReq(BaseModel):
    text: str
    voice: str = DEFAULT_VOICE
    mode: str = DEFAULT_MODE
    speed: float | None = None
    space: str | None = None
    emotion: str = ""


class RespondReq(BaseModel):
    text: str
    voice: str = DEFAULT_VOICE
    mode: str = DEFAULT_MODE
    lang: str = "sk"
    speed: float | None = None
    space: str | None = None
    history: list = []


@app.get("/modes")
def modes():
    return {"default": DEFAULT_MODE,
            "modes": {k: v["desc"] for k, v in MODES.items()}}


@app.get("/voices")
def voices():
    return {"default": DEFAULT_VOICE,
            "voices": {k: v["label"] for k, v in VOICES.items()}}


@app.post("/say")
def say(req: SayReq):
    if not req.text.strip():
        return Response(status_code=400, content="empty text")
    wav, meta, drawls = _render(req.text.strip(), req.voice, req.mode,
                                req.speed, req.space, req.emotion)
    return _audio_response(wav, meta, drawls, {"X-Bag-Mode": req.mode})


@app.post("/respond")
def respond(req: RespondReq):
    """DM/player types -> Bag (in persona) answers, spoken."""
    heard = req.text.strip()
    if not heard:
        return Response(status_code=400, content="empty text")
    v = VOICES.get(req.voice, VOICES[DEFAULT_VOICE])
    reply = _llm_reply(heard, _persona(v, req.lang), req.history)
    wav, meta, drawls = _render(reply, req.voice, req.mode, req.speed, req.space)
    return _audio_response(wav, meta, drawls,
                           {"X-Bag-Heard": quote(heard), "X-Bag-Reply": quote(reply)})


def _lang_rule(lang: str) -> str:
    return ("Odpovedaj po anglicky." if (lang or "sk").lower().startswith("en")
            else "Odpovedaj po slovensky.")


class LineReq(BaseModel):
    voice: str = DEFAULT_VOICE
    mode: str = DEFAULT_MODE
    type: str = ""                 # a phrase-type from PHRASE_TYPES
    lang: str = "sk"
    speed: float | None = None
    space: str | None = None


@app.get("/phrases")
def phrases(voice: str = DEFAULT_VOICE, lang: str = "sk"):
    key = "en" if _is_en(lang) else "sk"
    types = PHRASE_TYPES.get(voice, NPC_PHRASE_TYPES)
    return {"voice": voice, "lang": key, "types": types[key]}


def _improv_line(voice_key: str, kind: str, lang: str) -> str:
    """One improvised in-character line. Monolingual prompt per language, with
    the model asked to place one or two natural short pauses as em dashes —
    those become short deterministic silences downstream."""
    v = VOICES.get(voice_key, VOICES[DEFAULT_VOICE])
    if _is_en(lang):
        prompt = (f"Say ONE short line. Situation or type: {kind or 'a line'}. "
                  f"Reply with ONLY the line, one or two sentences, in character, "
                  f"no quotes. Put one or two natural short pauses as an em dash (—) "
                  f"where the character would hesitate or breathe. English only.")
    else:
        prompt = (f"Povedz JEDNU krátku repliku. Situácia alebo typ: {kind or 'replika'}. "
                  f"Odpovedz IBA replikou, jedna až dve vety, v úlohe, bez úvodzoviek. "
                  f"Vlož jednu až dve prirodzené krátke pauzy ako pomlčku (—) tam, "
                  f"kde by postava zaváhala alebo sa nadýchla. Len po slovensky.")
    text = _llm_reply(prompt, _persona(v, lang), None, temperature=0.8, num_predict=90)
    return text.strip().strip('"').split("\n")[0].strip()


@app.post("/linetext")
def linetext(req: LineReq):
    """Click a phrase-type -> the LLM improvises a line. TEXT ONLY, so it lands in
    the box to read; the user triggers Speak themselves."""
    return {"text": _improv_line(req.voice, (req.type or "").strip(), req.lang)}


@app.post("/line")
def line(req: LineReq):
    """Improvise a line AND speak it (kept for callers that want one shot)."""
    text = _improv_line(req.voice, (req.type or "").strip(), req.lang)
    wav, meta, drawls = _render(text, req.voice, req.mode, req.speed, req.space)
    return _audio_response(wav, meta, drawls, {"X-Bag-Line": quote(text)})


class FixReq(BaseModel):
    text: str
    lang: str = "sk"


@app.post("/fix")
def fix(req: FixReq):
    """Clean up typos/grammar without changing meaning, slang, or the ** markup."""
    t = req.text.strip()
    if not t:
        return {"text": ""}
    if (req.lang or "sk").lower().startswith("en"):
        system = (
            "You are an automatic proofreader for a game. Fix ONLY typos, spelling "
            "and punctuation. Do NOT change words, meaning or style. Keep slang and "
            "profanity as-is (they are character lines). The asterisks (**) are "
            "markers — keep them EXACTLY where and how many they are. NEVER refuse or "
            "comment — return ONLY the corrected text.\n"
            "Example: input 'heey braa**cho whats up man' -> output 'Heey, braa**cho, what's up, man?'")
    else:
        system = (
            "Si automatický korektor slovenského textu pre hru. Opravuj IBA preklepy, "
            "diakritiku a interpunkciu. NEMEŇ slová, význam ani štýl. Slang a hovorové "
            "slová (brácho, kámo, čávo, hej) NECHAJ PRESNE TAK, neprepisuj ich na "
            "spisovné. Vulgarizmy nechaj — sú to repliky postáv. Hviezdičky (**) sú "
            "značky a musíš ich nechať PRESNE tam a v presnom počte ako sú. NIKDY "
            "neodmietni ani nekomentuj — vráť LEN opravený text.\n"
            "Príklad: vstup 'brá**cho co ti dava kamo' -> výstup 'Brá**cho, čo ti dáva, kámo?'")
    try:
        fixed = _llm_reply(t, system, None, temperature=0.2, num_predict=200)
    except Exception as e:  # noqa: BLE001
        return {"text": t, "error": str(e)}
    # the small model sometimes prefixes chatter like "Výstup: ..." — salvage it
    for marker in ("Výstup:", "výstup:", "Output:", "OUTPUT:"):
        if marker in fixed:
            fixed = fixed.split(marker)[-1]
    fixed = fixed.strip().strip('"').strip()
    return {"text": fixed or t}


@app.post("/converse")
async def converse(audio: UploadFile = File(...), voice: str = Form(DEFAULT_VOICE),
                   mode: str = Form(DEFAULT_MODE), lang: str = Form("sk")):
    """Push-to-talk: speak -> Whisper -> Bag answers, spoken. One round trip."""
    heard = _stt(await audio.read(), audio.filename, lang)
    if not heard:
        return JSONResponse({"heard": "", "reply": "", "error": "no speech"},
                            status_code=200)
    v = VOICES.get(voice, VOICES[DEFAULT_VOICE])
    reply = _llm_reply(heard, _persona(v, lang))
    wav, meta, drawls = _render(reply, voice, mode)
    return _audio_response(wav, meta, drawls,
                           {"X-Bag-Heard": quote(heard), "X-Bag-Reply": quote(reply)})


@app.on_event("startup")
def _warm():
    elongation.load()          # pull the MMS aligner into memory once


@app.get("/healthz")
def healthz():
    return {"ok": True, "tts": TTS_URL, "llm": LLM_MODEL, "stt": STT_URL,
            "voices": list(VOICES), "elongation": elongation.load(),
            "elongation_error": elongation._load_error}


@app.get("/", response_class=HTMLResponse)
def index():
    with open(os.path.join(HERE, "web", "index.html"), encoding="utf-8") as f:
        return f.read()
