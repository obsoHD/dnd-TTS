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

import collections
import difflib
import hashlib
import json
import os
import random
import re
import struct
import subprocess
import threading
import time
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
BROKER_URL = os.environ.get("BAG_BROKER_URL", "http://127.0.0.1:8090")   # lifeos VRAM broker
MAX_TRIES = int(os.environ.get("BAG_MAX_TRIES", "8"))
# ASR round-trip quality gate: whisper re-transcribes each rendered chunk; if
# the character error rate vs the intended text is above the threshold the
# line is re-rolled once (catches garbled, looping or truncated generations).
ASR_GATE = os.environ.get("BAG_ASR_GATE", "1") == "1"
ASR_CER_MAX = float(os.environ.get("BAG_ASR_CER_MAX", "0.35"))
# Adaptive speed: the pace of the speech itself is measured (deliberate drawls,
# breaks and chunk gaps subtracted) and pulled toward the delivery mode's target
# words/sec — slow lines speed up, rushed lines ease off. Fallback target here.
ADAPTIVE_TARGET_WPS = float(os.environ.get("BAG_TARGET_WPS", "2.8"))

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

# Slovak quality rules + native few-shot examples: mid-size models measurably
# improve with native-language demonstrations and explicit case/agreement and
# anti-bohemism instructions. Appended to every Slovak persona.
SK_RULES = (
    " Píš výhradne spisovnou slovenčinou: dodržiavaj pády a zhodu prídavného mena s "
    "podstatným menom, žiadne bohemizmy (vždyť, doporučiť, tady, jelikož, prostě), "
    "žiadne anglické kalky. Príklady správnych replík: „Jasné, ja to vyriešim. Ako "
    "vždy.“ „Toto? To ti nedám, kamoš. Ani náhodou.“ „Máš tridsať životov. Prestaň "
    "fňukať a bojuj.“ „Kto zachránil deň? No predsa ja.“ „Tak poď, nemám na to celý deň.“")
SK_FIX_SYSTEM = (
    "Si korektor spisovnej slovenčiny. Oprav IBA gramatiku, pády, zhodu a bohemizmy. "
    "Nemeň štýl, vulgarizmy, pomlčky, hviezdičky ani význam; rob čo najmenšie zmeny. "
    "Ak je text správny, vráť ho nezmenený. Vráť LEN opravený text, nič iné.")
BAG_PERSONA += SK_RULES
NPC_PERSONA += SK_RULES
SHOPKEEP_PERSONA += SK_RULES

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
               # cloned from a deliberate audiobook narrator: he inherits that
               # pace, so bias him faster with the model's own speed tokens
               "pace": {"default": "<|prosody:speed_fast|>",
                        "fast": "<|prosody:speed_very_fast|>"},
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

# Curated phrase bank: natively written lines per character x type, each
# approved by two independent judges (grammar, type-fit, character). The board
# serves these first — instant and reliable; the LLM improvises only as a
# fallback, seeded with bank lines as examples. {"sk": {"bag": {type: [..]}}}
PHRASES_PATH = os.path.join(HERE, "phrases.json")
try:
    with open(PHRASES_PATH, encoding="utf-8") as _f:
        PHRASE_BANK = json.load(_f)
except Exception:                                   # noqa: BLE001
    PHRASE_BANK = {"sk": {}, "en": {}}
BANK_KEY = {"bag": "bag", "shopkeep": "shopkeep", "male": "npc", "female": "npc"}
_last_served: dict = {}


def _bank_lines(voice_key: str, kind: str, lang: str) -> list:
    bank = PHRASE_BANK.get("en" if _is_en(lang) else "sk", {})
    return list(bank.get(BANK_KEY.get(voice_key, "npc"), {}).get(kind, []))


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

# Target ARTICULATION rate per delivery mode, syllables/sec over speech-only
# time. Anchored on phonetics: Czech/Slovak conversational 5-6 syl/s (Slovak has
# no published figure; Czech is the accepted proxy), slow/menacing 3.5-4.5,
# hyped/furious/panic 6.5-7.5; English runs ~15% lower. Words/sec was wrong for
# Slovak's long words and pinned every line at the cap.
MODE_SPS = {"bro": 6.0, "deadpan": 5.4, "smug": 5.6, "pissed": 6.4,
            "menace": 4.0, "panic": 6.6, "soft": 4.2}
EN_RATE_SCALE = 0.85
_VOW = "aeiouyáéíóúýäô"
_SYL_DIPH = re.compile(r"i[aeu]|ô")
_SYL_RL = re.compile(r"(?:^|[^aeiouyáéíóúýäô\W])[rlŕĺ](?=[^aeiouyáéíóúýäô\W]|$)")


def _syllables(text: str) -> int:
    """Rough syllable count for Slovak/English: vowel nuclei, diphthongs
    (ia ie iu ô) counted once, syllabic r/l between consonants."""
    n = 0
    for w in re.findall(r"[a-záéíóúýäôčšžťďňľŕĺ']+", text.lower()):
        v = sum(1 for ch in w if ch in _VOW) - len(_SYL_DIPH.findall(w))
        rl = len(_SYL_RL.findall(w))
        n += (v + rl) if v > 0 else (rl or 1)
    return max(1, n)


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


def _synth(text: str, seed: int, ref: str, ref_text: str, max_tokens: int = 700) -> tuple[bytes, int]:
    # the server's own defaults are T=1.0 with NO top_k/top_p (unfiltered) and
    # no repetition penalty at all — always send explicit, tamer sampling
    body = {"model": "/model", "stream": True, "response_format": "pcm",
            "temperature": 0.8, "top_k": 40, "top_p": 0.95,
            "max_new_tokens": max_tokens, "seed": seed,
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


def voice_gen(text: str, voice: dict, max_tokens: int = 700,
              seed_hint=None) -> tuple[bytes, int, dict]:
    with _TTS_LOCK:
        return _voice_gen_unlocked(text, voice, max_tokens, seed_hint)


def _voice_gen_unlocked(text: str, voice: dict, max_tokens: int = 700,
                        seed_hint=None) -> tuple[bytes, int, dict]:
    """Generate until the pitch lands in this voice's band — the same gate that
    keeps Bag from drifting female also keeps a female voice from drifting deep.
    A seed that already worked (seed_hint) is tried first: seeds are
    deterministic here, so reusing one keeps the timbre stable across chunks."""
    lo, hi = voice["band"]
    mid = (lo + hi) / 2
    attempts = []
    best = None                                       # fallback: closest to band
    seeds = list(range(MAX_TRIES))
    if seed_hint is not None:
        seeds = [seed_hint] + [s for s in seeds if s != seed_hint]
    for seed in seeds:
        pcm, sr = _synth(text, seed, voice["ref"], voice["text"], max_tokens)
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


# max internal pause kept per delivery mode (s): hyped/furious modes snap,
# menacing/soft modes are allowed to breathe
MODE_PAUSE = {"bro": 0.30, "deadpan": 0.40, "smug": 0.35, "pissed": 0.28,
              "menace": 0.60, "panic": 0.25, "soft": 0.55}


def _trim(pcm: bytes, sr: int, keep_pause: float = 0.4) -> bytes:
    """Trim leading/trailing silence and tighten internal pauses for dialogue:
    the model parks long silences at commas and periods (~40% of a clip), so
    any internal pause longer than keep_pause+0.1 s is shortened to keep_pause —
    still a clear beat, no dead air. Gentle head/tail thresholds so quiet word
    endings survive, and 300 ms of tail kept so the last word's decay stays."""
    af = ("silenceremove=start_periods=1:start_threshold=-55dB:start_silence=0.08,"
          "areverse,silenceremove=start_periods=1:start_threshold=-55dB:start_silence=0.30,"
          f"areverse,silenceremove=stop_periods=-1:stop_duration={keep_pause + 0.1:.2f}"
          f":stop_threshold=-40dB:stop_silence={keep_pause:.2f}")
    cmd = ["ffmpeg", "-f", "s16le", "-ar", str(sr), "-ac", "1", "-i", "pipe:0",
           "-af", af, "-f", "s16le", "pipe:1"]
    return subprocess.run(cmd, input=pcm, stdout=subprocess.PIPE,
                          stderr=subprocess.DEVNULL).stdout or pcm


def _speech_seconds(pcm: bytes, sr: int, floor_db: float = -40.0) -> float:
    """Seconds of actual speech: 20 ms frames whose RMS is above floor_db.
    Pauses (the model's own and our breaks) are excluded, so a rate computed
    over this is an articulation rate, not a words-over-silence rate."""
    x = np.frombuffer(pcm, dtype=np.int16).astype(np.float32) / 32768.0
    hop = int(sr * 0.02)
    if x.size < hop:
        return x.size / sr
    n = x.size // hop
    rms = np.sqrt(np.mean(x[:n * hop].reshape(n, hop) ** 2, axis=1) + 1e-12)
    return float(np.count_nonzero(20 * np.log10(rms) > floor_db)) * hop / sr


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
                        Compressor(threshold_db=-20, ratio=1.8, attack_ms=8, release_ms=120),
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
    # gentle fade-in and a real fade-out over the reverb tail: no hard ending
    fi, fo = int(0.008 * sr), int(0.07 * sr)
    if y.size > fi + fo:
        y[:fi] *= np.linspace(0.0, 1.0, fi, dtype=np.float32)
        y[-fo:] *= np.linspace(1.0, 0.0, fo, dtype=np.float32)
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


def _cue(sentence: str) -> float:
    """What this sentence IS nudges its pace: exclamations and punchy short
    lines faster, hesitation (…/—) and long descriptive sentences slower,
    SHOUTED words faster."""
    st = sentence.strip()
    f = 1.0
    if st.endswith("!"):
        f *= 1.08
    elif st.endswith("?"):
        f *= 1.03
    if "…" in st or "..." in st or "—" in st or "\t" in st:
        f *= 0.94
    n = len(st.split())
    if n <= 3:
        f *= 1.05
    elif n >= 14:
        f *= 0.96
    if any(w.isupper() and len(w) > 2 for w in st.split()):
        f *= 1.06
    return f


def _pace_sentences(chunk: str, pcm: bytes, sr: int, mode_key: str, applied,
                    lang: str = "sk") -> tuple[bytes, list, list]:
    """One generation, per-sentence pace: align once, measure each sentence's
    articulation rate (syllables over its own speech-only time, drawl holds
    excluded) and stretch just that sentence toward the mode target adjusted by
    its cues. Joins land in the pauses between sentences, so they are silent."""
    sents = [s for s in _SENT.split(chunk) if s.strip()]
    counts = [len(elongation.parse_marks(s)[1]) for s in sents]
    _, all_words, _, _ = elongation.parse_marks(chunk)
    if sum(counts) != len(all_words) or not all_words:
        return pcm, [], []
    bounds = elongation.word_bounds(pcm, sr, all_words)
    if len(bounds) != len(all_words):
        return pcm, [], []

    x = np.frombuffer(pcm, dtype=np.int16)
    holds = [(lbl, ms) for lbl, ms in applied if not lbl.endswith("|")]
    out, pos, idx, factors, rates = [], 0, 0, [], []
    target_base = MODE_SPS.get(mode_key, 5.4) * (EN_RATE_SCALE if _is_en(lang) else 1.0)
    for s, c in zip(sents, counts):
        if c == 0:
            continue
        a = int(bounds[idx][0] * sr)
        b = int(bounds[idx + c - 1][1] * sr)
        words_here = elongation.parse_marks(s)[1]
        idx += c
        seg = x[a:b]
        held = 0.0
        for w in words_here:                       # this sentence's drawl holds
            for k, (lbl, ms) in enumerate(holds):
                if lbl == w:
                    held += ms / 1000.0
                    holds.pop(k)
                    break
        syl = _syllables(s.replace("*", ""))
        speech = _speech_seconds(seg.tobytes(), sr) - held
        if syl < 3 or speech < 0.4:                # too short to measure reliably
            f, sps = 1.0, 0.0
        else:
            sps = syl / speech
            # the model's own speed tokens carry the pace; post-stretch nudges.
            # Speeding up is far more tolerant than slowing down, so allow
            # +18% up but only -6% down; small dead-band.
            f = max(0.94, min(1.22, target_base * _cue(s) / sps))
            if 0.97 <= f <= 1.03:
                f = 1.0
        y = seg.astype(np.float32) / 32768.0
        if f != 1.0:
            y = _stretch_np(y, sr, f)
        out.append(x[pos:a])
        out.append((np.clip(y, -1.0, 1.0) * 32767).astype(np.int16))
        pos = b
        factors.append(f)
        rates.append(round(sps, 2))
    out.append(x[pos:])
    return np.concatenate(out).tobytes(), factors, rates


_TOKEN = re.compile(r"<\|[^|]*\|>")


def _spoken(text: str) -> str:
    """The text without control tokens (for counting and CER)."""
    return " ".join(_TOKEN.sub(" ", text).split())


def _place_tags(lead: str, clean: str) -> str:
    """Documented placement: delivery tokens (emotion/style/pitch/speed/
    expressive) lead the turn, before any text; positional tokens (pause,
    long_pause) stay inline where they fall. Drift is handled by the pitch
    gate + pitch_low, not by moving the tags off the start."""
    return lead + clean


def _render(text: str, voice_key: str, mode_key: str,
            speed=None, space=None, emotion="", lang: str = "sk") -> tuple[bytes, dict, list]:
    """text (with ** markup) -> the chosen voice, in the chosen delivery mode,
    with exact-vowel drawls and speed/space shaping. The whole TTS path."""
    v = VOICES.get(voice_key, VOICES[DEFAULT_VOICE])
    m = MODES.get(mode_key, MODES[DEFAULT_MODE])
    spc = space if space is not None else m["space"]
    mlead = m["lead"]
    pace = v.get("pace")
    if pace:                                 # per-voice pace bias (slow narrator clones)
        if "<|prosody:speed_fast|>" in mlead:
            mlead = mlead.replace("<|prosody:speed_fast|>", pace["fast"])
        elif "<|prosody:speed_" not in mlead:
            mlead += pace["default"]
    lead = v["pitch"] + mlead                # voice sets timbre, mode sets delivery
    if emotion:
        lead = f"<|emotion:{emotion}|>" + lead

    # Long text renders sentence-by-sentence: each generation stays short and
    # stable and the model's runaway on long inputs is bounded. Markup (** and
    # TAB/—/… breaks) is parsed per chunk so it still lands where written; the
    # model speaks a clean line and breaks become short deterministic silences.
    parts, drawls, meta, sr, factors, rates, seed_hint = [], [], {}, 24000, [], [], None
    for chunk in _chunks(text):
        clean, aligner_words, marks, breaks = elongation.parse_marks(chunk)
        if not clean:
            continue
        # per-line token cap: a held vowel gets truncated instead of running 10 s
        # (25 codec frames/s; est. 4.5 syl/s plus the deliberate holds)
        est = (_syllables(_spoken(clean)) / 4.5 + sum(m[3] for m in marks) * 0.15
               + 0.6 * clean.count("<|prosody:") + 1.0)
        max_tokens = max(150, min(900, int(est * 25 * 1.6) + 60))
        pcm, sr, cmeta = voice_gen(_place_tags(lead, clean), v, max_tokens, seed_hint)
        if ASR_GATE:
            cer = _asr_cer(pcm, sr, clean, lang)
            if cer > ASR_CER_MAX:                     # garbled: one re-roll, keep the better
                alt = ((cmeta.get("accepted_seed") or 0) + 1) % MAX_TRIES
                pcm2, sr2, cmeta2 = voice_gen(_place_tags(lead, clean), v, max_tokens, alt)
                cer2 = _asr_cer(pcm2, sr2, clean, lang)
                if cer2 < cer:
                    pcm, sr, cmeta, cer = pcm2, sr2, cmeta2, cer2
                cmeta["rerolled"] = True
            cmeta["cer"] = cer
        seed_hint = cmeta.get("accepted_seed", seed_hint)   # same seed across chunks
        pcm, applied = elongation.elongate(pcm, sr, aligner_words, marks, breaks)
        drawls += applied
        meta = meta or cmeta
        if speed is None:
            # context-aware pacing: each sentence measured and stretched on its own
            pcm, f, r = _pace_sentences(chunk, pcm, sr, mode_key, applied, lang)
            factors += f
            rates += r
        parts.append(pcm)
    if not parts:
        raise ValueError("nothing to say")
    meta["chunks"] = len(parts)
    pcm = _trim((b"\x00\x00" * int(0.25 * sr)).join(parts), sr,
                MODE_PAUSE.get(mode_key, 0.4))

    if speed is None:
        meta["adaptive_speed"] = round(float(np.mean(factors)), 2) if factors else 1.0
        meta["sps"] = round(float(np.mean(rates)), 2) if rates else 0.0
        meta["pace"] = ",".join(f"{x:.2f}" for x in factors)
        sp = 1.0                                   # already paced per sentence
    else:
        sp = speed
    return _process(pcm, sr, sp, spc), meta, drawls


def _audio_response(wav, meta, drawls, extra=None) -> Response:
    headers = {"X-Bag-Hz": str(meta.get("hz")),
               "X-Bag-Tries": ",".join(map(str, meta.get("tries", []))),
               "X-Bag-Speed": str(meta.get("adaptive_speed", "")),
               "X-Bag-Chunks": str(meta.get("chunks", 1)),
               "X-Bag-Sps": str(meta.get("sps", "")),
               "X-Bag-Pace": str(meta.get("pace", "")),
               "X-Bag-Cer": str(meta.get("cer", "")),
               "X-Bag-Drawls": ";".join(f"{w}+{ms}ms" for w, ms in drawls)}
    if extra:
        headers.update(extra)
    return Response(content=wav, media_type="audio/wav", headers=headers)


_llm_check = {"t": 0.0}


def _llm_on_gpu() -> bool:
    """Is Bag's model loaded AND (>=90%) GPU-resident? Another app's model can
    displace it to CPU, where a 27B answers in 50-90 s instead of <1 s."""
    try:
        ps = requests.get(LLM_URL + "/api/ps", timeout=5).json().get("models", [])
    except Exception:                                   # noqa: BLE001
        return True                                     # can't tell; don't block
    for m in ps:
        if LLM_MODEL in (m.get("name"), m.get("model")):
            return m.get("size_vram", 0) >= 0.9 * max(1, m.get("size", 1))
    return False


def _ensure_llm_gpu():
    """Work WITH the lifeos scheduler: if our brain got displaced, ask the
    broker for GPU 0 VRAM (it unloads idle, non-pinned ollama models first —
    its own policy; Bag's model is pinned), then reload ours onto the GPU.
    Checked at most every 30 s so it costs nothing on the normal path."""
    now = time.time()
    if now - _llm_check["t"] < 30:
        return
    _llm_check["t"] = now
    if _llm_on_gpu():
        return
    try:
        requests.post(BROKER_URL + "/broker/request-vram", timeout=90,
                      json={"amount_mb": 18000, "requester": "bag", "gpu": 0})
        requests.post(LLM_URL + "/api/generate", timeout=30,      # unload -> re-place
                      json={"model": LLM_MODEL, "keep_alive": 0, "prompt": ""})
        requests.post(LLM_URL + "/api/generate", timeout=240,     # reload on the GPU
                      json={"model": LLM_MODEL, "keep_alive": -1, "prompt": "",
                            "options": {"num_ctx": 4096}})
    except Exception:                                   # noqa: BLE001
        pass


def _llm_reply(user_text: str, persona: str, history=None,
               temperature=0.85, num_predict=180) -> str:
    """The LLM via ollama, in persona, kept short. Warm-pinned via keep_alive."""
    _ensure_llm_gpu()
    msgs = [{"role": "system", "content": persona}]
    for h in (history or [])[-8:]:
        if h.get("role") in ("user", "assistant") and h.get("content"):
            msgs.append({"role": h["role"], "content": h["content"]})
    msgs.append({"role": "user", "content": user_text})
    txt = ""
    for attempt in range(2):                     # the model occasionally returns ""
        r = requests.post(LLM_URL + "/api/chat", timeout=180, json={
            "model": LLM_MODEL, "messages": msgs, "stream": False,
            "keep_alive": -1,           # stay resident: a cold reload is ~90 s
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


def _norm_for_cer(s: str) -> str:
    """Fold diacritics, drop punctuation, collapse repeated letters (so the
    drawl spelling 'Bráácho' matches 'brácho'), for a fair CER."""
    s = "".join(elongation._fold_char(c) for c in _spoken(s).lower())
    s = re.sub(r"[^a-z0-9 ]+", " ", s)
    s = re.sub(r"(.)\1+", r"\1", s)
    return " ".join(s.split())


def _asr_cer(pcm: bytes, sr: int, expected: str, lang: str) -> float:
    """Character error rate of whisper's transcript vs the intended text.
    0.0 if the STT is unavailable — the gate never blocks speaking."""
    try:
        heard = _stt(_wav(pcm, sr), "check.wav", "en" if _is_en(lang) else "sk")
    except Exception:                                   # noqa: BLE001
        return 0.0
    a, b = _norm_for_cer(expected), _norm_for_cer(heard)
    if not a:
        return 0.0
    return round(1.0 - difflib.SequenceMatcher(None, a, b).ratio(), 3)


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
    lang: str = "sk"               # steers pace targets (EN runs ~15% slower)
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


_CACHE: "collections.OrderedDict[str, tuple]" = collections.OrderedDict()
_CACHE_MAX = 256


@app.post("/say")
def say(req: SayReq):
    if not req.text.strip():
        return Response(status_code=400, content="empty text")
    key = hashlib.sha1(json.dumps([req.text.strip(), req.voice, req.mode, req.lang,
                                   req.speed, req.space, req.emotion],
                                  ensure_ascii=False).encode("utf-8")).hexdigest()
    hit = _CACHE.get(key)
    if hit:                                          # identical line: instant replay
        _CACHE.move_to_end(key)
        wav, meta, drawls = hit
        return _audio_response(wav, meta, drawls, {"X-Bag-Mode": req.mode, "X-Bag-Cache": "hit"})
    wav, meta, drawls = _render(req.text.strip(), req.voice, req.mode,
                                req.speed, req.space, req.emotion, req.lang)
    _CACHE[key] = (wav, meta, drawls)
    while len(_CACHE) > _CACHE_MAX:
        _CACHE.popitem(last=False)
    return _audio_response(wav, meta, drawls, {"X-Bag-Mode": req.mode, "X-Bag-Cache": "miss"})


@app.post("/respond")
def respond(req: RespondReq):
    """DM/player types -> Bag (in persona) answers, spoken."""
    heard = req.text.strip()
    if not heard:
        return Response(status_code=400, content="empty text")
    v = VOICES.get(req.voice, VOICES[DEFAULT_VOICE])
    reply = _llm_reply(heard, _persona(v, req.lang), req.history)
    wav, meta, drawls = _render(reply, req.voice, req.mode, req.speed, req.space, lang=req.lang)
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
    # few-shot from the curated bank: this type if we have it, else the voice's
    # other lines — the model copies the register far better than a label
    shots = _bank_lines(voice_key, kind, lang)[:3]
    if not shots:
        pool = PHRASE_BANK.get("en" if _is_en(lang) else "sk", {}).get(BANK_KEY.get(voice_key, "npc"), {})
        shots = [ls[0] for ls in pool.values() if ls][:3]
    if _is_en(lang):
        ex = (" Examples of the tone (write a NEW one, do not copy): " + " | ".join(shots)) if shots else ""
        prompt = (f"Say ONE short line. Situation or type: {kind or 'a line'}. "
                  f"Reply with ONLY the line, one or two sentences, in character, "
                  f"no quotes. Put one or two natural short pauses as an em dash (—) "
                  f"where the character would hesitate or breathe. English only.{ex}")
    else:
        ex = (" Príklady tónu (napíš NOVÚ, nekopíruj): " + " | ".join(shots)) if shots else ""
        prompt = (f"Povedz JEDNU krátku repliku. Situácia alebo typ: {kind or 'replika'}. "
                  f"Odpovedz IBA replikou, jedna až dve vety, v úlohe, bez úvodzoviek. "
                  f"Vlož jednu až dve prirodzené krátke pauzy ako pomlčku (—) tam, "
                  f"kde by postava zaváhala alebo sa nadýchla. Len po slovensky.{ex}")
    text = _llm_reply(prompt, _persona(v, lang), None, temperature=0.8, num_predict=90)
    text = text.strip().strip('"').split("\n")[0].strip()
    if text and not _is_en(lang):
        # minimal-edit grammar pass (cases/agreement/bohemisms), style untouched
        try:
            fixed = _llm_reply(text, SK_FIX_SYSTEM, None, temperature=0.2, num_predict=120)
            fixed = fixed.split("\n")[0].strip().strip('"')
            if fixed and 0.5 < len(fixed) / len(text) < 2.0:
                text = fixed
        except Exception:                                   # noqa: BLE001
            pass
    return text


@app.post("/linetext")
def linetext(req: LineReq):
    """Click a phrase-type -> a line lands in the box to read; the user triggers
    Speak. Curated bank first (random, never the same line twice in a row);
    LLM improv only when the bank has nothing for this type."""
    kind = (req.type or "").strip()
    lines = _bank_lines(req.voice, kind, req.lang)
    if len(lines) >= 2:
        key = (req.voice, kind, req.lang)
        pool = [l for l in lines if l != _last_served.get(key)] or lines
        line = random.choice(pool)
        _last_served[key] = line
        return {"text": line, "source": "bank"}
    return {"text": _improv_line(req.voice, kind, req.lang), "source": "improv"}


@app.post("/line")
def line(req: LineReq):
    """Improvise a line AND speak it (kept for callers that want one shot)."""
    text = _improv_line(req.voice, (req.type or "").strip(), req.lang)
    wav, meta, drawls = _render(text, req.voice, req.mode, req.speed, req.space, lang=req.lang)
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
    wav, meta, drawls = _render(reply, voice, mode, lang=lang)
    return _audio_response(wav, meta, drawls,
                           {"X-Bag-Heard": quote(heard), "X-Bag-Reply": quote(reply)})


@app.on_event("startup")
def _warm():
    elongation.load()          # pull the MMS aligner into memory once

    def _ping_llm():           # load the brain now, not on the first click
        try:
            _llm_reply("Ahoj.", "Odpovedz jedným slovom.", None, temperature=0.1, num_predict=3)
        except Exception:      # noqa: BLE001
            pass
    threading.Thread(target=_ping_llm, daemon=True).start()


@app.get("/healthz")
def healthz():
    return {"ok": True, "tts": TTS_URL, "llm": LLM_MODEL, "stt": STT_URL,
            "voices": list(VOICES), "elongation": elongation.load(),
            "elongation_error": elongation._load_error}


@app.get("/", response_class=HTMLResponse)
def index():
    with open(os.path.join(HERE, "web", "index.html"), encoding="utf-8") as f:
        return f.read()
