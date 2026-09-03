"""One line in, one gated and mastered take out (REBUILD.md §2.5-§2.7).

``render_line`` is the whole per-line pipeline: canonicalize -> N concurrent
takes from the TTS -> gate (sanity, speaker similarity, CER on the winner) ->
master. It never plays anything and never blocks: when no take passes the gate
after one retry round, the best take is still returned, flagged ``failed``, so
the table can play it and offer Regenerate.

``render_id`` is what makes the cache honest: it hashes everything that can
change the served bytes (voice + version, the recipe, the canonical text, the
take number) and nothing else, so an identical request across restarts and
machines lands on the same file.
"""
from __future__ import annotations

import hashlib
import json
import math
import threading
import time
from dataclasses import asdict, dataclass
from functools import lru_cache

from app import canon, gate, master, tts_client, voices
from app.canon import Canon
from app.gate import TakeScore
from app.voices import Voice

LONG_LINE_CHARS = 120        # above this the model wanders more; a third take is cheap (server batches)
TAKE_STRIDE = 8              # seed room per take_no: first round 0-2, retry 3-5, spare 6-7
RETRY_OFFSET = 3
RETRY_TAKES = 3
MAX_NEW_TOKENS_CAP = 1024


def _gate_rules() -> str:
    """The gate's thresholds are its module-level constants; hashing those (not
    the source text) bumps the recipe when a rule changes and not when a comment does."""
    rules = {k: v for k, v in vars(gate).items() if k.isupper() and isinstance(v, (int, float, str))}
    return json.dumps(rules, sort_keys=True)


def _tokens_json() -> str:
    """tokens.json in canonical form so whitespace or line endings never bump the recipe."""
    return json.dumps(json.loads(canon.TOKENS_PATH.read_text(encoding="utf-8")),
                      sort_keys=True, separators=(",", ":"))


def _recipe_version() -> str:
    parts = [json.dumps(asdict(tts_client.Sampler()), sort_keys=True), _gate_rules(),
             canon.CANON_VERSION, _tokens_json()]
    return hashlib.sha256("\n".join(parts).encode("utf-8")).hexdigest()


RECIPE_VERSION: str = _recipe_version()


@dataclass
class RenderResult:
    render_id: str
    voice_id: str
    text: str
    canon: Canon
    seed: int
    take_no: int
    raw_pcm: bytes
    pcm: bytes
    sr: int
    sim: float
    cer: float | None
    verified: bool
    gate: str                      # pass | failed
    scores: list[TakeScore]
    timings: dict
    # Identity of the recipe that produced this take; the renders row needs
    # them and only the render step has the voice in hand.
    voice_version: int = 0
    recipe_version: str = ""
    master_version: str = ""
    lufs: float | None = None


def render_id(voice: Voice, canon_text: str, take_no: int) -> str:
    key = "|".join([voice.id, str(voice.version), RECIPE_VERSION, canon_text, str(take_no)])
    return hashlib.sha256(key.encode("utf-8")).hexdigest()


def master_version(voice: Voice) -> str:
    """sha256(master block of voice.yaml + master.py VERSION): names the mastered
    file's recipe so an Energy change re-masters without touching the GPU."""
    key = json.dumps(voice.master, sort_keys=True) + "|" + master.VERSION
    return hashlib.sha256(key.encode("utf-8")).hexdigest()


def max_new_tokens(syllables: int) -> int:
    """Token budget grows with the line (§2.2) so a long line is not truncated
    and a short one cannot ramble on."""
    return min(MAX_NEW_TOKENS_CAP, 120 + 12 * syllables)


def params_for(voice_master: dict) -> dict:
    """Voice master block -> concrete chain parameters via the Energy knob."""
    return master.energy_to_params(int(voice_master["energy"]), voice_master)


def mastered(raw_pcm: bytes, sr: int, voice_master: dict) -> bytes:
    return master.master(raw_pcm, sr, params_for(voice_master))


@lru_cache(maxsize=8)
def _speaker_gate(ref: str, ref_sha: str) -> gate.SpeakerGate:
    """One encoder + reference embedding per locked clip; the sha in the key
    means a re-lock at the same path gets a fresh gate."""
    return gate.SpeakerGate(ref)


def speaker_gate_for(voice: Voice) -> gate.SpeakerGate:
    """The voice's gate, bound to its reference only after proving the clip on
    disk is still the locked one."""
    return _speaker_gate(str(voices.check_reference(voice)), voice.ref_sha256)


def _threshold(voice: Voice) -> float:
    return float(voice.gate["loose"] if voice.gate.get("mode") == "loose" else voice.gate["strict"])


def _seeds(voice: Voice, take_no: int, offset: int, n: int) -> list[int]:
    base = voice.golden_seed + take_no * TAKE_STRIDE + offset
    return [base + i for i in range(n)]


def _loudness(pcm: bytes, sr: int) -> float | None:
    """Integrated loudness, or None when the meter cannot gate the clip (shorter
    than its 400 ms block reads -inf); a one-word filler must still render and store."""
    lufs = float(master.measure(pcm, sr)["lufs"])
    return lufs if math.isfinite(lufs) else None


def _best(candidates: list[tuple[int, TakeScore]]) -> tuple[int, TakeScore]:
    """Fallback ordering when nothing passed: sane first, then similarity."""
    return max(candidates, key=lambda c: (c[1].sane, c[1].sim))


def render_line(voice: Voice, text: str, take_no: int = 0, n_takes: int = 2, *,
                cancel: threading.Event | None = None) -> RenderResult:
    """``cancel`` is the worker's cooperative stop: checked before each round of
    takes and inside the streaming read, so a live line pre-empts a batch job
    within one chunk instead of one full render."""
    t0 = time.perf_counter()
    c = canon.canonicalize(text, lang=voice.lang, banned=set(voice.banned_tokens))
    speaker = speaker_gate_for(voice)
    n = 3 if len(c.spoken) > LONG_LINE_CHARS else n_takes
    budget = max_new_tokens(c.syllables)
    timings = {"canon": time.perf_counter() - t0, "synth": 0.0, "gate": 0.0, "rounds": 0}
    seen: list[tuple[int, TakeScore]] = []          # (index into pool, score)
    pool: list[tuple[int, bytes, int]] = []
    chosen: tuple[int, TakeScore] | None = None
    # Forwarded only when given, so the call stays byte-identical for M1 callers
    # and their test doubles that predate the kwarg.
    stop = {"cancel": cancel} if cancel is not None else {}
    for offset, count in ((0, n), (RETRY_OFFSET, RETRY_TAKES)):
        tts_client.check_cancel(cancel)
        t = time.perf_counter()
        takes = tts_client.synth_many(c.text, voice.ref_tts_path, voice.ref_transcript,
                                      _seeds(voice, take_no, offset, count), voice.sampler, budget, **stop)
        timings["synth"] += time.perf_counter() - t
        t = time.perf_counter()
        idx, scores = gate.select(speaker, takes, c, voice.lang, _threshold(voice),
                                  float(voice.gate.get("cer_max", 0.15)))
        timings["gate"] += time.perf_counter() - t
        timings["rounds"] += 1
        seen.extend((len(pool) + i, s) for i, s in enumerate(scores))
        pool.extend(takes)
        if idx is not None and scores[idx].reason == "ok":
            chosen = (len(pool) - len(takes) + idx, scores[idx])
            break
    passed = chosen is not None
    pick, score = chosen if passed else _best(seen)
    _, raw, sr = pool[pick]
    t = time.perf_counter()
    pcm = mastered(raw, sr, voice.master)
    timings["master"] = time.perf_counter() - t
    timings["total"] = time.perf_counter() - t0
    return RenderResult(
        render_id=render_id(voice, c.text, take_no), voice_id=voice.id, text=text, canon=c,
        seed=score.seed, take_no=take_no, raw_pcm=raw, pcm=pcm, sr=sr, sim=score.sim, cer=score.cer,
        verified=score.verified, gate="pass" if passed else "failed", scores=[s for _, s in seen],
        timings=timings, voice_version=voice.version, recipe_version=RECIPE_VERSION,
        master_version=master_version(voice), lufs=_loudness(pcm, sr))
