"""Take gate: identity is the only measured quality (REBUILD.md §1.4, §2.5).

Per take, in order: sanity (is it audio of about the right length), speaker
similarity against the locked reference (the selector), and whisper CER on the
similarity winner only (truncation/garbage detector). Pitch and pace are never
scored. STT being down never changes which seed is served; the take is merely
flagged unverified.

All audio is int16 mono PCM bytes plus a sample rate. The heavy dependencies
(resemblyzer, which drags in torch) are imported only when a ``SpeakerGate`` is
constructed, so the pure helpers here and their tests load anywhere.
"""
from __future__ import annotations

import difflib
import math
import re
import unicodedata
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

from app import stt

if TYPE_CHECKING:
    from app.canon import Canon

PEAK_MIN_DBFS = -30.0        # quieter than this is a failed generation, not a soft delivery
SILENCE_DBFS = -45.0         # frame RMS below this counts as silence
FRAME_MS = 20
MAX_INTERNAL_SILENCE_S = 1.5
SEC_PER_SYLLABLE_MIN = 0.12
SEC_PER_SYLLABLE_MAX = 0.45
DURATION_SLACK_S = 2.0       # room for the model's own lead-in/outro
STT_TIMEOUT_S = 2.0          # the table's budget: a slow verifier must never delay a line
EMBEDDING_SUFFIX = ".emb.npy"

_TOKEN = re.compile(r"<\|[^|]*\|>")
_NON_ALNUM = re.compile(r"[^a-z0-9 ]+")
_REPEATED = re.compile(r"(.)\1+")


def _floats(pcm: bytes) -> np.ndarray:
    """int16 bytes -> float32 in [-1, 1]. A stream cut mid-sample drops its
    dangling byte instead of crashing the gate on a truncated take."""
    usable = len(pcm) - len(pcm) % 2
    return np.frombuffer(pcm[:usable], dtype=np.int16).astype(np.float32) / 32768.0


def _dbfs(linear: float) -> float:
    return 20.0 * math.log10(linear) if linear > 0.0 else -math.inf


def _cosine(a: np.ndarray, b: np.ndarray) -> float:
    denominator = float(np.linalg.norm(a) * np.linalg.norm(b))
    return float(np.dot(a, b) / denominator) if denominator > 0.0 else 0.0


class SpeakerGate:
    """Cosine similarity of resemblyzer speaker embeddings against one reference.

    The reference embedding is computed once and cached next to the wav as
    ``<ref>.emb.npy`` because the reference is immutable per voice version; a
    cache older than the wav is recomputed so a re-lock can never be scored
    against the previous voice.
    """

    def __init__(self, ref_wav: str | Path) -> None:
        try:
            import webrtcvad  # noqa: F401  (resemblyzer's VAD; pip name is webrtcvad-wheels)
            from resemblyzer import VoiceEncoder, preprocess_wav
        except ImportError as e:
            raise ImportError(
                "SpeakerGate needs resemblyzer and webrtcvad-wheels "
                "(python -m pip install resemblyzer webrtcvad-wheels)") from e
        self._encoder = VoiceEncoder("cpu", verbose=False)
        self._preprocess = preprocess_wav
        self._reference = self._reference_embedding(Path(ref_wav))

    def _reference_embedding(self, ref_wav: Path) -> np.ndarray:
        cache = ref_wav.with_suffix(EMBEDDING_SUFFIX)
        if cache.exists() and cache.stat().st_mtime >= ref_wav.stat().st_mtime:
            return np.load(cache)
        embedding = self._encoder.embed_utterance(self._preprocess(ref_wav))
        np.save(cache, embedding)
        return embedding

    def similarity(self, pcm: bytes, sr: int) -> float:
        """Cosine in [-1, 1]; resemblyzer resamples to 16 kHz and trims silence itself."""
        wav = self._preprocess(_floats(pcm), source_sr=sr)
        return _cosine(self._encoder.embed_utterance(wav), self._reference)


def _longest_internal_silence(x: np.ndarray, sr: int) -> float:
    """Longest run of quiet frames strictly between the first and last loud
    frame. Leading/trailing silence is the master's edge-trim job, not a defect;
    a long hole in the middle is a stalled generation."""
    frame = max(1, sr * FRAME_MS // 1000)
    n_frames = x.size // frame
    if n_frames == 0:
        return 0.0
    rms = np.sqrt(np.mean(x[: n_frames * frame].reshape(n_frames, frame) ** 2, axis=1))
    loud = rms > 10.0 ** (SILENCE_DBFS / 20.0)
    loud_idx = np.flatnonzero(loud)
    if loud_idx.size < 2:
        return 0.0
    quiet_inside = ~loud[loud_idx[0]: loud_idx[-1] + 1]
    edges = np.flatnonzero(np.diff(np.concatenate(([False], quiet_inside, [False])).astype(np.int8)))
    runs = edges[1::2] - edges[0::2]
    return float(runs.max()) * frame / sr if runs.size else 0.0


def speech_seconds(pcm: bytes, sr: int) -> float:
    """Seconds of frames above the silence floor: how long the voice actually
    speaks, pauses excluded. WHY: pace judged on total duration punishes a line
    for its sentence breaks; a four-question line is not slow because it pauses."""
    x = _floats(pcm)
    frame = max(1, sr * FRAME_MS // 1000)
    n_frames = x.size // frame
    if n_frames == 0:
        return 0.0
    rms = np.sqrt(np.mean(x[: n_frames * frame].reshape(n_frames, frame) ** 2, axis=1))
    return float(np.count_nonzero(rms > 10.0 ** (SILENCE_DBFS / 20.0))) * frame / sr


def sanity(pcm: bytes, sr: int, syllables: int) -> tuple[bool, str]:
    """Cheap structural checks before any model runs: a take that is near-silent,
    wildly long/short for its syllable count, or has a > 1.5 s hole is a broken
    generation regardless of how it sounds. Returns (ok, reason)."""
    x = _floats(pcm)
    peak = _dbfs(float(np.max(np.abs(x)))) if x.size else -math.inf
    if peak < PEAK_MIN_DBFS:
        return False, f"peak {peak:.1f} dBFS < {PEAK_MIN_DBFS:.0f}"
    duration = x.size / sr
    lo = SEC_PER_SYLLABLE_MIN * syllables
    hi = SEC_PER_SYLLABLE_MAX * syllables + DURATION_SLACK_S
    if not lo <= duration <= hi:
        return False, f"duration {duration:.2f} s outside [{lo:.2f}, {hi:.2f}]"
    gap = _longest_internal_silence(x, sr)
    if gap > MAX_INTERNAL_SILENCE_S:
        return False, f"internal silence {gap:.2f} s > {MAX_INTERNAL_SILENCE_S}"
    return True, "ok"


def _norm_for_cer(s: str) -> str:
    """Lower-case, fold diacritics, drop punctuation and control tokens, collapse
    repeated letters. The drawl spelling 'Bráácho' must match whisper's 'brácho'
    and whisper's accent slips (c/č) must not count: CER exists only to catch
    truncation and garbage, not to grade orthography."""
    s = _TOKEN.sub(" ", s).lower()
    s = unicodedata.normalize("NFKD", s).encode("ascii", "ignore").decode()
    s = _NON_ALNUM.sub(" ", s)
    s = _REPEATED.sub(r"\1", s)
    return " ".join(s.split())


def cer(pcm: bytes, sr: int, spoken: str, lang: str) -> float | None:
    """Character error rate of whisper's transcript against the intended text,
    or None when the STT is unavailable: the gate never blocks speaking, it only
    withholds the "verified" mark."""
    heard = stt.transcribe(pcm, sr, lang, timeout=STT_TIMEOUT_S)
    if heard is None:
        return None
    expected, got = _norm_for_cer(spoken), _norm_for_cer(heard)
    if not expected:
        return 0.0
    return round(1.0 - difflib.SequenceMatcher(None, expected, got).ratio(), 3)


def word_ratio(spoken: str, heard: str) -> float:
    """Heard-over-expected word count on normalised text; 1.0 when nothing was
    expected. The golden test bounds it to 0.75-1.25 to catch dropped or
    hallucinated words that character distance alone can hide."""
    expected = _norm_for_cer(spoken).split()
    if not expected:
        return 1.0
    return len(_norm_for_cer(heard).split()) / len(expected)


@dataclass
class TakeScore:
    """One take's verdict. ``reason`` is "ok" when the take cleared every check
    applied to it, else the first failed check; ``cer`` is None when it was not
    measured (STT down, or a better take was chosen first)."""

    seed: int
    sim: float
    sane: bool
    reason: str
    cer: float | None = None
    verified: bool = False


def _score_take(gate: SpeakerGate, seed: int, pcm: bytes, sr: int, syllables: int,
                strict: float) -> TakeScore:
    """Sanity and similarity for every take, even failed ones: calibration needs
    each take's SIM and the fallback needs a full ordering."""
    sane, reason = sanity(pcm, sr, syllables)
    sim = gate.similarity(pcm, sr)
    if sane and sim < strict:
        reason = f"sim {sim:.3f} < strict {strict:.3f}"
    return TakeScore(seed=seed, sim=sim, sane=sane, reason=reason)


def _verify(score: TakeScore, pcm: bytes, sr: int, spoken: str, lang: str, cer_max: float) -> bool:
    """CER on one take. True when it passed, or when the STT could not answer:
    a missing verdict marks the take unverified, it never demotes it. False only
    on a measured failure, which is the one thing that moves on to the next seed."""
    score.cer = cer(pcm, sr, spoken, lang)
    if score.cer is None:
        return True
    if score.cer <= cer_max:
        score.verified = True
        return True
    score.reason = f"cer {score.cer:.3f} > {cer_max}"
    return False


def select(gate: SpeakerGate, takes: list[tuple[int, bytes, int]], canon: Canon, lang: str,
           strict: float, cer_max: float = 0.15) -> tuple[int | None, list[TakeScore]]:
    """Pick the take to serve. Order: sanity -> similarity (desc) -> CER on the
    best candidate only, moving to the next by similarity when CER fails. When
    nothing passes, the best take (sane first, then by similarity) is returned
    with its failure in ``reason`` so the table is never blocked; the caller
    reads ``scores[idx].reason == "ok"`` as the gate verdict.

    Returns (index into ``takes`` or None when empty, one score per take in input order).
    """
    scores = [_score_take(gate, seed, pcm, sr, canon.syllables, strict) for seed, pcm, sr in takes]
    ranked = sorted(range(len(takes)), key=lambda i: (scores[i].sane, scores[i].sim), reverse=True)
    for i in (i for i in ranked if scores[i].reason == "ok"):
        _, pcm, sr = takes[i]
        if _verify(scores[i], pcm, sr, canon.spoken, lang, cer_max):
            return i, scores
    return (ranked[0] if ranked else None), scores
