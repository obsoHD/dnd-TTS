# -*- coding: utf-8 -*-
"""Deterministic drawls and breaks, plus word timing for sentence pacing.

Drawls — `Brá**cho` (stars after a vowel, each ≈ +150 ms):
  The model's own drawl sounds natural (it carries the voice's intonation), but
  its length is random. Time-stretching a short vowel in post is exact but
  synthetic at big factors. So: the vowel is repeated in the spelling the model
  sees ("Bráácho") so it PERFORMS the drawl; forced alignment locates it and
  post enforces the length — too long -> the steady middle is cut out with a
  phase-aligned crossfade; a little short -> gentle Rubber Band stretch.

Breaks — TAB, em dash, "...", " - " (≈250 ms each, they stack):
  A pause opened at the quietest point of the gap between two words and filled
  with matched room tone (not digital zero, which reads as a hole), with long
  fades on both sides.

All joins use WSOLA-style phase-aligned crossfades so nothing "cuts".
CPU-only; alignment of a line is fast.
"""
from __future__ import annotations

import re
import struct
import subprocess

MS_PER_STAR = 150
VOWEL_BASE_MS = 120            # a plain vowel's own length; stars add to this
MAX_STARS = 6
BREAK_MS = 250                 # one break; they stack
BREAK_CHARS = {"\t", "—", "–", "…"}   # after _norm_breaks
_HYPHEN_BREAK = re.compile(r"\s[-–]\s")

_FOLD = {"á": "a", "ä": "a", "é": "e", "í": "i", "ó": "o", "ô": "o", "ú": "u",
         "ý": "y", "ĺ": "l", "ŕ": "r", "č": "c", "š": "s", "ž": "z", "ť": "t",
         "ď": "d", "ň": "n", "ľ": "l"}
_VOWELS = set("aeiouy")

_model = _tokenizer = _aligner = None
_load_error = None


def load():
    """Load MMS_FA once; errors are captured so the service still runs."""
    global _model, _tokenizer, _aligner, _load_error
    if _model is not None or _load_error is not None:
        return _model is not None
    try:
        import torchaudio
        bundle = torchaudio.pipelines.MMS_FA
        _model = bundle.get_model()          # CPU
        _tokenizer = bundle.get_tokenizer()
        _aligner = bundle.get_aligner()
        return True
    except Exception as e:                    # noqa: BLE001
        _load_error = repr(e)
        return False


def _fold_char(c: str) -> str:
    c = c.lower()
    return _FOLD.get(c, c)


def _norm_breaks(text: str) -> str:
    """'...' and ' - ' become the canonical break characters."""
    return _HYPHEN_BREAK.sub(" — ", text.replace("...", "…"))


# ------------------------------------------------------------------- parsing
def parse_marks(text: str):
    """Pull the markup out. Returns (tts_text, aligner_words, marks, breaks):
      tts_text      what the model speaks — stars/break chars stripped, marked
                    vowels REPEATED so the model performs the drawl
      aligner_words folded ASCII words matching tts_text (for alignment)
      marks         [(word_index, first_vowel_char_index, copies, stars)]
      breaks        [(word_index, count)] -> count x BREAK_MS after that word"""
    text = _norm_breaks(text)
    disp, aligner_words, marks, breaks = [], [], [], []
    cur = []

    def flush():
        if cur:
            aligner_words.append("".join(cur))
            cur.clear()

    i, n = 0, len(text)
    while i < n:
        c = text[i]
        if c == "*":
            j = i
            while j < n and text[j] == "*":
                j += 1
            stars = min(j - i, MAX_STARS)
            if cur and cur[-1] in _VOWELS:
                # clean spelling for the model (repeated letters are out of its
                # distribution); the hold is made in post, pitch-synchronously
                marks.append((len(aligner_words), len(cur) - 1, 1, stars))
            i = j
            continue
        if c in BREAK_CHARS:
            flush()
            j = i
            while j < n and text[j] in BREAK_CHARS:
                j += 1
            # The model's own documented pause tokens, inline where the break
            # falls (pause ~400-700 ms; long_pause ~700-1500 ms for stacked or
            # "..."). It renders the pause with natural decay and onset — no
            # spliced silence, nothing "cuts".
            longp = (j - i) >= 2 or "…" in text[i:j]
            disp.append(" <|prosody:long_pause|> " if longp else " <|prosody:pause|> ")
            i = j
            continue
        if c.isspace():
            flush()
            disp.append(" ")
            i += 1
            continue
        disp.append(c)
        fc = _fold_char(c)
        if "a" <= fc <= "z":
            cur.append(fc)
        i += 1
    flush()

    tts_text = " ".join("".join(disp).split())
    marks = [m for m in marks if m[0] < len(aligner_words)]
    return tts_text, aligner_words, marks, breaks


# ------------------------------------------------------------------ alignment
def _align(pcm: bytes, sr: int, aligner_words):
    """MMS forced alignment. Returns (flat, offsets, wlen, word_start, word_end)
    with times in seconds, or None if unavailable."""
    if not aligner_words or not load():
        return None
    import torch
    import torchaudio
    samples = torch.tensor(struct.unpack("<%dh" % (len(pcm) // 2), pcm),
                           dtype=torch.float32) / 32768.0
    wav = samples.unsqueeze(0)
    wav16 = torchaudio.functional.resample(wav, sr, 16000) if sr != 16000 else wav
    with torch.inference_mode():
        emission, _ = _model(wav16)
        token_spans = _aligner(emission[0], _tokenizer(aligner_words))
    spf = wav16.size(1) / emission.size(1) / 16000.0
    flat, offsets, wlen, word_start, word_end, off = [], [], [], [], [], 0
    for spans in token_spans:
        offsets.append(off)
        wlen.append(len(spans))
        for sp in spans:
            flat.append((sp.start * spf, sp.end * spf))
        off += len(spans)
        word_start.append(spans[0].start * spf if spans else (flat[-1][0] if flat else 0.0))
        word_end.append(spans[-1].end * spf if spans else (flat[-1][1] if flat else 0.0))
    return flat, offsets, wlen, word_start, word_end


def _quietest_point(pcm: bytes, sr: int, t_a: float, t_b: float) -> float:
    """Lowest-energy 10 ms window between a word's end and the next onset."""
    a, b = int(t_a * sr), int(t_b * sr)
    if b - a < int(0.02 * sr):
        return t_b
    seg = struct.unpack("<%dh" % (b - a), pcm[a * 2:b * 2])
    w = max(8, int(0.010 * sr))
    sq = [0]
    for v in seg:
        sq.append(sq[-1] + v * v)
    best, best_i = None, a + (b - a) // 2
    for i in range(0, len(seg) - w, max(1, w // 2)):
        e = sq[i + w] - sq[i]
        if best is None or e < best:
            best, best_i = e, a + i + w // 2
    return best_i / sr


def word_bounds(pcm: bytes, sr: int, aligner_words) -> list:
    """[(start, end)] per aligner word; end = the quiet point after the word.
    Used by sentence pacing. [] if alignment is unavailable."""
    al = _align(pcm, sr, aligner_words)
    if al is None:
        return []
    _, _, _, word_start, word_end = al
    clip_end = len(pcm) / (2 * sr)
    out = []
    for i in range(len(word_start)):
        nxt = word_start[i + 1] if i + 1 < len(word_start) else clip_end
        out.append((word_start[i], _quietest_point(pcm, sr, word_end[i], max(word_end[i], nxt))))
    return out


# --------------------------------------------------------------- DSP helpers
def _np():
    import numpy as np
    return np


def _stretch(seg: bytes, sr: int, add_sec: float) -> bytes:
    """Lengthen a segment by add_sec, pitch-preserving (Rubber Band; atempo fallback)."""
    dur = (len(seg) // 2) / sr
    if dur <= 0 or add_sec <= 0:
        return seg
    try:
        np = _np()
        from pedalboard import time_stretch
        factor = (dur + add_sec) / dur
        x = np.frombuffer(seg, dtype=np.int16).astype(np.float32) / 32768.0
        y = time_stretch(x[None, :], sr, stretch_factor=factor)[0]
        if len(y) < len(x):
            y = time_stretch(x[None, :], sr, stretch_factor=1.0 / factor)[0]
        return (np.clip(y, -1.0, 1.0) * 32767).astype(np.int16).tobytes()
    except Exception:                                     # noqa: BLE001
        pass
    rate = dur / (dur + add_sec)
    rates, r = [], rate
    while r < 0.5:
        rates.append(0.5)
        r *= 2
    rates.append(r)
    cmd = ["ffmpeg", "-f", "s16le", "-ar", str(sr), "-ac", "1", "-i", "pipe:0",
           "-af", ",".join(f"atempo={x:.5f}" for x in rates), "-f", "s16le", "pipe:1"]
    return subprocess.run(cmd, input=seg, stdout=subprocess.PIPE,
                          stderr=subprocess.DEVNULL).stdout or seg


def _xfade_join(a: bytes, b: bytes, sr: int, ms: float = 16.0, search_ms: float = 12.0) -> bytes:
    """Join two PCM pieces with a crossfade whose start in `b` is chosen by
    cross-correlation within one pitch period (WSOLA), so the waveforms line up
    in phase and the seam is inaudible."""
    np = _np()
    A = np.frombuffer(a, dtype=np.int16).astype(np.float32)
    B = np.frombuffer(b, dtype=np.int16).astype(np.float32)
    w = int(sr * ms / 1000)
    s = int(sr * search_ms / 1000)
    if len(A) < w or len(B) < w + s + 1:
        w = max(1, min(w, len(A), len(B)))
        s = 0
    tail = A[-w:]
    best = 0
    if s > 0:
        scores = [float(np.dot(tail, B[off:off + w])) for off in range(0, s)]
        best = int(np.argmax(scores))
    Bs = B[best:]
    ramp = np.linspace(0.0, 1.0, w, dtype=np.float32)
    mix = tail * (1.0 - ramp) + Bs[:w] * ramp
    out = np.concatenate([A[:-w], mix, Bs[w:]])
    return np.clip(out, -32768, 32767).astype(np.int16).tobytes()


def _cut_to(region: bytes, sr: int, keep_sec: float) -> bytes:
    """Shorten a sustained vowel to keep_sec by removing its steady middle;
    the join is phase-aligned so it does not warble."""
    n = len(region) // 2
    keep = int(keep_sec * sr)
    if keep >= n:
        return region
    w = int(0.016 * sr)
    head_n = keep // 2 + w // 2
    tail_n = keep - head_n + w + int(0.012 * sr)      # slack for the phase search
    return _xfade_join(region[:head_n * 2], region[-tail_n * 2:], sr)


def _ramp(buf: bytes, nbytes: int, rising: bool) -> bytes:
    n = min(nbytes // 2, len(buf) // 2)
    if n <= 0:
        return buf
    if rising:
        head = struct.unpack("<%dh" % n, buf[:n * 2])
        head = [int(v * (i + 1) / n) for i, v in enumerate(head)]
        return struct.pack("<%dh" % n, *head) + buf[n * 2:]
    tail = struct.unpack("<%dh" % n, buf[-n * 2:])
    tail = [int(v * (n - i) / n) for i, v in enumerate(tail)]
    return buf[:-n * 2] + struct.pack("<%dh" % n, *tail)


def _splice(out: bytes, sb: int, eb: int, new: bytes, sr: int, fade_ms: float = 8.0) -> bytes:
    """Replace out[sb:eb] with `new`, fading fade_ms on every cut edge."""
    w = int(sr * fade_ms / 1000) * 2
    left = _ramp(out[:sb], w, rising=False)
    right = _ramp(out[eb:], w, rising=True)
    return left + _ramp(_ramp(new, w, True), w, False) + right


def _room_tone(pcm: bytes, sr: int, t: float, n_samples: int) -> bytes:
    """A pause that sounds like the room, not a hole: noise shaped to the level
    of the quiet audio around t (floor -60 dBFS)."""
    np = _np()
    a = int(t * sr)
    w = int(0.03 * sr)
    x = np.frombuffer(pcm[max(0, a - w) * 2:(a + w) * 2], dtype=np.int16).astype(np.float32)
    rms = float(np.sqrt(np.mean(x ** 2))) if x.size else 0.0
    level = max(rms * 0.8, 32767 * 10 ** (-60 / 20))
    noise = np.random.default_rng(0).standard_normal(n_samples).astype(np.float32) * level
    return np.clip(noise, -32768, 32767).astype(np.int16).tobytes()


def _nucleus(t0: float, t1: float) -> tuple[float, float]:
    """The steady part of a vowel to change: skip 20 ms of transition on each
    side (consonant-vowel boundaries carry the place-of-articulation cues and
    sound broken when stretched); short vowels use the central 40%."""
    d = t1 - t0
    if d >= 0.08:
        return t0 + 0.02, t1 - 0.02
    return t0 + 0.3 * d, t1 - 0.3 * d


def _psola(pcm: bytes, sr: int, t0: float, t1: float, factor: float, ctx: float = 0.12):
    """TD-PSOLA duration change of [t0, t1] via Praat (parselmouth): pitch
    periods are repeated/dropped pitch-synchronously, so F0 and formants are
    untouched — what speech-editing research uses for phone-level duration.
    Returns (a, b, new_bytes) for the context-extended span to splice, or None."""
    try:
        import numpy as np
        import parselmouth
        from parselmouth.praat import call
    except Exception:                                     # noqa: BLE001
        return None
    x = np.frombuffer(pcm, dtype=np.int16).astype(np.float64) / 32768.0
    total = len(x) / sr
    a, b = max(0.0, t0 - ctx), min(total, t1 + ctx)
    ia, ib = int(a * sr), int(b * sr)
    if ib - ia < int(0.1 * sr):
        return None
    try:
        snd = parselmouth.Sound(x[ia:ib], sampling_frequency=sr)
        manip = call(snd, "To Manipulation", 0.01, 60, 400)
        tier = call(manip, "Extract duration tier")
        eps = 0.003
        for t, v in ((t0 - a - eps, 1.0), (t0 - a, factor), (t1 - a, factor), (t1 - a + eps, 1.0)):
            call(tier, "Add point", max(0.0, t), v)
        call([tier, manip], "Replace duration tier")
        y = call(manip, "Get resynthesis (overlap-add)").values[0]
        return a, b, (np.clip(y, -1.0, 1.0) * 32767).astype(np.int16).tobytes()
    except Exception:                                     # noqa: BLE001
        return None


# ------------------------------------------------------------------ the pass
def elongate(pcm: bytes, sr: int, aligner_words, marks, breaks=None,
             break_ms: int = BREAK_MS) -> tuple[bytes, list]:
    """Apply the markup. Returns (pcm, applied) with applied = [(label, ms)];
    a drawl's ms is its full deliberate hold, a break's its silence."""
    if not marks:
        return pcm, []
    al = _align(pcm, sr, aligner_words)
    if al is None:
        return pcm, []
    flat, offsets, wlen, word_start, word_end = al
    clip_end = len(pcm) / (2 * sr)

    def gap_after(awi: int) -> float:
        nxt = word_start[awi + 1] if awi + 1 < len(word_start) else clip_end
        return _quietest_point(pcm, sr, word_end[awi], max(word_end[awi], nxt))

    edits = []                                   # (t0, t1, new, label, ms, fade_ms)
    for awi, fi, copies, stars in marks:
        if awi >= len(offsets):
            continue
        gi = offsets[awi] + fi
        g_end = gi + copies
        if gi >= len(flat):
            continue
        t0 = flat[gi][0]
        t1 = flat[g_end][0] if g_end < offsets[awi] + wlen[awi] else gap_after(awi)
        if t1 <= t0:
            continue
        actual = t1 - t0
        target = (VOWEL_BASE_MS + stars * MS_PER_STAR) / 1000.0
        delta = target - actual
        if abs(delta) <= 0.03:
            continue                                       # already the right length
        # PSOLA on the vowel nucleus (pitch-synchronous; transitions untouched)
        n0, n1 = _nucleus(t0, t1)
        factor = max(0.2, min(6.0, ((n1 - n0) + delta) / max(0.02, n1 - n0)))
        res = _psola(pcm, sr, n0, n1, factor)
        if res is not None:
            a, b, new = res
            edits.append((a, b, new, aligner_words[awi], int(target * 1000), 3.0))
            continue
        region = pcm[int(t0 * sr) * 2:int(t1 * sr) * 2]   # fallbacks
        new = _cut_to(region, sr, target) if delta < 0 else _stretch(region, sr, delta)
        edits.append((t0, t1, new, aligner_words[awi], int(target * 1000), 6.0))
    # breaks are rendered by the model itself (pause tokens); nothing to splice

    applied, out = [], pcm
    for t0, t1, new, label, ms, fade in sorted(edits, key=lambda e: e[0], reverse=True):
        out = _splice(out, int(t0 * sr) * 2, int(t1 * sr) * 2, new, sr, fade)
        applied.append((label, ms))
    return out, list(reversed(applied))
