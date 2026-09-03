# -*- coding: utf-8 -*-
"""Deterministic drawls and breaks.

Drawls — `Brá**cho` (stars after a vowel, each ≈ +150 ms):
  The model's own drawl sounds natural (it carries the voice's intonation), but
  its length is random: the same spelling holds a vowel for 0.3 s one time and
  13 s the next. Time-stretching a short vowel in post is exact but sounds
  synthetic at big factors. So we do both, each for what it is good at:
    1. The vowel is repeated in the spelling the model sees ("Bráácho"), so the
       model PERFORMS a drawl with natural timbre and pitch.
    2. MMS forced alignment locates that vowel, and post enforces the length:
       too long -> cut the steady middle out with a crossfade (steady-state
       audio splices invisibly); a little short -> gentle Rubber Band stretch.
  Exact syllable, exact length, natural sound.

Breaks — TAB, em dash, "...", " - " (≈250 ms each, they stack):
  A short silence opened at the quietest point of the gap between two words,
  never at the aligner's token end (that sits inside the word's tail and cuts
  voiced audio, which stutters). Every cut is faded.

CPU-only (the GPUs are full with the models); alignment of a line is fast.
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

# Slovak diacritics -> ASCII, 1:1 per character so positions are preserved.
# MMS_FA aligns romanized text; we keep our own fold to map back.
_FOLD = {"á": "a", "ä": "a", "é": "e", "í": "i", "ó": "o", "ô": "o", "ú": "u",
         "ý": "y", "ĺ": "l", "ŕ": "r", "č": "c", "š": "s", "ž": "z", "ť": "t",
         "ď": "d", "ň": "n", "ľ": "l"}
_VOWELS = set("aeiouy")

_model = _tokenizer = _aligner = None
_load_error = None


def load():
    """Load MMS_FA once. Errors are captured so the service still runs
    (drawls/breaks become no-ops) if torch is missing."""
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
    """People (and LLMs) write pauses as '...' and ' - '; fold them into the
    canonical break characters so they behave like TAB / em dash."""
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
    cur = []                                   # folded chars of the word being built

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
                extra = 1 if stars <= 2 else 2          # cue the model, post fixes length
                first = len(cur) - 1
                cur.extend([cur[-1]] * extra)
                disp.extend([disp[-1]] * extra)
                marks.append((len(aligner_words), first, 1 + extra, stars))
            i = j
            continue
        if c in BREAK_CHARS:
            flush()
            if aligner_words:
                idx = len(aligner_words) - 1
                if breaks and breaks[-1][0] == idx:
                    breaks[-1] = (idx, breaks[-1][1] + 1)    # stacked breaks add up
                else:
                    breaks.append((idx, 1))
            disp.append(" ")
            i += 1
            continue
        if c.isspace():
            flush()
            disp.append(" ")
            i += 1
            continue
        disp.append(c)
        fc = _fold_char(c)
        if "a" <= fc <= "z":                   # aligner dictionary is ASCII a-z only
            cur.append(fc)
        i += 1
    flush()

    tts_text = " ".join("".join(disp).split())
    marks = [m for m in marks if m[0] < len(aligner_words)]
    return tts_text, aligner_words, marks, breaks


# --------------------------------------------------------------- DSP helpers
def _stretch(seg: bytes, sr: int, add_sec: float) -> bytes:
    """Lengthen one PCM segment by add_sec, pitch-preserving. Rubber Band via
    pedalboard when available; chained ffmpeg atempo as the fallback."""
    dur = (len(seg) // 2) / sr
    if dur <= 0 or add_sec <= 0:
        return seg
    try:
        import numpy as np
        from pedalboard import time_stretch
        factor = (dur + add_sec) / dur                     # >1 = longer
        x = np.frombuffer(seg, dtype=np.int16).astype(np.float32) / 32768.0
        y = time_stretch(x[None, :], sr, stretch_factor=factor)[0]
        if len(y) < len(x):                               # library convention guard
            y = time_stretch(x[None, :], sr, stretch_factor=1.0 / factor)[0]
        return (np.clip(y, -1.0, 1.0) * 32767).astype(np.int16).tobytes()
    except Exception:                                     # noqa: BLE001
        pass
    rate = dur / (dur + add_sec)              # <1 => longer
    rates, r = [], rate
    while r < 0.5:
        rates.append(0.5)
        r *= 2
    rates.append(r)
    af = ",".join(f"atempo={x:.5f}" for x in rates)
    cmd = ["ffmpeg", "-f", "s16le", "-ar", str(sr), "-ac", "1", "-i", "pipe:0",
           "-af", af, "-f", "s16le", "pipe:1"]
    out = subprocess.run(cmd, input=seg, stdout=subprocess.PIPE,
                         stderr=subprocess.DEVNULL).stdout
    return out or seg


def _xfade_join(a: bytes, b: bytes, sr: int, ms: float = 10.0) -> bytes:
    """Join two PCM pieces with an equal-power-ish linear crossfade."""
    w = int(sr * ms / 1000)
    A = list(struct.unpack("<%dh" % (len(a) // 2), a))
    B = list(struct.unpack("<%dh" % (len(b) // 2), b))
    w = max(1, min(w, len(A), len(B)))
    mix = [int(A[len(A) - w + i] * (1 - (i + 1) / w) + B[i] * ((i + 1) / w))
           for i in range(w)]
    out = A[:len(A) - w] + mix + B[w:]
    return struct.pack("<%dh" % len(out), *out)


def _cut_to(region: bytes, sr: int, keep_sec: float) -> bytes:
    """Shorten a sustained vowel to keep_sec by removing its steady middle —
    head and tail (the transitions) are kept, the join is crossfaded."""
    n = len(region) // 2
    keep = int(keep_sec * sr)
    if keep >= n:
        return region
    w = int(0.010 * sr)
    head_n = keep // 2 + w // 2
    tail_n = keep - head_n + w
    return _xfade_join(region[:head_n * 2], region[-tail_n * 2:], sr)


def _ramp(buf: bytes, nbytes: int, rising: bool) -> bytes:
    """Linear fade over the first (rising) or last (falling) nbytes of buf.
    Only the window is unpacked, so this is cheap on long clips."""
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


def _splice(out: bytes, sb: int, eb: int, new: bytes, sr: int) -> bytes:
    """Replace out[sb:eb] with `new`, fading ~4 ms on every cut edge so the
    splice is click-free."""
    w = int(sr * 0.004) * 2
    left = _ramp(out[:sb], w, rising=False)
    right = _ramp(out[eb:], w, rising=True)
    return left + _ramp(_ramp(new, w, True), w, False) + right


def _quietest_point(pcm: bytes, sr: int, t_a: float, t_b: float) -> float:
    """Best place to open a gap between two words: the lowest-energy 10 ms
    window between the first word's end and the next word's onset."""
    a, b = int(t_a * sr), int(t_b * sr)
    if b - a < int(0.02 * sr):
        return t_b                            # no real gap: cut right at the onset
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


# ------------------------------------------------------------------ the pass
def elongate(pcm: bytes, sr: int, aligner_words, marks, breaks=None,
             break_ms: int = BREAK_MS) -> tuple[bytes, list]:
    """Apply the markup to the audio. Returns (pcm, applied) where applied is
    [(label, ms)] — a drawl's ms is its full deliberate hold, a break's is its
    silence — so the caller can keep them out of pace measurements."""
    breaks = breaks or []
    if (not marks and not breaks) or not aligner_words or not load():
        return pcm, []
    import torch
    import torchaudio

    samples = torch.tensor(struct.unpack("<%dh" % (len(pcm) // 2), pcm),
                           dtype=torch.float32) / 32768.0
    wav = samples.unsqueeze(0)
    wav16 = torchaudio.functional.resample(wav, sr, 16000) if sr != 16000 else wav
    with torch.inference_mode():
        emission, _ = _model(wav16)
        token_spans = _aligner(emission[0], _tokenizer(aligner_words))

    spf = wav16.size(1) / emission.size(1) / 16000.0     # seconds per frame
    flat, offsets, wlen, word_start, word_end, off = [], [], [], [], [], 0
    for spans in token_spans:
        offsets.append(off)
        wlen.append(len(spans))
        for sp in spans:
            flat.append((sp.start * spf, sp.end * spf))
        off += len(spans)
        word_start.append(spans[0].start * spf if spans else (flat[-1][0] if flat else 0.0))
        word_end.append(spans[-1].end * spf if spans else (flat[-1][1] if flat else 0.0))
    clip_end = len(pcm) / (2 * sr)

    def gap_after(awi: int) -> float:
        nxt = word_start[awi + 1] if awi + 1 < len(word_start) else clip_end
        return _quietest_point(pcm, sr, word_end[awi], max(word_end[awi], nxt))

    edits = []                                   # (t0, t1, new_bytes, label, ms)
    for awi, fi, copies, stars in marks:
        if awi >= len(offsets):
            continue
        gi = offsets[awi] + fi
        g_end = gi + copies
        if gi >= len(flat):
            continue
        t0 = flat[gi][0]
        if g_end < offsets[awi] + wlen[awi]:         # a consonant follows the vowel
            t1 = flat[g_end][0]
        else:                                        # vowel ends the word
            t1 = gap_after(awi)
        if t1 <= t0:
            continue
        actual = t1 - t0
        target = (VOWEL_BASE_MS + stars * MS_PER_STAR) / 1000.0
        sb, eb = int(t0 * sr) * 2, int(t1 * sr) * 2
        region = pcm[sb:eb]
        if actual > target + 0.03:
            new = _cut_to(region, sr, target)
        elif actual < target - 0.03:
            new = _stretch(region, sr, target - actual)
        else:
            new = region
        edits.append((t0, t1, new, aligner_words[awi], int(target * 1000)))
    for awi, n in breaks:
        if awi < len(word_end):
            t = gap_after(awi)
            edits.append((t, t, b"\x00\x00" * int(n * break_ms / 1000.0 * sr),
                          aligner_words[awi] + "|", n * break_ms))

    applied, out = [], pcm
    for t0, t1, new, label, ms in sorted(edits, key=lambda e: e[0], reverse=True):
        sb, eb = int(t0 * sr) * 2, int(t1 * sr) * 2   # last-to-first keeps offsets valid
        out = _splice(out, sb, eb, new, sr)
        applied.append((label, ms))
    return out, list(reversed(applied))
