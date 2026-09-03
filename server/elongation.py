# -*- coding: utf-8 -*-
"""Deterministic vowel elongation.

Model-side vowel spelling ('dáááva') is stochastic — the model holds the vowel
for anywhere from 0.3s to 13s on the same input. So we do the drawl in POST,
deterministically:

  1. The author marks a vowel with stars: `Brá**cho` — each star adds ~150 ms of
     hold to THAT vowel. Stars are stripped before the model ever sees the text.
  2. Bag speaks the line clean (reliable).
  3. MMS forced alignment finds exactly where each vowel lands in the audio.
  4. We time-stretch only the marked vowel's segment (pitch-preserving), so the
     drawl is exactly on the syllable you marked, exactly as long — every time.

CPU-only (the GPUs are full with the model); alignment of a short line is fast.
"""
from __future__ import annotations

import re
import struct
import subprocess

MS_PER_STAR = 150
MAX_STARS = 6
BREAK_MS = 250                 # one break = a short, deterministic silence; they stack
BREAK_CHARS = {"\t", "—", "–", "…"}   # TAB, em/en dash, ellipsis (after _norm_breaks)
_HYPHEN_BREAK = re.compile(r"\s[-–]\s")


def _norm_breaks(text: str) -> str:
    """People (and LLMs) write pauses as '...' and ' - '; fold them into the
    canonical break characters so they behave like TAB / em dash."""
    return _HYPHEN_BREAK.sub(" — ", text.replace("...", "…"))

# Slovak diacritics -> ASCII, 1:1 at the character level so vowel positions are
# preserved. MMS_FA aligns romanized text; we keep our own fold to map back.
_FOLD = {"á": "a", "ä": "a", "é": "e", "í": "i", "ó": "o", "ô": "o", "ú": "u",
         "ý": "y", "ĺ": "l", "ŕ": "r", "č": "c", "š": "s", "ž": "z", "ť": "t",
         "ď": "d", "ň": "n", "ľ": "l"}
_VOWELS = set("aeiouy")

_model = _tokenizer = _aligner = None
_load_error = None


def load():
    """Load MMS_FA once. Safe to call repeatedly; errors are captured so the
    service still runs (elongation just becomes a no-op) if torch is missing."""
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


def parse_marks(text: str):
    """Pull the markup out of the text. Returns
    (clean_text_for_tts, aligner_words, marks, breaks) where
      marks  = [(aligner_word_index, vowel_index_in_word, stars)]  -> stretch
      breaks = [(aligner_word_index, count)]  -> count x BREAK_MS silence AFTER it
    Stars and break chars are stripped so the model speaks a clean line."""
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
            if cur and cur[-1] in _VOWELS:  # word index = the one being built
                marks.append((len(aligner_words), len(cur) - 1, min(j - i, MAX_STARS)))
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


def _stretch(seg: bytes, sr: int, add_sec: float) -> bytes:
    """Lengthen one PCM segment by add_sec, pitch-preserving (chained atempo,
    which floors at 0.5 per pass)."""
    dur = (len(seg) // 2) / sr
    if dur <= 0:
        return seg
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
    splice is click-free (a hard PCM cut at non-zero amplitude clicks)."""
    w = int(sr * 0.004) * 2
    left = _ramp(out[:sb], w, rising=False)
    right = _ramp(out[eb:], w, rising=True)
    return left + _ramp(_ramp(new, w, True), w, False) + right


def elongate(pcm: bytes, sr: int, aligner_words, marks, breaks=None,
             break_ms: int = BREAK_MS) -> tuple[bytes, list]:
    """Apply the markup to the audio: stretch marked vowels, insert a short
    silence after each break word. Returns (pcm, applied) for logging. No-ops
    cleanly when there is nothing to do or alignment is unavailable."""
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
    flat, offsets, word_end, off = [], [], [], 0
    for spans in token_spans:
        offsets.append(off)
        for sp in spans:
            flat.append((sp.start * spf, sp.end * spf))
        off += len(spans)
        word_end.append(spans[-1].end * spf if spans else (flat[-1][1] if flat else 0.0))

    # edits: ("stretch", t0, t1, add_sec, label) / ("break", t, silence_sec, label)
    edits = []
    for awi, fi, stars in marks:
        if awi >= len(offsets):
            continue
        gi = offsets[awi] + fi
        if gi >= len(flat):
            continue
        start = flat[gi][0]
        end = flat[gi + 1][0] if gi + 1 < len(flat) else flat[gi][1]
        if end <= start:
            end = flat[gi][1]
        edits.append(("stretch", start, end, stars * MS_PER_STAR / 1000.0, aligner_words[awi]))
    for awi, n in breaks:
        if awi < len(word_end):
            edits.append(("break", word_end[awi], word_end[awi],
                          n * break_ms / 1000.0, aligner_words[awi]))

    applied, out = [], pcm
    # apply last-to-first so earlier byte offsets stay valid; every cut is faded
    for kind, t0, t1, amt, label in sorted(edits, key=lambda e: e[1], reverse=True):
        sb = int(t0 * sr) * 2
        if kind == "stretch":
            eb = int(t1 * sr) * 2
            seg = out[sb:eb]
            if len(seg) < 2:
                continue
            out = _splice(out, sb, eb, _stretch(seg, sr, amt), sr)
            applied.append((label, int(amt * 1000)))
        else:
            out = _splice(out, sb, sb, b"\x00\x00" * int(amt * sr), sr)
            applied.append((label + "|", int(amt * 1000)))
    return out, list(reversed(applied))
