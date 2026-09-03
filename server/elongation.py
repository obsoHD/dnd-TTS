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

import struct
import subprocess

MS_PER_STAR = 150
MAX_STARS = 6

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
    """Pull the star markup out. Returns (clean_text_for_tts, aligner_words,
    marks) where each mark is (aligner_word_index, vowel_index_in_word, stars)."""
    disp_words, folded_words, raw_marks = [], [], []
    for token in text.split():
        disp, folded, wm = [], [], []
        i = 0
        while i < len(token):
            c = token[i]
            if c == "*":
                j = i
                while j < len(token) and token[j] == "*":
                    j += 1
                stars = min(j - i, MAX_STARS)
                if folded and folded[-1] in _VOWELS:
                    wm.append((len(folded) - 1, stars))
                i = j
                continue
            disp.append(c)
            fc = _fold_char(c)
            if "a" <= fc <= "z":          # aligner dictionary is ASCII a-z only
                folded.append(fc)
            i += 1
        disp_words.append("".join(disp))
        folded_words.append("".join(folded))
        for fi, stars in wm:
            raw_marks.append((len(folded_words) - 1, fi, stars))

    tts_text = " ".join(disp_words)
    aligner_words, idx_map = [], {}
    for k, fw in enumerate(folded_words):
        if fw:
            idx_map[k] = len(aligner_words)
            aligner_words.append(fw)
    marks = [(idx_map[wk], fi, s) for (wk, fi, s) in raw_marks if wk in idx_map]
    return tts_text, aligner_words, marks


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


def elongate(pcm: bytes, sr: int, aligner_words, marks) -> tuple[bytes, list]:
    """Stretch each marked vowel in place. Returns (pcm, applied) where applied
    lists (word, +ms) for logging. No-ops cleanly if alignment is unavailable."""
    if not marks or not aligner_words or not load():
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

    sec_per_frame = wav16.size(1) / emission.size(1) / 16000.0
    flat, offsets, off = [], [], 0                 # flat token times across line
    for spans in token_spans:
        offsets.append(off)
        for sp in spans:
            flat.append((sp.start * sec_per_frame, sp.end * sec_per_frame))
        off += len(spans)

    regions = []                                   # (start_sec, end_sec, add_sec)
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
        regions.append((start, end, stars * MS_PER_STAR / 1000.0, awi))

    applied = []
    out = pcm
    for start, end, add, awi in sorted(regions, reverse=True):  # last-to-first
        sb, eb = int(start * sr) * 2, int(end * sr) * 2
        seg = out[sb:eb]
        if len(seg) < 2:
            continue
        out = out[:sb] + _stretch(seg, sr, add) + out[eb:]
        applied.append((aligner_words[awi], int(add * 1000)))
    return out, list(reversed(applied))
