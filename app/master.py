"""Deterministic finishing chain for one rendered line (REBUILD §2.7).

WHY a fixed chain: the reference clip *is* the voice. The master only makes every
take of a voice sit in the same room at the same level; nothing here adapts to the
individual line (no loudness normalisation, no internal silence editing). That is
what makes ``master()`` byte-identical for identical input and params, and what
lets an Energy change re-master the bank from the stored raw takes without the GPU.

All audio crosses the boundary as int16 mono PCM bytes; float32 is internal only.
"""
from __future__ import annotations

import numpy as np
import pyloudnorm as pyln
from pedalboard import (Compressor, Gain, HighpassFilter, HighShelfFilter, Limiter,
                        PeakFilter, Pedalboard, Reverb, time_stretch)

VERSION = "1"

DEFAULT = dict(energy=65, hpf_hz=50, presence_hz=3000, presence_q=1.0, air_hz=10000, comp_ratio=3.0,
               comp_attack_ms=5, comp_release_ms=100, comp_threshold_db=-24.0, tempo=1.0, tempo_cap=1.10,
               pitch_st=0.0, gain_db=0.0, room_size=0.22, room_damping=0.65, room_wet=0.08, limiter_dbtp=-1.0,
               fade_in_ms=8, fade_out_ms=60, lead_ms=40, tail_ms=250)

# Energy (0-100) is the only knob; these slopes are the spec's (REBUILD §2.7).
PRESENCE_DB_PER_ENERGY = 0.04
AIR_DB_PER_ENERGY = 0.025
GAIN_REDUCTION_DB_PER_ENERGY = 0.07
TEMPO_PER_ENERGY = 0.0012

TRIM_FLOOR_DBFS = -50.0   # a frame at or above this level is speech for the edge trim
FRAME_S = 0.02            # the same analysis frame as gate.sanity, so both agree on "silence"
PEAK_CEILING_DBFS = -1.0  # absolute ceiling enforced after the limiter
INT16_FULL_SCALE = 32768.0


# ------------------------------------------------------------------ parameters
def energy_to_params(energy: int, base: dict) -> dict:
    """Expand the Energy knob into concrete stage settings.

    Returns a new dict (``base`` is never mutated) carrying every key of ``base``
    plus the derived ``presence_db``, ``air_db``, ``comp_gr_db``,
    ``comp_threshold_db`` and ``tempo``. ``base`` must be a consistent block: its
    ``comp_threshold_db`` is the one calibrated at its ``energy`` (DEFAULT and a
    voice's saved master block both are). Re-applying at the same energy is a
    no-op, so an expanded dict can be stored and expanded again safely.
    """
    p = {**DEFAULT, **base}
    gr_db = GAIN_REDUCTION_DB_PER_ENERGY * energy
    p.update(
        energy=energy,
        presence_db=PRESENCE_DB_PER_ENERGY * energy,
        air_db=AIR_DB_PER_ENERGY * energy,
        comp_gr_db=gr_db,
        comp_threshold_db=_threshold_for(gr_db, p),
        tempo=min(float(p["tempo_cap"]), 1.0 + TEMPO_PER_ENERGY * energy),
    )
    return p


def _threshold_for(gr_db: float, base: dict) -> float:
    """Move the compressor threshold so its static curve delivers ``gr_db``.

    WHY relative: calibration fixes ``comp_threshold_db`` for the voice's own
    energy (about 4.5 dB median gain reduction at Energy 65). A ratio-R
    compressor reduces ``(level - threshold) * (1 - 1/R)`` dB, so a change of
    ``d`` dB in the target needs the threshold moved by ``d * R / (R - 1)`` dB.
    Anchoring on the base pair keeps the calibrated point exact and needs no
    assumed input level.
    """
    ratio = float(base["comp_ratio"])
    threshold = float(base["comp_threshold_db"])
    if ratio <= 1.0:  # no compression: the threshold has no effect either way
        return threshold
    base_gr = GAIN_REDUCTION_DB_PER_ENERGY * float(base["energy"])
    return threshold - (gr_db - base_gr) * ratio / (ratio - 1.0)


def _resolve(params: dict) -> dict:
    """Complete the params: explicit keys win, gaps are derived from ``energy``.

    WHY: render passes an expanded dict; a caller handing over a voice's master
    block untouched still gets the right EQ and compression for its energy. A
    ``tempo`` present in the dict is honoured as written (and capped in
    :func:`_stretch`), so change energy through :func:`energy_to_params`.
    """
    base = {**DEFAULT, **params}
    return {**energy_to_params(int(base["energy"]), base), **params}


# ------------------------------------------------------------------ the chain
def master(raw_pcm: bytes, sr: int, params: dict) -> bytes:
    """Run one raw take through the fixed chain; int16 PCM in, int16 PCM out.

    Order (contract): edge trim, HPF, presence bell, air shelf, compressor,
    tempo, fixed gain, room, limiter, peak ceiling, fades. Two pedalboard passes
    bracket the time-stretch because it is a function, not a plugin. Every stage
    is a fixed-coefficient filter or a pure array operation and each call builds
    fresh plugin instances, so identical input and params give identical bytes
    and no state leaks between lines.
    """
    p = _resolve(params)
    x = _decode(raw_pcm)
    if x.size == 0:
        return b""
    x = _edge_trim(x, sr, p["lead_ms"], p["tail_ms"])
    x = _run(_tone_stage(p), x, sr)
    x = _stretch(x, sr, p)
    x = _run(_room_stage(p), x, sr)
    x = _ceiling(x)
    x = _fade(x, sr, p["fade_in_ms"], p["fade_out_ms"])
    return _encode(x)


def _tone_stage(p: dict) -> Pedalboard:
    """HPF, presence bell, air shelf, compressor: the per-voice tone, before tempo."""
    return Pedalboard([
        HighpassFilter(cutoff_frequency_hz=float(p["hpf_hz"])),
        PeakFilter(cutoff_frequency_hz=float(p["presence_hz"]), gain_db=float(p["presence_db"]),
                   q=float(p["presence_q"])),
        HighShelfFilter(cutoff_frequency_hz=float(p["air_hz"]), gain_db=float(p["air_db"])),
        Compressor(threshold_db=float(p["comp_threshold_db"]), ratio=float(p["comp_ratio"]),
                   attack_ms=float(p["comp_attack_ms"]), release_ms=float(p["comp_release_ms"])),
    ])


def _room_stage(p: dict) -> Pedalboard:
    """Fixed gain, the shared room, limiter: every voice in one space at its calibrated level.

    WHY dry = 1 - wet: the room must not change the level of the voice, only add
    its space, so the two levels always sum to unity.
    """
    wet = float(p["room_wet"])
    return Pedalboard([
        Gain(gain_db=float(p["gain_db"])),
        Reverb(room_size=float(p["room_size"]), damping=float(p["room_damping"]),
               wet_level=wet, dry_level=1.0 - wet),
        Limiter(threshold_db=float(p["limiter_dbtp"])),
    ])


def _run(board: Pedalboard, x: np.ndarray, sr: int) -> np.ndarray:
    """Process a mono float32 signal; pedalboard wants (channels, samples)."""
    return board(x[None, :], sr)[0]


def _stretch(x: np.ndarray, sr: int, p: dict) -> np.ndarray:
    """Tempo (and any pitch offset) in one Rubber Band pass, only when needed.

    WHY the guard: ``time_stretch`` resynthesises even at 1.0 and would alter the
    bytes for nothing. The cap is enforced here as well as in
    :func:`energy_to_params` so a pinned tempo can never exceed the voice's limit.
    ``preserve_formants`` keeps the voice's size when the pitch moves;
    ``high_quality`` selects the offline-grade engine.
    """
    tempo = min(float(p["tempo"]), float(p["tempo_cap"]))
    pitch = float(p["pitch_st"])
    if tempo == 1.0 and pitch == 0.0:
        return x
    return time_stretch(x[None, :], sr, stretch_factor=tempo, pitch_shift_in_semitones=pitch,
                        high_quality=True, preserve_formants=True)[0]


def _edge_trim(x: np.ndarray, sr: int, lead_ms: float, tail_ms: float) -> np.ndarray:
    """Cut leading/trailing silence, keeping ``lead_ms`` before the onset and ``tail_ms`` after the offset.

    Onset and offset are the first and last 20 ms frames whose RMS reaches
    -50 dBFS. WHY frames: one dithered sample must not count as speech. WHY no
    internal editing: the pauses are the performance (REBUILD §2.7). All-silent
    input is returned untouched: there is nothing to anchor a cut on, and the
    gate has already rejected it.
    """
    hop = _samples(sr, FRAME_S * 1000)
    n = x.size // hop
    if n == 0:
        return x
    frames = x[: n * hop].reshape(n, hop)
    level_db = 10.0 * np.log10(np.mean(frames * frames, axis=1) + 1e-12)
    loud = np.flatnonzero(level_db >= TRIM_FLOOR_DBFS)
    if loud.size == 0:
        return x
    start = max(0, int(loud[0]) * hop - _samples(sr, lead_ms))
    end = min(x.size, (int(loud[-1]) + 1) * hop + _samples(sr, tail_ms))
    return x[start:end]


def _ceiling(x: np.ndarray) -> np.ndarray:
    """Exact peak ceiling after the limiter.

    WHY: pedalboard's Limiter is not a brickwall (fast transients overshoot), and
    clipping would distort. A single scale-down of the whole line guarantees the
    ceiling, is deterministic, and leaves the line's dynamics untouched.
    """
    limit = _lin(PEAK_CEILING_DBFS)
    peak = float(np.max(np.abs(x)))
    return x * np.float32(limit / peak) if peak > limit else x


def _fade(x: np.ndarray, sr: int, fade_in_ms: float, fade_out_ms: float) -> np.ndarray:
    """Linear edge ramps.

    WHY: the trim cuts on frame boundaries and the room leaves energy at the end;
    the ramps stop clicks on both edges without touching the body of the line.
    """
    n_in = min(_samples(sr, fade_in_ms), x.size)
    n_out = min(_samples(sr, fade_out_ms), x.size)
    y = x.copy()
    if n_in > 1:
        y[:n_in] *= np.linspace(0.0, 1.0, n_in, dtype=np.float32)
    if n_out > 1:
        y[-n_out:] *= np.linspace(1.0, 0.0, n_out, dtype=np.float32)
    return y


# ------------------------------------------------------------------ measuring
def measure(pcm: bytes, sr: int) -> dict:
    """Peak (dBFS), integrated loudness (LUFS, ITU-R BS.1770 via pyloudnorm) and duration (s).

    WHY here: calibration and the golden test judge the *mastered* output with
    this meter, so its definition lives next to the chain. Silence, and clips
    shorter than the meter's 400 ms gating block, read -inf LUFS, which is the
    meter's own value for "nothing above the gate".
    """
    x = _decode(pcm)
    peak = float(np.max(np.abs(x))) if x.size else 0.0
    return {"peak_dbfs": _db(peak), "lufs": _lufs(x, sr), "dur_s": x.size / sr}


def _lufs(x: np.ndarray, sr: int) -> float:
    meter = pyln.Meter(sr)
    if x.size < meter.block_size * sr:
        return float("-inf")
    return float(meter.integrated_loudness(x.astype(np.float64)))


# ------------------------------------------------------------------ conversions
def _decode(pcm: bytes) -> np.ndarray:
    """int16 bytes to float32 in [-1, 1). A dangling odd byte cannot be a sample and is dropped."""
    usable = pcm[: len(pcm) - (len(pcm) % 2)]
    return np.frombuffer(usable, dtype=np.int16).astype(np.float32) / INT16_FULL_SCALE


def _encode(x: np.ndarray) -> bytes:
    """float32 to int16 bytes; rounding (not truncation) keeps quantisation symmetric."""
    return np.rint(np.clip(x, -1.0, 1.0) * (INT16_FULL_SCALE - 1.0)).astype(np.int16).tobytes()


def _samples(sr: int, ms: float) -> int:
    return int(round(sr * ms / 1000.0))


def _lin(db: float) -> float:
    return 10.0 ** (db / 20.0)


def _db(amplitude: float) -> float:
    return 20.0 * np.log10(amplitude) if amplitude > 0.0 else float("-inf")
