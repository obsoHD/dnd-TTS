"""A voice is its locked reference clip plus the numbers calibrated from it.

``voice.yaml`` is the single source of truth per voice (REBUILD.md §2.1, §6):
the reference is immutable once locked (its sha256 is recorded and checked
before every render), a new clip bumps ``version`` so every cached render_id
changes, and calibration writes the gate thresholds and the fixed gain here.
Nothing in this file talks to the GPU except ``calibrate``, which goes through
``app.render`` so the takes it measures are exactly the takes the table hears.
"""
from __future__ import annotations

import hashlib
import math
import shutil
import subprocess
from dataclasses import asdict, dataclass, field, fields
from datetime import datetime, timezone
from pathlib import Path
from statistics import median

import numpy as np
import yaml

from app import config
from app.tts_client import Sampler

TARGET_LUFS = -18.0
REF_SR = 24_000
REF_MAX_S = 30.0
REF_TRIM_DBFS = -42.0
REF_PEAK_DBFS = -3.0
REF_EDGE_KEEP_S = 0.05      # a hair of silence kept at each end so the clip does not start on a click
UNCALIBRATED_GATE = {"mode": "strict", "baseline": 0.0, "p10": 0.0, "strict": 0.0, "loose": 0.0,
                     "cer_max": 0.15, "calibrated_at": None}


class ReferenceMismatch(RuntimeError):
    """ref.wav on disk is not the clip that was locked; render refuses until re-lock."""


class CalibrationError(RuntimeError):
    """Calibration produced no usable take, so no thresholds were written."""


@dataclass
class Voice:
    id: str
    label: str
    lang: str
    version: int
    ref_file: str
    ref_sha256: str
    ref_transcript: str
    ref_tts_path: str
    sampler: Sampler
    golden_seed: int
    gate: dict
    master: dict
    banned_tokens: list[str] = field(default_factory=list)
    persona: dict = field(default_factory=dict)
    fillers: list[str] = field(default_factory=list)
    f0_band: list[int] = field(default_factory=lambda: [50, 420])


def voice_yaml(voice_id: str) -> Path:
    return config.voice_dir(voice_id) / "voice.yaml"


def ref_path(v: Voice) -> Path:
    return config.voice_dir(v.id) / v.ref_file


def load_voice(voice_id: str) -> Voice:
    data = yaml.safe_load(voice_yaml(voice_id).read_text(encoding="utf-8")) or {}
    return _from_mapping(data)


def save_voice(v: Voice) -> None:
    """Atomic write: a crash mid-save must never leave a half-written lock."""
    path = voice_yaml(v.id)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".yaml.tmp")
    tmp.write_text(yaml.safe_dump(asdict(v), sort_keys=False, allow_unicode=True), encoding="utf-8")
    tmp.replace(path)


def _from_mapping(d: dict) -> Voice:
    """Accept both the flat contract keys and the nested ``reference:`` block
    REBUILD.md §6 sketches, so a hand-written seed loads either way."""
    ref = d.get("reference") or {}
    sampler_keys = {f.name for f in fields(Sampler)}
    sampler = Sampler(**{k: v for k, v in (d.get("sampler") or {}).items() if k in sampler_keys})
    gate = {**UNCALIBRATED_GATE, **(d.get("gate") or {})}
    return Voice(
        id=str(d["id"]), label=str(d.get("label", d["id"])), lang=str(d.get("lang", "sk")),
        version=int(d.get("version", 1)),
        ref_file=str(d.get("ref_file", ref.get("file", "ref.wav"))),
        ref_sha256=str(d.get("ref_sha256", ref.get("sha256", ""))),
        ref_transcript=str(d.get("ref_transcript", ref.get("transcript", ""))),
        ref_tts_path=str(d.get("ref_tts_path", ref.get("tts_path", f"/refs/{d['id']}/ref.wav"))),
        sampler=sampler, golden_seed=int(d.get("golden_seed", 0)), gate=gate,
        master=dict(d.get("master") or {}),
        banned_tokens=[str(t) for t in d.get("banned_tokens") or []],
        persona=dict(d.get("persona") or {}),
        fillers=[str(f) for f in d.get("fillers") or []],
        f0_band=[int(x) for x in d.get("f0_band") or [50, 420]],
    )


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def check_reference(v: Voice) -> Path:
    """The path of the locked clip, after proving it is still the locked clip."""
    path = ref_path(v)
    digest = sha256_file(path)
    if digest != v.ref_sha256:
        raise ReferenceMismatch(f"{path} sha256 {digest[:12]} != locked {v.ref_sha256[:12]}; re-lock the voice")
    return path


def _new_voice(voice_id: str) -> Voice:
    """Defaults for a voice that has no voice.yaml yet (the New-voice wizard path)."""
    from app import master  # pedalboard is heavy; only a brand-new voice needs DEFAULT

    return Voice(id=voice_id, label=voice_id, lang="sk", version=0, ref_file="ref.wav", ref_sha256="",
                 ref_transcript="", ref_tts_path=f"/refs/{voice_id}/ref.wav", sampler=Sampler(),
                 golden_seed=0, gate=dict(UNCALIBRATED_GATE), master={**master.DEFAULT, "energy": 65})


def _convert_reference(wav_in: Path) -> bytes:
    """ffmpeg -> mono 24 kHz int16 with head/tail silence trimmed at -42 dBFS.
    No EQ, no compression, no denoise: the clip *is* the voice (§1.1)."""
    trim = (f"silenceremove=start_periods=1:start_threshold={REF_TRIM_DBFS}dB:start_silence={REF_EDGE_KEEP_S},"
            "areverse,"
            f"silenceremove=start_periods=1:start_threshold={REF_TRIM_DBFS}dB:start_silence={REF_EDGE_KEEP_S},"
            "areverse")
    cmd = ["ffmpeg", "-nostdin", "-v", "error", "-i", str(wav_in), "-ac", "1", "-ar", str(REF_SR),
           "-af", trim, "-f", "s16le", "pipe:1"]
    return subprocess.run(cmd, stdout=subprocess.PIPE, check=True).stdout


def _peak_scale(pcm: bytes, peak_dbfs: float) -> bytes:
    """Deterministic peak normalisation in numpy (no ffmpeg dither in the loop)."""
    x = np.frombuffer(pcm, dtype=np.int16).astype(np.float32)
    peak = float(np.max(np.abs(x))) if x.size else 0.0
    if peak == 0.0:
        raise ValueError("reference clip is silent")
    y = x * (10 ** (peak_dbfs / 20) * 32767.0 / peak)
    return np.clip(np.round(y), -32768, 32767).astype(np.int16).tobytes()


def lock_reference(voice_id: str, wav_in: Path, transcript: str) -> Voice:
    """Convert, trim, cap and level a clip, then make it the voice's immutable
    reference: sha256 recorded, version bumped (every cached render_id changes),
    calibration reset (the old thresholds described the old clip). The clip is
    also copied under the ``/refs`` mount so the TTS container reads the same
    bytes at ``ref_tts_path``."""
    from app import store  # wav writer lives with the other file writers

    v = load_voice(voice_id) if voice_yaml(voice_id).exists() else _new_voice(voice_id)
    pcm = _peak_scale(_convert_reference(wav_in)[: int(REF_MAX_S * REF_SR) * 2], REF_PEAK_DBFS)
    out = config.voice_dir(voice_id) / "ref.wav"
    store.write_wav(out, pcm, REF_SR)
    _publish_reference(out, voice_id)
    v.ref_file = out.name
    v.ref_sha256 = sha256_file(out)
    v.ref_transcript = " ".join(transcript.split())
    v.ref_tts_path = f"/refs/{voice_id}/ref.wav"
    v.version += 1
    v.gate = {**v.gate, **{k: UNCALIBRATED_GATE[k] for k in ("baseline", "p10", "strict", "loose", "calibrated_at")}}
    save_voice(v)
    return v


def _publish_reference(ref: Path, voice_id: str) -> None:
    """Copy the locked clip to where the TTS container looks for it. Skipped when
    the mount is absent (a developer box without ``/refs``)."""
    if not config.REFS_DIR.is_dir():
        return
    target = config.REFS_DIR / voice_id / "ref.wav"
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(ref, target)


def calibrate(v: Voice, lines: list[str], n_takes: int = 3) -> Voice:
    """Measure the voice against itself: SIM over lines x takes gives the
    baseline and the strict/loose thresholds (paper thresholds are never used,
    §2.5); the accepted raw takes give the one fixed gain that puts the set's
    median loudness at -18 LUFS after the master chain (§2.7). Writes voice.yaml."""
    from app import render  # render imports this module; a lazy import breaks the cycle

    results = [render.render_line(v, line, take_no=0, n_takes=n_takes) for line in lines]
    sims = sorted(s.sim for r in results for s in r.scores if s.sane)
    if not sims:
        raise CalibrationError("no take passed sanity; check the TTS and the reference")
    baseline = float(median(sims))
    p10 = float(np.percentile(sims, 10))
    v.gate.update(baseline=round(baseline, 4), p10=round(p10, 4),
                  strict=round(max(baseline - 0.04, p10), 4), loose=round(baseline - 0.08, 4),
                  calibrated_at=datetime.now(timezone.utc).isoformat(timespec="seconds"))
    v.master["gain_db"] = _solve_gain([(r.raw_pcm, r.sr) for r in results], v.master)
    save_voice(v)
    return v


def _solve_gain(raws: list[tuple[bytes, int]], base: dict, target: float = TARGET_LUFS) -> float:
    """Fixed gain such that the median integrated loudness of the mastered set
    hits ``target``. Iterated a few times because the limiter after the gain
    stage makes loudness slightly sub-linear in gain."""
    from app import master, render

    # WHY trim off: the solver wants the voice's true median level; with the
    # per-line trim active every reading near the target would already sit on it.
    params = {**base, "target_lufs": target, "trim_max_db": 0.0}
    for _ in range(3):
        readings = (master.measure(render.mastered(pcm, sr, params), sr)["lufs"] for pcm, sr in raws)
        loud = [lufs for lufs in readings if math.isfinite(lufs)]   # sub-400 ms clips read -inf
        if not loud:
            raise CalibrationError("no calibration take is long enough to meter")
        delta = target - float(median(loud))
        if abs(delta) < 0.1:
            break
        params["gain_db"] = float(params.get("gain_db") or 0.0) + delta
    return round(float(params.get("gain_db") or 0.0), 2)
