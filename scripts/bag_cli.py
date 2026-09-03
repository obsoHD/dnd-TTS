"""M1 front door: render, calibrate, audition and golden-test a voice from a shell.

Run as ``python -m scripts.bag_cli <command> ...`` from the repo root. Every
command goes through the ``app`` contract modules, so what the CLI hears is
exactly what the table will serve in M2 (REBUILD.md §8, M1 acceptance).
"""
from __future__ import annotations

import argparse
import json
import sys
import warnings
from pathlib import Path
from statistics import mean

import numpy as np
import requests
import urllib3

from app import config, gate, render, store, tts_client, voices

REPO = Path(__file__).resolve().parents[1]
GOLDEN_LUFS_TOLERANCE = 2.0
WORD_RATIO = (0.75, 1.25)
SEC_PER_SYLLABLE = (0.12, 0.35)
STT_TIMEOUT_S = 30.0        # offline check: unlike the gate, waiting for whisper costs nothing here


def golden_lines_path(voice_id: str) -> Path:
    return REPO / "tests" / "golden" / voice_id / "lines.json"


def load_lines(path: Path) -> list[str]:
    """Bank line files come as a list of strings, a list of ``{text, ...}``
    rows, or ``{"lines": [...]}``; all three yield the texts in file order."""
    data = json.loads(path.read_text(encoding="utf-8"))
    rows = data.get("lines", []) if isinstance(data, dict) else data
    return [row["text"] if isinstance(row, dict) else str(row) for row in rows]


def require_tts() -> None:
    if not tts_client.health():
        sys.exit(f"TTS not healthy at {config.TTS_URL}")


def cmd_render(args: argparse.Namespace) -> int:
    require_tts()
    store.init_db()
    v = voices.load_voice(args.voice)
    r = render.render_line(v, args.text, take_no=args.take)
    line_id = store.upsert_line(v.id, v.lang, "improv", args.text, "improv")
    raw_path, path = store.write_render_files(r)
    store.put_render(r, line_id, raw_path, path)
    if r.gate == "pass":
        store.set_active_render(line_id, r.render_id)
    if args.out:
        store.write_wav(Path(args.out), r.pcm, r.sr)
    print(f"render_id  {r.render_id}")
    print(f"seed       {r.seed}  take {r.take_no}  gate {r.gate}  verified {r.verified}")
    print(f"sim        {r.sim:.3f}  cer {_fmt(r.cer)}  lufs {_fmt(r.lufs)}")
    print(f"timings    {json.dumps({k: round(t, 3) if isinstance(t, float) else t for k, t in r.timings.items()})}")
    print(f"files      {raw_path}\n           {path}")
    return 0


def cmd_calibrate(args: argparse.Namespace) -> int:
    require_tts()
    v = voices.load_voice(args.voice)
    lines = load_lines(Path(args.lines) if args.lines else golden_lines_path(v.id))
    v = voices.calibrate(v, lines)
    g = v.gate
    print(f"takes      {len(lines)} lines x 3")
    print(f"baseline   {g['baseline']:.4f}  p10 {g['p10']:.4f}")
    print(f"strict     {g['strict']:.4f}  loose {g['loose']:.4f}")
    print(f"gain_db    {v.master['gain_db']:+.2f}  (median -> {voices.TARGET_LUFS} LUFS)")
    print(f"written    {voices.voice_yaml(v.id)}")
    return 0


def parse_seeds(spec: str) -> list[int]:
    """``0-9`` or ``1,4,7``."""
    if "-" in spec:
        lo, hi = spec.split("-", 1)
        return list(range(int(lo), int(hi) + 1))
    return [int(s) for s in spec.split(",") if s.strip()]


def cmd_audition(args: argparse.Namespace) -> int:
    """Render the first N bank lines with every candidate seed, bare, into files
    named for listening; SIM per seed is printed only to break ties by ear (§2.2)."""
    require_tts()
    v = voices.load_voice(args.voice)
    lines = load_lines(Path(args.lines_file) if args.lines_file else golden_lines_path(v.id))[: args.lines]
    seeds = parse_seeds(args.seeds)
    speaker = render.speaker_gate_for(v)
    out_dir = config.DATA_DIR / "audition" / v.id
    sims: dict[int, list[float]] = {seed: [] for seed in seeds}
    for i, text in enumerate(lines):
        c = render.canon.canonicalize(text, lang=v.lang, banned=set(v.banned_tokens))
        takes = tts_client.synth_many(c.text, v.ref_tts_path, v.ref_transcript, seeds, v.sampler,
                                      render.max_new_tokens(c.syllables))
        for seed, pcm, sr in takes:
            sims[seed].append(speaker.similarity(pcm, sr))
            store.write_wav(out_dir / f"seed{seed}_{i}.wav", render.mastered(pcm, sr, v.master), sr)
        print(f"line {i:2d}  {text[:60]}")
    print(f"\nwritten    {out_dir}")
    for seed, values in sorted(sims.items(), key=lambda kv: -mean(kv[1])):
        print(f"seed {seed:3d}  mean sim {mean(values):.3f}  min {min(values):.3f}")
    return 0


def heard(pcm: bytes, sr: int, lang: str) -> str | None:
    """Whisper transcript for the word-ratio check. The gate keeps its transcript
    to itself and answers within the table's 2 s budget; the golden run can wait."""
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", urllib3.exceptions.InsecureRequestWarning)
            r = requests.post(config.STT_URL, timeout=STT_TIMEOUT_S, verify=False,
                              files={"audio": ("take.wav", store.wav_bytes(pcm, sr), "audio/wav")},
                              data={"lang": lang})
        r.raise_for_status()
        return str((r.json() or {}).get("text", "")).strip()
    except (requests.RequestException, ValueError):
        return None


def f0_median_hz(pcm: bytes, sr: int) -> float:
    """Median voiced F0 by probabilistic YIN; 0.0 when nothing voiced was found."""
    import librosa  # slow import, only the golden run needs it

    y = np.frombuffer(pcm, dtype=np.int16).astype(np.float32) / 32768.0
    f0, voiced, _ = librosa.pyin(y, fmin=50, fmax=500, sr=sr)
    voiced_f0 = f0[voiced & ~np.isnan(f0)]
    return float(np.median(voiced_f0)) if voiced_f0.size else 0.0


def golden_failures(v: voices.Voice, r: render.RenderResult, transcript: str | None, f0: float,
                    sec_per_syl: float) -> list[str]:
    """Every golden rule that this render breaks, named so the table says why."""
    fails = []
    if r.sim < v.gate["strict"]:
        fails.append(f"sim<{v.gate['strict']:.3f}")
    if transcript is None:
        fails.append("stt down")
    else:
        if r.cer is None or r.cer > v.gate.get("cer_max", 0.15):
            fails.append("cer")
        if not WORD_RATIO[0] <= gate.word_ratio(r.canon.spoken, transcript) <= WORD_RATIO[1]:
            fails.append("word ratio")
    if not v.f0_band[0] <= f0 <= v.f0_band[1]:
        fails.append("f0")
    if r.lufs is None or abs(r.lufs - voices.TARGET_LUFS) > GOLDEN_LUFS_TOLERANCE:
        fails.append("lufs")
    if not SEC_PER_SYLLABLE[0] <= sec_per_syl <= SEC_PER_SYLLABLE[1]:
        fails.append("s/syl")
    return fails


def cmd_golden(args: argparse.Namespace) -> int:
    require_tts()
    v = voices.load_voice(args.voice)
    lines = load_lines(golden_lines_path(v.id))
    print(f"{'#':>2}  {'sim':>5}  {'cer':>5}  {'ratio':>5}  {'f0':>5}  {'lufs':>6}  {'s/syl':>5}  {'gate':6}  result")
    failed = 0
    for i, text in enumerate(lines):
        r = render.render_line(v, text)
        transcript = heard(r.pcm, r.sr, v.lang)
        f0 = f0_median_hz(r.pcm, r.sr)
        sps = store.duration_s(r.pcm, r.sr) / r.canon.syllables
        fails = golden_failures(v, r, transcript, f0, sps)
        failed += bool(fails)
        ratio = gate.word_ratio(r.canon.spoken, transcript) if transcript is not None else None
        print(f"{i:2d}  {r.sim:5.3f}  {_fmt(r.cer):>5}  {_fmt(ratio):>5}  {f0:5.0f}  {_fmt(r.lufs):>6}  "
              f"{sps:5.2f}  {r.gate:6}  {'ok' if not fails else ', '.join(fails)}")
    print(f"\n{len(lines) - failed}/{len(lines)} passed")
    return 1 if failed else 0


def _fmt(value: float | None) -> str:
    return "-" if value is None else f"{value:.2f}"


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="bag_cli", description=__doc__.splitlines()[0])
    sub = p.add_subparsers(dest="command", required=True)

    r = sub.add_parser("render", help="render one line through the full pipeline and store it")
    r.add_argument("--voice", required=True)
    r.add_argument("--text", required=True)
    r.add_argument("--take", type=int, default=0, help="take number (Regenerate = previous + 1)")
    r.add_argument("--out", help="also write the mastered wav here")
    r.set_defaults(run=cmd_render)

    c = sub.add_parser("calibrate", help="measure SIM thresholds and the fixed gain; writes voice.yaml")
    c.add_argument("--voice", required=True)
    c.add_argument("--lines", help="lines.json (default: tests/golden/<voice>/lines.json)")
    c.set_defaults(run=cmd_calibrate)

    a = sub.add_parser("audition", help="render lines x seeds into DATA_DIR/audition/<voice> for listening")
    a.add_argument("--voice", required=True)
    a.add_argument("--seeds", default="0-9", help="range a-b or comma list")
    a.add_argument("--lines", type=int, default=10, help="how many bank lines to use")
    a.add_argument("--lines-file", help="lines.json (default: tests/golden/<voice>/lines.json)")
    a.set_defaults(run=cmd_audition)

    g = sub.add_parser("golden", help="render the golden set and check every rule; non-zero exit on failure")
    g.add_argument("--voice", required=True)
    g.set_defaults(run=cmd_golden)
    return p


def main(argv: list[str] | None = None) -> int:
    # Windows consoles default to a legacy code page and every bank line is Slovak.
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    args = build_parser().parse_args(argv)
    return int(args.run(args))


if __name__ == "__main__":
    sys.exit(main())
