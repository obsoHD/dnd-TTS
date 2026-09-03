"""Lock a reference clip as a new voice, then calibrate it.

The Lab wizard of M5 will do this from the browser; until then this is the
supported way to add a voice, and the only way that guarantees the same
lock -> calibrate -> golden path that Bag went through.

    python -m scripts.new_voice --id shopkeep --label "Crazy Shopkeep" \
        --ref /refs/shopkeep_ref.wav --transcript-file /refs/shopkeep_ref.txt \
        --f0 85 175 --lang sk --calibrate

Calibration lines default to the voice's own bank lines, so a voice is measured
on the kind of sentence it will actually speak at the table.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from app import config, store, voices

CALIBRATION_LINES = 30


def bank_lines(voice_id: str, lang: str, limit: int = CALIBRATION_LINES) -> list[str]:
    """The voice's own bank lines, longest first.

    WHY longest first: the gain solver needs clips the loudness meter can gate
    (over 400 ms) and the similarity baseline is honest only on real sentences.
    """
    rows = store.db().execute(
        "SELECT text FROM lines WHERE voice_id = ? AND lang = ? AND source = 'bank'",
        (voice_id, lang)).fetchall()
    texts = sorted({r["text"] for r in rows}, key=len, reverse=True)
    if not texts:
        raise SystemExit(f"no bank lines for {voice_id}/{lang}; pass --lines-file")
    return texts[:limit]


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="new_voice", description=__doc__.splitlines()[0])
    p.add_argument("--id", required=True)
    p.add_argument("--label", required=True)
    p.add_argument("--ref", required=True, type=Path, help="reference clip readable from this container")
    p.add_argument("--transcript-file", type=Path, help="what the clip says, verbatim")
    p.add_argument("--transcript", help="alternative to --transcript-file")
    p.add_argument("--lang", default="sk")
    p.add_argument("--f0", nargs=2, type=int, metavar=("LO", "HI"), required=True,
                   help="expected median-pitch band in Hz; the golden test fails outside it")
    p.add_argument("--persona-file", type=Path, help="JSON with sk/en persona strings")
    p.add_argument("--energy", type=int, default=65)
    p.add_argument("--calibrate", action="store_true")
    p.add_argument("--lines-file", type=Path, help="JSON list of calibration lines")
    a = p.parse_args(argv)

    transcript = a.transcript or (a.transcript_file.read_text(encoding="utf-8").strip()
                                  if a.transcript_file else None)
    if not transcript:
        raise SystemExit("a reference needs its transcript (--transcript or --transcript-file)")

    store.init_db()
    v = voices.lock_reference(a.id, a.ref, transcript)
    v.label = a.label
    v.lang = a.lang
    v.f0_band = [int(a.f0[0]), int(a.f0[1])]
    v.master["energy"] = a.energy
    if a.persona_file:
        v.persona = json.loads(a.persona_file.read_text(encoding="utf-8"))
    voices.save_voice(v)
    print(f"locked     {v.id} v{v.version}  sha {v.ref_sha256[:16]}  -> {v.ref_tts_path}")

    if not a.calibrate:
        return 0
    lines = (json.loads(a.lines_file.read_text(encoding="utf-8")) if a.lines_file
             else bank_lines(a.id, a.lang))
    print(f"calibrate  {len(lines)} lines x 3 takes")
    v = voices.calibrate(v, lines)
    print(f"gate       {v.gate}")
    print(f"gain_db    {v.master['gain_db']:+.2f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
