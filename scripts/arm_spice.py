"""Lab tool: measure a spice against the golden set before it may be armed.

Run as ``python -m scripts.arm_spice --voice <id> [--spice ID ...] [--lines N] [--apply]``
from the repo root.

Why bare and spiced are always rendered together, in the same run, and never against a
stored baseline (docs/M3-contracts.md; the rule that governs this milestone: identity
outranks expressiveness): the box, the model and the master chain drift together, so a
spice can only be judged fairly against *this run's own* bare numbers. Comparing against
yesterday's calibration would blame the spice for drift that was never its fault. Nothing
here runs inside the render path -- this script only writes ``voice.yaml``; the API reads
the ``armed_spices`` block back through ``app/delivery.py``.

The spice catalog (``SPICES``, ``MAX_ARMED``) is owned by ``app/delivery.py`` and imported
from there so the Lab and the API can never drift onto two different lists. ``apply`` is
not imported: it refuses anything not already armed, which is exactly what this script
exists to measure before it is armed, so token placement is reimplemented here against the
same contract rule (§3) instead.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from statistics import median

from app import config, render, store, tts_client, voices
from app.delivery import MAX_ARMED, SPICES, Spice

REPO = Path(__file__).resolve().parents[1]
DEFAULT_LINES = 20
MAX_DROP = 0.02         # median SIM drop, bare -> spiced, above which a spice is refused.

_BY_ID = {s.id: s for s in SPICES}
# The first word plus the whitespace that follows it, then everything else; DOTALL so a
# pasted two-line note still finds its second word. Mirrors app.delivery.apply's rule (§3)
# without importing its arming gate -- see the module docstring.
_AFTER_FIRST_WORD = re.compile(r"(\S+\s+)(\S.*)", re.DOTALL)
_TOKEN = re.compile(r"<\|[^|<>]*\|>")


@dataclass
class Row:
    spice: Spice
    n: int
    median_bare: float | None
    median_spiced: float | None
    drop: float | None
    min_sim: float | None
    verdict: str            # armed | drop | floor | thin | refused
    reason: str = ""


def golden_lines_path(voice_id: str) -> Path:
    return REPO / "tests" / "golden" / voice_id / "lines.json"


def load_lines(path: Path) -> list[str]:
    """Bank line files come as a list of strings, a list of ``{text, ...}`` rows,
    or ``{"lines": [...]}``; all three yield the texts in file order (mirrors
    scripts/bag_cli.py, which this script does not import to stay self-contained)."""
    data = json.loads(path.read_text(encoding="utf-8"))
    rows = data.get("lines", []) if isinstance(data, dict) else data
    return [row["text"] if isinstance(row, dict) else str(row) for row in rows]


def voice_lines(v: voices.Voice) -> list[str]:
    """The voice's golden set, or its own bank lines when it has none.

    WHY the fallback: a voice made in the Creator has a soundboard long before
    anyone writes it a golden file, and a delivery must still be measured on
    sentences that voice actually speaks.
    """
    path = golden_lines_path(v.id)
    if path.exists():
        return load_lines(path)
    rows = store.db().execute(
        "SELECT text FROM lines WHERE voice_id=? AND lang=? AND source='bank' ORDER BY created, rowid",
        (v.id, v.lang)).fetchall()
    return [r["text"] for r in rows]


def require_tts() -> None:
    if not tts_client.health():
        sys.exit(f"TTS not healthy at {config.TTS_URL}")


def spice_text(line: str, token: str) -> str:
    """Attach the token after the first word (docs/M3-contracts.md, `app/delivery.py`
    rule 3): ``Ten nie.`` -> ``Ten <|token|>nie.``. A one-word line has no second word
    to attach to, and a line that already carries a token is left alone -- either way
    the bare and "spiced" renders end up identical, which correctly contributes zero
    signal rather than a fabricated one."""
    if _TOKEN.search(line):
        return line
    lead = line[: len(line) - len(line.lstrip())]
    match = _AFTER_FIRST_WORD.fullmatch(line.lstrip())
    if match is None:
        return line
    first_word, rest = match.groups()
    return f"{lead}{first_word}{token}{rest}"


def refusal(spice: Spice, voice: voices.Voice) -> str | None:
    """Why a spice may never be armed, or ``None`` when it is eligible. ``krik`` is
    refused outright for every voice (REBUILD.md §3); anything the voice already
    forbids is refused for that voice specifically."""
    if spice.id == "krik":
        return "krik is forbidden outright"
    key = spice.token.strip("<|>").lower()
    banned = {t.strip("<|>").lower() for t in voice.banned_tokens}
    if key in banned or key.rsplit(":", 1)[-1] in banned:
        return f"{spice.token} is on voice.banned_tokens"
    return None


MIN_PAIRS = 10          # below this the measurement says nothing about the spice


def evaluate(spice: Spice, bare: list[float], spiced: list[float] | None, reason: str | None,
            strict: float) -> Row:
    """One candidate's verdict, measured on paired takes of the same lines.

    Promote only when the median drop is at most MAX_DROP and no *spiced* take
    fell below ``strict`` (docs/M3-contracts.md). Lines whose bare take is
    already under ``strict`` are dropped from the comparison first: the voice
    cannot say those plainly either, so counting them against a delivery blames
    the wrong thing -- and because the bare run is shared, one such line would
    otherwise fail every candidate with an identical minimum.
    """
    if reason is not None:
        return Row(spice, 0, None, None, None, None, "refused", reason)
    pairs = [(b, sp) for b, sp in zip(bare, spiced) if b >= strict]
    dropped = len(bare) - len(pairs)
    note = f"{dropped} line(s) excluded: bare take under strict" if dropped else ""
    if len(pairs) < MIN_PAIRS:
        return Row(spice, len(pairs), median(bare), median(spiced), None, None, "thin",
                   f"only {len(pairs)} usable line(s); {note or 'too few lines'}")
    mb, ms = median([b for b, _ in pairs]), median([sp for _, sp in pairs])
    lo = min(sp for _, sp in pairs)
    drop = mb - ms
    if lo < strict:
        return Row(spice, len(pairs), mb, ms, drop, lo, "floor",
                   f"min sim {lo:.3f} < strict {strict:.3f}; {note}".rstrip("; "))
    if drop > MAX_DROP:
        return Row(spice, len(pairs), mb, ms, drop, lo, "drop",
                   f"drop {drop:.3f} > {MAX_DROP:.2f}; {note}".rstrip("; "))
    return Row(spice, len(pairs), mb, ms, drop, lo, "armed", note)


def measure(v: voices.Voice, lines: list[str], candidates: list[Spice]) -> list[Row]:
    """Bare, rendered once and shared across every candidate, then each spice's
    own takes -- all in this one run, one take per line (§ above)."""
    strict = float(v.gate["strict"])
    bare = [render.render_line(v, line, take_no=0).sim for line in lines]
    weak = sum(1 for sim in bare if sim < strict)
    if weak:
        # Worth saying out loud: these lines would fail the gate at the table too.
        print(f"note       {weak}/{len(bare)} bare take(s) under strict {strict:.3f} "
              f"(worst {min(bare):.3f}) -- excluded from every comparison")
    rows = []
    for spice in candidates:
        reason = refusal(spice, v)
        if reason is not None:
            rows.append(evaluate(spice, bare, None, reason, strict))
            continue
        spiced = [render.render_line(v, spice_text(line, spice.token), take_no=0).sim for line in lines]
        rows.append(evaluate(spice, bare, spiced, None, strict))
    return rows


def _fmt(value: float | None) -> str:
    return "-" if value is None else f"{value:.3f}"


def print_table(rows: list[Row]) -> None:
    print(f"{'spice':11}  {'n':>3}  {'bare':>5}  {'spiced':>6}  {'drop':>6}  {'min':>5}  {'verdict':7}  reason")
    for r in rows:
        print(f"{r.spice.id:11}  {r.n:3d}  {_fmt(r.median_bare):>5}  {_fmt(r.median_spiced):>6}  "
              f"{_fmt(r.drop):>6}  {_fmt(r.min_sim):>5}  {r.verdict:7}  {r.reason}")
    armed = sum(1 for r in rows if r.verdict == "armed")
    print(f"\n{armed}/{len(rows)} eligible")


def promote(existing: dict, rows: list[Row], v: voices.Voice) -> dict:
    """Fold this run's verdicts into the existing block: newest measurement wins,
    whether that means arming an id or disarming one that no longer clears the bar
    (ids not tested this run are left exactly as they were), then keep only the
    MAX_ARMED entries with the smallest drop. A defensive filter at the end refuses
    krik and anything now on voice.banned_tokens even if it was armed under an
    older, looser rule -- this script never ships a voice change."""
    merged = dict(existing)
    now = datetime.now(timezone.utc).isoformat(timespec="seconds")
    for r in rows:
        if r.verdict == "armed":
            merged[r.spice.id] = {"sim_drop": round(r.drop, 4), "min_sim": round(r.min_sim, 4),
                                   "n": r.n, "armed_at": now}
        else:
            merged.pop(r.spice.id, None)
    merged = {k: val for k, val in merged.items() if k not in _BY_ID or refusal(_BY_ID[k], v) is None}
    ranked = sorted(merged.items(), key=lambda kv: kv[1].get("sim_drop", float("inf")))
    return dict(ranked[:MAX_ARMED])


def cmd_arm(args: argparse.Namespace) -> int:
    require_tts()
    v = voices.load_voice(args.voice)
    lines = voice_lines(v)[: args.lines]
    if not lines:
        sys.exit(f"no golden or bank lines for {v.id}")
    ids = args.spice or [s.id for s in SPICES]
    unknown = [i for i in ids if i not in _BY_ID]
    if unknown:
        sys.exit(f"unknown spice id(s): {', '.join(unknown)}")
    candidates = [_BY_ID[i] for i in ids]

    print(f"voice {v.id}  lines {len(lines)}  strict {float(v.gate['strict']):.3f}\n")
    rows = measure(v, lines, candidates)
    print_table(rows)

    if args.apply:
        v.armed_spices = promote(v.armed_spices, rows, v)
        voices.save_voice(v)
        print(f"\nwritten    {voices.voice_yaml(v.id)}")
        print(f"armed      {', '.join(v.armed_spices) if v.armed_spices else '(none)'}")
    else:
        print("\ndry run -- pass --apply to write voice.yaml:armed_spices")
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="arm_spice", description=__doc__.splitlines()[0])
    p.add_argument("--voice", required=True)
    p.add_argument("--spice", nargs="+", metavar="ID", choices=[s.id for s in SPICES],
                   help="candidate spice ids (default: all of SPICES)")
    p.add_argument("--lines", type=int, default=DEFAULT_LINES, help=f"golden lines to use (default {DEFAULT_LINES})")
    p.add_argument("--apply", action="store_true", help="write the winners into voice.yaml:armed_spices")
    return p


def main(argv: list[str] | None = None) -> int:
    # Windows consoles default to a legacy code page and every bank line is Slovak.
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    args = build_parser().parse_args(argv)
    return cmd_arm(args)


if __name__ == "__main__":
    sys.exit(main())
