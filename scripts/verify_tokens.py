"""Diff configs/tokens.json against the Higgs cookbook shipped inside the TTS image.

Control tokens are parsed by the SGLang server and never appear in tokenizer.json,
so the only ground truth we can inspect is the cookbook markdown inside the running
image. Exit 0 when both lists match, 1 when they differ, 2 when the cookbook could
not be read, so a docker outage is never mistaken for a clean diff.

Usage (on the box, with bag-tts running):
    python -m scripts.verify_tokens [--container bag-tts] [--doc PATH] [--tokens configs/tokens.json]
"""
from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_TOKENS = REPO_ROOT / "configs" / "tokens.json"
DEFAULT_CONTAINER = "bag-tts"
DEFAULT_DOC = "/sgl-workspace/sglang-omni/docs/cookbook/higgs_tts.md"
# One expression for both grep -E and re so the two sides can never drift apart.
TOKEN_RE = r"<\|[a-z_]+:[a-z_]+\|>"


def expected_tokens(tokens_path: Path) -> set[str]:
    """Expand tokens.json (category -> names) into literal <|category:name|> strings."""
    data = json.loads(tokens_path.read_text(encoding="utf-8"))
    return {f"<|{cat}:{name}|>" for cat, names in data["categories"].items() for name in names}


def cookbook_tokens(container: str, doc: str) -> set[str]:
    """Grep the cookbook inside the TTS container.

    grep exits 1 with empty stderr when it simply found nothing; docker reports its
    own failures (no such container, not running) on stderr, so stderr decides
    whether exit 1 is "no matches" or an error.
    """
    cmd = ["docker", "exec", container, "grep", "-oE", TOKEN_RE, doc]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    no_matches = proc.returncode == 1 and not proc.stderr.strip()
    if proc.returncode != 0 and not no_matches:
        raise RuntimeError(f"{' '.join(cmd)} exited {proc.returncode}: {proc.stderr.strip()}")
    return set(re.findall(TOKEN_RE, proc.stdout))


def diff(expected: set[str], found: set[str]) -> tuple[list[str], list[str]]:
    """Return (in cookbook but not in tokens.json, in tokens.json but not in cookbook)."""
    return sorted(found - expected), sorted(expected - found)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--tokens", type=Path, default=DEFAULT_TOKENS)
    parser.add_argument("--container", default=DEFAULT_CONTAINER)
    parser.add_argument("--doc", default=DEFAULT_DOC)
    args = parser.parse_args(argv)

    expected = expected_tokens(args.tokens)
    try:
        found = cookbook_tokens(args.container, args.doc)
    except (OSError, RuntimeError) as exc:
        print(f"cannot read cookbook: {exc}", file=sys.stderr)
        return 2

    missing, unknown = diff(expected, found)
    for tok in missing:
        print(f"MISSING in tokens.json (cookbook has it): {tok}")
    for tok in unknown:
        print(f"UNKNOWN in tokens.json (cookbook lacks it): {tok}")
    if missing or unknown:
        return 1
    print(f"tokens.json matches the cookbook: {len(found)} tokens")
    return 0


if __name__ == "__main__":
    sys.exit(main())
