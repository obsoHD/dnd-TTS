"""Service locations and data roots, read once from the environment.

Everything that talks to a service or the disk asks this module instead of
``os.environ`` so a test can point the whole app at a temp dir by setting
``BAG_DATA`` before the first import (contract: docs/M1-contracts.md).
"""
from __future__ import annotations

import os
from pathlib import Path


def _data_dir() -> Path:
    """``BAG_DATA`` wins; otherwise the compose mount inside a container and a
    repo-relative ``./data`` on a developer box. Resolved once so a later
    ``chdir`` cannot move the database out from under an open app."""
    override = os.environ.get("BAG_DATA")
    if override:
        return Path(override).resolve()
    if Path("/.dockerenv").exists():
        return Path("/data")
    return Path("data").resolve()


DATA_DIR: Path = _data_dir()
# Host side of the ``/refs`` mount the TTS container reads references from;
# ``lock_reference`` copies a locked ref there so the TTS sees the same bytes.
REFS_DIR: Path = Path(os.environ.get("BAG_REFS", "/refs"))
TTS_URL: str = os.environ.get("BAG_TTS_URL", "http://127.0.0.1:8010")
STT_URL: str = os.environ.get("BAG_STT_URL", "https://127.0.0.1:8443/stt")
LLM_URL: str = os.environ.get("BAG_LLM_URL", "http://127.0.0.1:11434")
LLM_MODEL: str = os.environ.get("BAG_LLM_MODEL", "huihui_ai/qwen3.8-abliterated:27b")


def voice_dir(voice_id: str) -> Path:
    """Where a voice's ``voice.yaml`` and locked ``ref.wav`` live."""
    return DATA_DIR / "voices" / voice_id


def renders_dir(voice_id: str) -> Path:
    """Where a voice's raw and mastered render files live."""
    return DATA_DIR / "renders" / voice_id
