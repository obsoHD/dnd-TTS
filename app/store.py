"""SQLite persistence: the bank, its renders, and the M2 tables (REBUILD.md §6).

The schema is created here and nowhere else. Render files are written next to
the database under ``renders/<voice>/`` and the rows only point at them, so an
Energy re-master can swap the mastered file without touching the accepted raw
take. WAL mode lets the render worker write while the API reads.

This module deliberately imports nothing from the audio stack: ``put_render``
takes a finished ``RenderResult`` and stores what it carries.
"""
from __future__ import annotations

import hashlib
import sqlite3
import wave
from contextlib import closing
from io import BytesIO
from pathlib import Path
from typing import TYPE_CHECKING

from app import config

if TYPE_CHECKING:
    from app.render import RenderResult

_NOW = "strftime('%Y-%m-%dT%H:%M:%fZ','now')"

SCHEMA = f"""
CREATE TABLE IF NOT EXISTS voices(
    id TEXT PRIMARY KEY, version INTEGER NOT NULL, ref_sha TEXT NOT NULL,
    golden_seed INTEGER NOT NULL, sim_baseline REAL, sim_strict REAL, sim_loose REAL,
    gain_db REAL, energy INTEGER NOT NULL, locked_at TEXT);
CREATE TABLE IF NOT EXISTS lines(
    id TEXT PRIMARY KEY, voice_id TEXT NOT NULL, lang TEXT NOT NULL, category TEXT NOT NULL,
    text TEXT NOT NULL,
    source TEXT NOT NULL CHECK(source IN ('bank','improv','dictate','script','suggest')),
    active_render_id TEXT, favourite INTEGER NOT NULL DEFAULT 0, slot INTEGER,
    created TEXT NOT NULL DEFAULT ({_NOW}));
CREATE TABLE IF NOT EXISTS renders(
    id TEXT PRIMARY KEY, line_id TEXT, voice_id TEXT NOT NULL, voice_version INTEGER NOT NULL,
    recipe_version TEXT NOT NULL, master_version TEXT NOT NULL, take_no INTEGER NOT NULL,
    seed INTEGER NOT NULL, sim REAL NOT NULL, cer REAL, dur_s REAL NOT NULL, lufs REAL,
    verified INTEGER NOT NULL, gate TEXT NOT NULL CHECK(gate IN ('pass','failed')),
    raw_path TEXT NOT NULL, path TEXT NOT NULL,
    created TEXT NOT NULL DEFAULT ({_NOW}));
CREATE TABLE IF NOT EXISTS jobs(
    id TEXT PRIMARY KEY, kind TEXT NOT NULL, priority TEXT NOT NULL, status TEXT NOT NULL,
    line_id TEXT, error TEXT, created TEXT NOT NULL DEFAULT ({_NOW}), started TEXT, done TEXT);
CREATE TABLE IF NOT EXISTS scenes(
    id TEXT PRIMARY KEY, name TEXT NOT NULL, voice_ids TEXT NOT NULL DEFAULT '[]',
    note TEXT, ambience TEXT);
CREATE TABLE IF NOT EXISTS playlists(
    id TEXT PRIMARY KEY, name TEXT NOT NULL, line_ids TEXT NOT NULL DEFAULT '[]');
CREATE TABLE IF NOT EXISTS session(
    id TEXT PRIMARY KEY, started TEXT NOT NULL, facts TEXT NOT NULL DEFAULT '[]',
    said TEXT NOT NULL DEFAULT '[]');
CREATE TABLE IF NOT EXISTS speaker(client_id TEXT PRIMARY KEY, claimed_at TEXT NOT NULL);
CREATE INDEX IF NOT EXISTS lines_board ON lines(voice_id, lang, category);
CREATE INDEX IF NOT EXISTS renders_line ON renders(line_id);
CREATE INDEX IF NOT EXISTS jobs_queue ON jobs(status, priority, created);
"""


def db_path() -> Path:
    return config.DATA_DIR / "app.db"


def db() -> sqlite3.Connection:
    """A fresh connection per call: they are cheap, and a per-thread cache would
    only hide misuse across the worker and API threads. Callers close it
    (``with closing(db()) as con``). WAL is persistent in the file, but the
    pragma is idempotent and keeps a hand-copied database consistent."""
    config.DATA_DIR.mkdir(parents=True, exist_ok=True)
    con = sqlite3.connect(db_path(), timeout=30)
    con.row_factory = sqlite3.Row
    con.execute("PRAGMA journal_mode=WAL")
    con.execute("PRAGMA synchronous=NORMAL")
    return con


def init_db() -> None:
    with closing(db()) as con, con:
        con.executescript(SCHEMA)


def line_id(voice_id: str, lang: str, category: str, text: str) -> str:
    """Bank ids hash the identity of a line so re-importing phrases.json is
    idempotent and a pinned render survives the re-import."""
    key = "|".join([lang, voice_id, category, text])
    return hashlib.sha1(key.encode("utf-8")).hexdigest()


def upsert_line(voice_id: str, lang: str, category: str, text: str, source: str) -> str:
    """Insert once; an existing row keeps its pin, favourite and slot untouched,
    because the id already says the text is identical."""
    lid = line_id(voice_id, lang, category, text)
    with closing(db()) as con, con:
        con.execute(
            "INSERT INTO lines(id, voice_id, lang, category, text, source) VALUES(?,?,?,?,?,?) "
            "ON CONFLICT(id) DO NOTHING",
            (lid, voice_id, lang, category, text, source))
    return lid


def set_active_render(line_id: str, render_id: str) -> None:
    with closing(db()) as con, con:
        con.execute("UPDATE lines SET active_render_id=? WHERE id=?", (render_id, line_id))


def put_render(r: RenderResult, line_id: str | None, raw_path: str, path: str) -> None:
    """Record a finished render. The id is deterministic, so a repeat of the same
    recipe replaces its own row instead of piling up duplicates."""
    with closing(db()) as con, con:
        con.execute(
            "INSERT OR REPLACE INTO renders(id, line_id, voice_id, voice_version, recipe_version, "
            "master_version, take_no, seed, sim, cer, dur_s, lufs, verified, gate, raw_path, path) "
            "VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
            (r.render_id, line_id, r.voice_id, r.voice_version, r.recipe_version, r.master_version,
             r.take_no, r.seed, r.sim, r.cer, duration_s(r.pcm, r.sr), r.lufs, int(r.verified),
             r.gate, raw_path, path))


def get_render(render_id: str) -> dict | None:
    with closing(db()) as con:
        row = con.execute("SELECT * FROM renders WHERE id=?", (render_id,)).fetchone()
    if row is None:
        return None
    out = dict(row)
    out["verified"] = bool(out["verified"])
    return out


def duration_s(pcm: bytes, sr: int) -> float:
    return len(pcm) / (2 * sr)


def wav_bytes(pcm: bytes, sr: int) -> bytes:
    """int16 mono PCM -> WAV container (ported from the legacy ``_wav``)."""
    buf = BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sr)
        w.writeframes(pcm)
    return buf.getvalue()


def write_wav(path: Path, pcm: bytes, sr: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(wav_bytes(pcm, sr))


def write_render_files(r: RenderResult) -> tuple[str, str]:
    """Persist the accepted raw take and its mastered file, returning their
    paths for ``put_render``. The mastered name carries ``master.VERSION`` so a
    chain change never serves a stale file under the old name."""
    from app import master  # pedalboard is heavy; only the file name needs it here

    folder = config.renders_dir(r.voice_id)
    raw = folder / f"{r.render_id}.raw.wav"
    mastered = folder / f"{r.render_id}.{master.VERSION}.wav"
    write_wav(raw, r.raw_pcm, r.sr)
    write_wav(mastered, r.pcm, r.sr)
    return str(raw), str(mastered)
