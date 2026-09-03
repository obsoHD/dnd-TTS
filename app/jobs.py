"""Render jobs: what the table asks the worker for, and their rows in ``jobs``.

A job is the worker's unit of scheduling: one line, one voice, one take number,
at one priority. It lives in memory while the process runs and in the ``jobs``
table so a ``docker restart`` in the middle of a bank pre-render resumes where
it stopped instead of leaving tiles ``pending`` (REBUILD.md §8, M2 acceptance).
The M1 schema only stored what a status view needs; the columns a job needs to
be *re-run* (voice, text, take, render id) are added in place by
``ensure_columns`` so an existing database keeps its history.
"""
from __future__ import annotations

import sqlite3
import time
import uuid
from contextlib import closing
from dataclasses import asdict, dataclass, field

from app import store

PRIORITY = {"live": 0, "prep": 1, "batch": 2}
TERMINAL = frozenset({"done", "failed", "cancelled"})
PENDING = ("queued", "running")

# Column -> declaration for ``ALTER TABLE ADD COLUMN``; SQLite insists on a
# default behind NOT NULL once rows exist. ``priority`` is already in the M1
# schema and stays listed so the contract's migration set is checked in full.
_EXTRA_COLUMNS = {
    "voice_id": "TEXT NOT NULL DEFAULT ''",
    "text": "TEXT NOT NULL DEFAULT ''",
    "take_no": "INTEGER NOT NULL DEFAULT 0",
    "render_id": "TEXT",
    "priority": "TEXT NOT NULL DEFAULT 'batch'",
}
_COLUMNS = ("id", "kind", "priority", "voice_id", "text", "line_id", "take_no", "status",
            "render_id", "error", "created", "started", "done")


def now() -> float:
    """Millisecond timestamps: plenty to order a queue, and they survive the
    round trip through the M1 table's TEXT-affinity columns unchanged."""
    return round(time.time(), 3)


@dataclass
class Job:
    id: str
    kind: str
    priority: str
    voice_id: str
    text: str
    line_id: str | None = None
    take_no: int = 0
    status: str = "queued"          # queued | running | done | failed | cancelled
    render_id: str | None = None
    error: str | None = None
    created: float = field(default_factory=now)
    started: float | None = None
    done: float | None = None

    def as_dict(self) -> dict:
        return asdict(self)


def new_job(kind: str, priority: str, voice_id: str, text: str, line_id: str | None = None,
            take_no: int = 0) -> Job:
    """A fresh queued job. The priority is validated here, at the edge, so the
    heap never meets a key it cannot rank."""
    if priority not in PRIORITY:
        raise ValueError(f"unknown priority {priority!r}; expected one of {sorted(PRIORITY)}")
    return Job(id=uuid.uuid4().hex, kind=kind, priority=priority, voice_id=voice_id, text=text,
               line_id=line_id, take_no=take_no)


def ensure_columns() -> None:
    """Bring an M1 ``jobs`` table up to what a job needs to be re-run.
    Idempotent; ``ADD COLUMN`` keeps the rows already there."""
    with closing(store.db()) as con, con:
        have = {row["name"] for row in con.execute("PRAGMA table_info(jobs)")}
        for name, decl in _EXTRA_COLUMNS.items():
            if name not in have:
                con.execute(f"ALTER TABLE jobs ADD COLUMN {name} {decl}")


def save(job: Job) -> None:
    """Whole-row upsert on every transition: a job changes state a handful of
    times in its life, so one statement beats a per-column update API."""
    columns = ", ".join(_COLUMNS)
    marks = ", ".join("?" * len(_COLUMNS))
    with closing(store.db()) as con, con:
        con.execute(f"INSERT OR REPLACE INTO jobs({columns}) VALUES({marks})",
                    tuple(getattr(job, name) for name in _COLUMNS))


def load(job_id: str) -> Job | None:
    with closing(store.db()) as con:
        row = con.execute(f"SELECT {', '.join(_COLUMNS)} FROM jobs WHERE id=?", (job_id,)).fetchone()
    return _from_row(row) if row is not None else None


def pending() -> list[Job]:
    """Jobs a previous process never finished, oldest first. ``created`` is
    cast because the M1 column is TEXT; rowid breaks the ties a coarse clock leaves."""
    marks = ", ".join("?" * len(PENDING))
    with closing(store.db()) as con:
        rows = con.execute(
            f"SELECT {', '.join(_COLUMNS)} FROM jobs WHERE status IN ({marks}) "
            "ORDER BY CAST(created AS REAL), rowid", PENDING).fetchall()
    return [_from_row(row) for row in rows]


def _stamp(value: str | float | None) -> float | None:
    """Timestamps are floats in memory and TEXT in the M1 table."""
    return None if value is None else float(value)


def _from_row(row: sqlite3.Row) -> Job:
    return Job(id=row["id"], kind=row["kind"], priority=row["priority"], voice_id=row["voice_id"],
               text=row["text"], line_id=row["line_id"], take_no=int(row["take_no"]),
               status=row["status"], render_id=row["render_id"], error=row["error"],
               created=float(row["created"]), started=_stamp(row["started"]), done=_stamp(row["done"]))
