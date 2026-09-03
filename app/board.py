"""The table's phrase board: the bank on disk becomes rows, rows become tiles.

Why a module of its own: the board is the one read model the Play page needs
per voice and language (categories, tiles with a state, the favourites row,
the Ten nie. key), and every write to it (favourite, slot, category, a new
improv line) must keep the invariants the tiles assume: one line per slot,
bank order preserved, a tile's state derived from its pinned render and never
stored on its own. Nothing here renders or plays; ``prerender_plan`` only says
in which order the worker should fill the board at boot.

Contract: docs/M2-contracts.md, section ``app/board.py``.
"""
from __future__ import annotations

import json
from contextlib import closing
from pathlib import Path
from sqlite3 import Connection, Row

from app import canon, config, store

SLOTS = 8               # the favourites row: keys 1-8
DEFAULT_SLOTS = 7       # defaults fill 1-7; slot 8 stays free for the DM's own pick
# The NPC bank is generic, so both neutral NPC voices get their own copy of
# every line; every other bank key names the voice that speaks it.
NPC_VOICES = ("male", "female")
# The category whose first line lives on the T key instead of a slot (§4).
TEN_NIE_CATEGORY = {"sk": "Ten nie.", "en": "Not that one."}

# One row per tile, joined to the pinned render because a tile's state is
# nothing but that render's verdict. Bank order is insertion order; ``rowid``
# breaks ties inside the same millisecond.
_LINE_SQL = """
SELECT l.id, l.voice_id, l.lang, l.text, l.category, l.favourite, l.slot,
       r.id AS render_id, r.gate, r.verified, r.take_no
FROM lines l LEFT JOIN renders r ON r.id = l.active_render_id
"""
_BOARD_SQL = _LINE_SQL + " WHERE l.voice_id=? AND l.lang=? ORDER BY l.created, l.rowid"


class LineNotFound(KeyError):
    """No line row with that id; the API maps this to 404."""


def ensure_columns() -> None:
    """Add ``favourite``/``slot`` to a ``lines`` table created before M2. The M1
    schema already has them, so on a fresh database this is a no-op; it exists
    for a database carried over from an older image."""
    wanted = {"favourite": "INTEGER NOT NULL DEFAULT 0", "slot": "INTEGER"}
    with closing(store.db()) as con, con:
        have = {row["name"] for row in con.execute("PRAGMA table_info(lines)")}
        for name, ddl in wanted.items():
            if name not in have:
                con.execute(f"ALTER TABLE lines ADD COLUMN {name} {ddl}")


def import_bank(path: Path | None = None) -> int:
    """Upsert every bank line as ``source="bank"`` and return how many line
    rows the bank maps to (NPC lines count once per NPC voice). Idempotent:
    ids hash the line's identity, so a re-import never duplicates a row or
    touches a pin, favourite or slot. A voice/language with no favourites yet
    gets the defaults. ``path`` defaults to ``DATA_DIR/phrases.json``, read at
    call time so a test's ``BAG_DATA`` override is honoured."""
    ensure_columns()
    bank = json.loads((path or config.DATA_DIR / "phrases.json").read_text(encoding="utf-8"))
    count = 0
    for lang, banks in bank.items():
        for key, categories in banks.items():
            for voice_id in NPC_VOICES if key == "npc" else (key,):
                for category, texts in categories.items():
                    for text in texts:
                        store.upsert_line(voice_id, lang, category, text, "bank")
                        count += 1
                _seed_favourites(voice_id, lang, categories)
    return count


def _seed_favourites(voice_id: str, lang: str, categories: dict[str, list[str]]) -> None:
    """Default favourites for a voice/language that has none: the first line of
    each category on slots 1-7, in bank order, skipping the Ten nie. category
    because its first line has the T key. Only runs when nothing is starred or
    slotted yet, so the DM's own row survives every re-import."""
    firsts = [(category, texts[0]) for category, texts in categories.items()
              if texts and category != TEN_NIE_CATEGORY.get(lang)]
    with closing(store.db()) as con, con:
        taken = con.execute(
            "SELECT 1 FROM lines WHERE voice_id=? AND lang=? AND (favourite OR slot IS NOT NULL) LIMIT 1",
            (voice_id, lang)).fetchone()
        if taken is not None:
            return
        for slot, (category, text) in enumerate(firsts[:DEFAULT_SLOTS], start=1):
            con.execute("UPDATE lines SET favourite=1, slot=? WHERE id=?",
                        (slot, store.line_id(voice_id, lang, category, text)))


def line_status(gate: str | None, verified: bool | int | None) -> str:
    """Tile state from the pinned render alone (§5): no render is ``pending``,
    a failed gate is ``gate-failed``, a passing take is ``ready`` once whisper
    has confirmed the words and ``unverified`` (amber, still playable) until then."""
    if gate is None:
        return "pending"
    if gate == "failed":
        return "gate-failed"
    return "ready" if verified else "unverified"


def _entry(row: Row) -> dict:
    return {"id": row["id"], "text": row["text"], "category": row["category"],
            "status": line_status(row["gate"], row["verified"]), "render_id": row["render_id"],
            "favourite": bool(row["favourite"]), "slot": row["slot"]}


def _row(con: Connection, line_id: str) -> Row:
    row = con.execute(_LINE_SQL + " WHERE l.id=?", (line_id,)).fetchone()
    if row is None:
        raise LineNotFound(line_id)
    return row


def _lines(voice_id: str, lang: str) -> list[dict]:
    with closing(store.db()) as con:
        return [_entry(row) for row in con.execute(_BOARD_SQL, (voice_id, lang))]


def _ten_nie(lines: list[dict], lang: str) -> str | None:
    """The first line of the Ten nie. category: the giant tile on the T key."""
    category = TEN_NIE_CATEGORY.get(lang)
    return next((line["id"] for line in lines if line["category"] == category), None)


def categories(voice_id: str, lang: str) -> list[str]:
    """Categories in bank order; one the DM added lands after the bank's."""
    return list(dict.fromkeys(line["category"] for line in _lines(voice_id, lang)))


def board(voice_id: str, lang: str) -> dict:
    """Everything the Play page draws for one voice and language, in one read."""
    lines = _lines(voice_id, lang)
    favourites: list[str | None] = [None] * SLOTS
    for line in lines:
        if line["slot"] is not None and 1 <= line["slot"] <= SLOTS:
            favourites[line["slot"] - 1] = line["id"]
    return {"categories": list(dict.fromkeys(line["category"] for line in lines)),
            "lines": lines, "favourites": favourites, "ten_nie": _ten_nie(lines, lang)}


def get_line(line_id: str) -> dict:
    with closing(store.db()) as con:
        return _entry(_row(con, line_id))


def set_line(line_id: str, favourite: bool | None = None, slot: int | None = None,
             category: str | None = None) -> dict:
    """Move a tile on the board; ``None`` leaves a field alone. ``slot`` 1-8
    puts the line on that key, evicting whoever held it and starring the line
    (a slot is a favourite with a key); 0 takes it off the row.
    ``favourite=False`` also frees the slot, so the row never shows an
    unstarred line. Applied in that order, so an unstar wins over a slot."""
    with closing(store.db()) as con, con:
        row = _row(con, line_id)
        if category is not None:
            con.execute("UPDATE lines SET category=? WHERE id=?", (_clean(category, "category"), line_id))
        if slot is not None:
            _place(con, row["voice_id"], row["lang"], line_id, slot)
        if favourite is not None:
            con.execute("UPDATE lines SET favourite=1 WHERE id=?" if favourite
                        else "UPDATE lines SET favourite=0, slot=NULL WHERE id=?", (line_id,))
        return _entry(_row(con, line_id))


def _place(con: Connection, voice_id: str, lang: str, line_id: str, slot: int) -> None:
    if not 0 <= slot <= SLOTS:
        raise ValueError(f"slot must be 0..{SLOTS}")
    if slot == 0:
        con.execute("UPDATE lines SET slot=NULL WHERE id=?", (line_id,))
        return
    con.execute("UPDATE lines SET slot=NULL WHERE voice_id=? AND lang=? AND slot=?", (voice_id, lang, slot))
    con.execute("UPDATE lines SET slot=?, favourite=1 WHERE id=?", (slot, line_id))


def add_line(voice_id: str, lang: str, category: str, text: str, source: str = "improv") -> dict:
    """Save a line the DM typed (or dictated, scripted, kept from Suggest) into
    the board. It is checked the way the render will check it, so a line that
    can never render (empty, over the spoken-character cap) is refused here
    with the same error instead of failing later on the queue."""
    clean = _clean(text, "text")
    canon.canonicalize(clean, lang=lang)
    return get_line(store.upsert_line(voice_id, lang, _clean(category, "category"), clean, source))


def _clean(value: str, what: str) -> str:
    """One spelling of free text, so ``line_id`` cannot fork on whitespace."""
    out = " ".join(value.split())
    if not out:
        raise ValueError(f"empty {what}")
    return out


def next_take(line_id: str) -> dict:
    """What a Regenerate job needs: the line's voice and text and the take
    after the pinned one, so the new render climbs a fresh seed ladder under a
    fresh render_id. A line that has never rendered starts at take 0."""
    with closing(store.db()) as con:
        row = _row(con, line_id)
    take_no = 0 if row["take_no"] is None else int(row["take_no"]) + 1
    return {"voice_id": row["voice_id"], "text": row["text"], "take_no": take_no}


def prerender_plan(voice_id: str, lang: str) -> list[str]:
    """Line ids in the order the worker should fill the board: the favourites
    row by key, then the starred rest, then Ten nie., then everything else in
    bank order, so the tiles the DM reaches for first are the first ready."""
    lines = _lines(voice_id, lang)
    ten_nie = _ten_nie(lines, lang)
    on_row = sorted((line for line in lines if line["slot"]), key=lambda line: line["slot"])
    starred = [line for line in lines if line["favourite"] and not line["slot"]]
    first = [line for line in lines if line["id"] == ten_nie]
    return list(dict.fromkeys(line["id"] for line in on_row + starred + first + lines))
