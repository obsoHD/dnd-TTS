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

import yaml

from app import canon, config, delivery, store
from app.voices import Voice, load_voice

SLOTS = 8               # the favourites row: keys 1-8
DEFAULT_SLOTS = 7       # defaults fill 1-7; slot 8 stays free for the DM's own pick
# The NPC bank is generic, so both neutral NPC voices get their own copy of
# every line; every other bank key names the voice that speaks it.
NPC_VOICES = ("male", "female")
# The signature refusal: the category whose first line lives on the giant red
# T tile instead of a slot (§4). Every voice has its own, because "Ten nie." is
# Bag's line and no one else's -- the merchant refuses to sell, an NPC tells you
# where to go. Keyed by the bank key, so both NPC voices share the NPC one.
SIGNATURE_CATEGORY = {
    "sk": {"bag": "Ten nie.", "shopkeep": "Toto nepredám", "npc": "Choď do piče"},
    "en": {"bag": "Not that one.", "shopkeep": "Not selling that", "npc": "Piss off"},
}
BANK_KEY = {"male": "npc", "female": "npc"}       # every other voice is its own bank key

# Where a DM's own line lands when no category is given (§2). It is free text
# picked by no one, so it can never collide with a bank's signature category
# and is deliberately left out of ``_seed_favourites``'s bank-only sweep.
SAVED_CATEGORY = {"sk": "Moje", "en": "Mine"}


def signature_category(voice_id: str, lang: str) -> str | None:
    """The signature category for this voice, or None for a voice with no bank.
    The four built-in voices are hard-coded; a voice made later by the Creator
    gets its own entry written to ``voice.yaml`` by ``set_signature``, so this
    stays data-driven with no code change per new voice."""
    builtin = SIGNATURE_CATEGORY.get(lang, {}).get(BANK_KEY.get(voice_id, voice_id))
    return builtin if builtin is not None else _yaml_signature(voice_id, lang)


def _voice_yaml_path(voice_id: str) -> Path:
    return config.voice_dir(voice_id) / "voice.yaml"


def _yaml_signature(voice_id: str, lang: str) -> str | None:
    path = _voice_yaml_path(voice_id)
    if not path.exists():
        return None
    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    return (data.get("signature_category") or {}).get(lang)


def set_signature(voice_id: str, lang: str, category: str) -> None:
    """Persist ``voice_id``'s signature category for ``lang`` into its
    ``voice.yaml`` (a ``signature_category: {sk: ..., en: ...}`` block), so
    ``signature_category`` picks it up for a voice the Creator makes without
    touching this module. Written directly (not through ``app.voices.Voice``,
    which has no field for it) so every other key in the file is preserved."""
    path = _voice_yaml_path(voice_id)
    data = yaml.safe_load(path.read_text(encoding="utf-8")) if path.exists() else None
    data = dict(data or {"id": voice_id})
    signatures = dict(data.get("signature_category") or {})
    signatures[lang] = category
    data["signature_category"] = signatures
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".yaml.tmp")
    tmp.write_text(yaml.safe_dump(data, sort_keys=False, allow_unicode=True), encoding="utf-8")
    tmp.replace(path)


# One row per tile, joined to the pinned render because a tile's state is
# nothing but that render's verdict. Bank order is insertion order; ``rowid``
# breaks ties inside the same millisecond.
_LINE_SQL = """
SELECT l.id, l.voice_id, l.lang, l.text, l.category, l.favourite, l.slot, l.source, l.delivery,
       r.id AS render_id, r.gate, r.verified, r.take_no
FROM lines l LEFT JOIN renders r ON r.id = l.active_render_id
"""
_BOARD_SQL = _LINE_SQL + " WHERE l.voice_id=? AND l.lang=? ORDER BY l.created, l.rowid"


# Database files this process has already migrated, so the board's own columns
# are checked once per file instead of once per read.
_migrated: set[str] = set()


class LineNotFound(KeyError):
    """No line row with that id; the API maps this to 404."""


class BankLine(ValueError):
    """A bank line was asked to delete itself; the API maps this to 409."""


def ensure_columns() -> None:
    """Add the board's own columns to a ``lines`` table that lacks them:
    ``favourite``/``slot`` (M2) and ``delivery`` (M5). The M1 schema in
    ``store`` carries the first two, so on a fresh database only ``delivery``
    is actually added; ``NULL`` there means a bare tile, which is exactly what
    every line saved before M5 was."""
    wanted = {"favourite": "INTEGER NOT NULL DEFAULT 0", "slot": "INTEGER", "delivery": "TEXT"}
    with closing(store.db()) as con, con:
        have = {row["name"] for row in con.execute("PRAGMA table_info(lines)")}
        for name, ddl in wanted.items():
            if name not in have:
                con.execute(f"ALTER TABLE lines ADD COLUMN {name} {ddl}")
    _migrated.add(str(store.db_path()))


def _ready() -> None:
    """Migrate this database before the board reads or writes it.

    WHY not only at boot: ``delivery`` is the board's column and lives outside
    ``store.SCHEMA``, so a database made by ``init_db`` alone (the Creator's
    commit path, a test, a script) has never seen it, and every query here
    names it. Memoised per database file, so the cost is one set lookup per
    call and the ALTER is attempted once -- ``ensure_columns`` itself stays
    unmemoised, because the boot sequence calls it to *check*, not to skip.
    """
    if str(store.db_path()) not in _migrated:
        ensure_columns()


def import_bank(path: Path | None = None) -> int:
    """Upsert every bank line as ``source="bank"`` and return how many line
    rows the bank maps to (NPC lines count once per NPC voice). Idempotent:
    ids hash the line's identity, so a re-import never duplicates a row or
    touches a pin, favourite or slot. A voice/language with no favourites yet
    gets the defaults. ``path`` defaults to ``DATA_DIR/phrases.json``, read at
    call time so a test's ``BAG_DATA`` override is honoured.

    Bank lines the file no longer carries are retired afterwards (see
    :func:`_retire_missing`), so editing a line's wording replaces its tile
    instead of leaving the old one behind."""
    ensure_columns()
    bank = json.loads((path or config.DATA_DIR / "phrases.json").read_text(encoding="utf-8"))
    count = 0
    seen: dict[tuple[str, str], set[str]] = {}
    for lang, banks in bank.items():
        for key, categories in banks.items():
            for voice_id in NPC_VOICES if key == "npc" else (key,):
                for category, texts in categories.items():
                    for text in texts:
                        seen.setdefault((voice_id, lang), set()).add(
                            store.upsert_line(voice_id, lang, category, text, "bank"))
                        count += 1
                _seed_favourites(voice_id, lang, categories)
    _retire_missing(seen)
    return count


def _retire_missing(seen: dict[tuple[str, str], set[str]]) -> None:
    """Drop bank rows for a voice/language the bank file no longer lists.

    WHY: a line id hashes voice+lang+category+text, so rewording a bank line
    writes a new row and the old tile would sit on the board forever. Only
    ``source='bank'`` rows are touched -- a line the DM saved is theirs, and a
    voice/language absent from this file is left completely alone. The renders
    stay in the store under their own ids; only the tile goes.
    """
    if not seen:
        return
    with closing(store.db()) as con, con:
        for (voice_id, lang), ids in seen.items():
            keep = ",".join("?" * len(ids))
            con.execute(
                f"DELETE FROM lines WHERE voice_id=? AND lang=? AND source='bank' AND id NOT IN ({keep})",
                (voice_id, lang, *ids))


def _seed_favourites(voice_id: str, lang: str, categories: dict[str, list[str]]) -> None:
    """Default favourites for a voice/language that has none: the first line of
    each category on slots 1-7, in bank order, skipping the signature category
    because its first line has the T key. Only runs when nothing is starred or
    slotted yet, so the DM's own row survives every re-import."""
    signature = signature_category(voice_id, lang)
    firsts = [(category, texts[0]) for category, texts in categories.items()
              if texts and category != signature]
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


def voice_for(voice_id: str, cache: dict[str, Voice | None] | None = None) -> Voice | None:
    """This line's voice, or ``None`` when it cannot be read.

    WHY it may be ``None``: a board read must never 500 because a voice folder
    was renamed or its ``voice.yaml`` is half-written -- the tiles are still
    playable, they simply lose the tone they were saved with. Read at call time
    (never cached across calls) so the Lab disarming a spice shows on the very
    next board read; ``cache`` is the per-read memo that keeps one board of
    hundreds of tiles down to a single file read.
    """
    if cache is not None and voice_id in cache:
        return cache[voice_id]
    try:
        voice: Voice | None = load_voice(voice_id)
    except (OSError, KeyError, ValueError, TypeError, yaml.YAMLError):
        voice = None
    if cache is not None:
        cache[voice_id] = voice
    return voice


def _armed(spice_id: str | None, voice: Voice | None) -> bool:
    """Whether the stored tone is *still* armed for this voice. ``delivery``
    owns the rule (the Lab's cap included), so this only asks it."""
    if not spice_id or voice is None:
        return False
    try:
        return delivery.resolve(spice_id, voice) is not None
    except delivery.NotArmed:
        return False


def line_text(line: dict, voice: Voice | None) -> str:
    """The text a job for this tile must carry: the stored tone applied, or the
    bare line when that tone is no longer armed.

    WHY this is the only place the rule lives: three paths turn a tile into a
    job (a tap through ``POST /api/say``, Regenerate, the boot pre-render) and
    they must agree, or the pre-render warms a cache key the tap never asks
    for. WHY it degrades instead of raising: the Lab can disarm a spice between
    the save and the tap, and silence at the table is worse than a flat line.
    """
    spice_id = line.get("delivery")
    if not spice_id or voice is None:
        return line["text"]
    try:
        return delivery.apply(line["text"], spice_id, voice)
    except delivery.NotArmed:
        return line["text"]


def _entry(row: Row, cache: dict[str, Voice | None] | None = None) -> dict:
    spice_id = row["delivery"]
    return {"id": row["id"], "text": row["text"], "category": row["category"],
            "status": line_status(row["gate"], row["verified"]), "render_id": row["render_id"],
            "favourite": bool(row["favourite"]), "slot": row["slot"],
            # The stored id even when it is no longer armed: the UI greys it out
            # rather than silently forgetting the tone the DM picked.
            "delivery": spice_id, "delivery_armed": _armed(spice_id, voice_for(row["voice_id"], cache))}


def _row(con: Connection, line_id: str) -> Row:
    row = con.execute(_LINE_SQL + " WHERE l.id=?", (line_id,)).fetchone()
    if row is None:
        raise LineNotFound(line_id)
    return row


def _lines(voice_id: str, lang: str) -> list[dict]:
    _ready()
    cache: dict[str, Voice | None] = {}
    with closing(store.db()) as con:
        return [_entry(row, cache) for row in con.execute(_BOARD_SQL, (voice_id, lang))]


def _signature(lines: list[dict], voice_id: str, lang: str) -> str | None:
    """The first line of this voice's signature category: the giant T tile."""
    category = signature_category(voice_id, lang)
    return next((line["id"] for line in lines if line["category"] == category), None)


def langs(voice_id: str) -> list[str]:
    """Every language this voice has lines in, in first-seen order.

    WHY: the boot pre-render walks these so a board flipped to EN mid-scene is
    already warm instead of rendering a hundred lines while the table waits.
    """
    with closing(store.db()) as con:
        rows = con.execute(
            "SELECT lang FROM lines WHERE voice_id=? GROUP BY lang ORDER BY MIN(created), MIN(rowid)",
            (voice_id,)).fetchall()
    return [row["lang"] for row in rows]


def voice_id_of(line_id: str) -> str:
    """The voice a line belongs to; the API needs it to queue the line's render."""
    with closing(store.db()) as con:
        return _row(con, line_id)["voice_id"]


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
            "lines": lines, "favourites": favourites,
            # Key kept from M2: the Play page draws whatever line it names on the
            # T tile, and that line is now the voice's own refusal, not Bag's.
            "ten_nie": _signature(lines, voice_id, lang)}


def get_line(line_id: str) -> dict:
    _ready()
    with closing(store.db()) as con:
        return _entry(_row(con, line_id))


def set_line(line_id: str, favourite: bool | None = None, slot: int | None = None,
             category: str | None = None, delivery: str | None = None) -> dict:
    """Move a tile on the board; ``None`` leaves a field alone. ``slot`` 1-8
    puts the line on that key, evicting whoever held it and starring the line
    (a slot is a favourite with a key); 0 takes it off the row.
    ``favourite=False`` also frees the slot, so the row never shows an
    unstarred line. Applied in that order, so an unstar wins over a slot.
    ``delivery`` re-tones the tile: an id the Lab armed for this voice,
    ``"bare"``/``""`` to clear it back to a plain line, and an unknown or
    unarmed id raises ``NotArmed`` (400) without writing anything -- validated
    first, inside the same transaction, so a refused tone leaves the whole
    patch untouched. A tone that really changes also unpins the tile's render
    (see ``_write_delivery``), so the tile goes back to ``pending`` and the next
    tap is heard in the new tone."""
    _ready()
    with closing(store.db()) as con, con:
        row = _row(con, line_id)
        if delivery is not None:
            _write_delivery(con, line_id, _checked_delivery(delivery, row["voice_id"]))
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


def _checked_delivery(spice_id: str, voice_id: str) -> str | None:
    """The delivery id to store for ``voice_id``, or ``None`` to clear it.

    ``"bare"`` and ``""`` clear the column and are answered before the voice is
    even read, because clearing a tone must work for a voice whose ``voice.yaml``
    has gone missing -- otherwise a tile could get stuck on a tone it can no
    longer be talked out of. Everything else must be armed *now*: writing a tone
    is the DM choosing it, and a choice the Lab never measured is refused loudly
    (``NotArmed`` -> 400) instead of being rendered bare behind their back.
    """
    if not spice_id.strip() or spice_id == delivery.BARE:
        return None
    voice = voice_for(voice_id)
    if voice is None:
        raise delivery.NotArmed(f"hlas {voice_id!r} nemá voice.yaml, podanie {spice_id!r} nie je overené")
    spice = delivery.resolve(spice_id, voice)
    return None if spice is None else spice.id


def _write_delivery(con: Connection, line_id: str, spice_id: str | None) -> None:
    """Store the tile's tone and, when it actually changed, drop its pin.

    WHY the pin goes: a tile's state is nothing but its pinned render's verdict,
    so a tile that has already rendered reports ``ready`` and the Play page
    plays that take straight from the cache -- which is the *old* tone, and
    stays the old tone forever, because ``worker.adopt_render`` only adopts when
    nothing is pinned. Clearing ``active_render_id`` puts the tile back to
    ``pending``, which is exactly the state the M5 contract describes, and the
    next tap renders the toned text and adopts it. Only the pin is dropped: the
    render row stays in the store, so no known-good take is lost and re-toning
    back to the previous tone finds its take in the cache.

    Unchanged tones keep their pin, so re-saving a line with the tone it already
    has (the star on a tile the DM never re-toned) does not throw away a ready
    tile.
    """
    before = con.execute("SELECT delivery FROM lines WHERE id=?", (line_id,)).fetchone()
    con.execute("UPDATE lines SET delivery=? WHERE id=?", (spice_id, line_id))
    if before is not None and before["delivery"] != spice_id:
        con.execute("UPDATE lines SET active_render_id=NULL WHERE id=?", (line_id,))


def add_line(voice_id: str, lang: str, category: str | None, text: str, source: str = "improv",
             delivery: str | None = None) -> dict:
    """Save a line the DM typed (or dictated, scripted, kept from Suggest) into
    the board. It is checked the way the render will check it, so a line that
    can never render (empty, over the spoken-character cap) is refused here
    with the same error instead of failing later on the queue. No category
    means the DM just hit save on something improvised: it lands in
    ``SAVED_CATEGORY`` for the line's language. The id hashes voice+lang+
    category+text and deliberately **not** the delivery, so saving the same
    text again with another tone re-tones the tile the DM already has instead
    of growing a second one -- which is what "favourites should remember the
    tone I picked" asks for, and what keeps this idempotent. ``delivery=None``
    leaves an existing tile's tone alone (a new tile is bare anyway);
    ``"bare"`` clears it, which is what the save button sends when the gear
    says normálne. Validated before anything is written, so a refused tone
    never leaves a tile behind, and a tone that really changes unpins the tile's
    render (see ``_write_delivery``) so the next tap is heard in it."""
    _ready()
    clean = _clean(text, "text")
    canon.canonicalize(clean, lang=lang)
    cat = _clean(category, "category") if category else SAVED_CATEGORY.get(lang, SAVED_CATEGORY["sk"])
    spice_id = None if delivery is None else _checked_delivery(delivery, voice_id)
    line_id = store.upsert_line(voice_id, lang, cat, clean, source)
    if delivery is not None:
        with closing(store.db()) as con, con:
            _write_delivery(con, line_id, spice_id)
    return get_line(line_id)


def delete_line(line_id: str) -> None:
    """Remove a line the DM saved. A bank line is permanent furniture, never
    the DM's to delete, so it refuses with ``BankLine`` (409); its renders are
    never touched here -- only the board row goes, so an old take already
    played stays in the store under its own render id."""
    _ready()
    with closing(store.db()) as con, con:
        row = _row(con, line_id)
        if row["source"] == "bank":
            raise BankLine(line_id)
        con.execute("DELETE FROM lines WHERE id=?", (line_id,))


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
    _ready()
    with closing(store.db()) as con:
        row = _row(con, line_id)
    take_no = 0 if row["take_no"] is None else int(row["take_no"]) + 1
    return {"voice_id": row["voice_id"], "text": row["text"], "take_no": take_no}


def prerender_plan(voice_id: str, lang: str) -> list[str]:
    """Line ids in the order the worker should fill the board: the favourites
    row by key, then the starred rest, then the signature refusal, then
    everything else in bank order, so the tiles the DM reaches for first are the
    first ready."""
    lines = _lines(voice_id, lang)
    ten_nie = _signature(lines, voice_id, lang)
    on_row = sorted((line for line in lines if line["slot"]), key=lambda line: line["slot"])
    starred = [line for line in lines if line["favourite"] and not line["slot"]]
    first = [line for line in lines if line["id"] == ten_nie]
    return list(dict.fromkeys(line["id"] for line in on_row + starred + first + lines))
