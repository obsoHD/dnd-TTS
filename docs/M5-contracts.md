# M5 — A tile remembers its tone

Builds on M1–M4. Read `docs/M3-contracts.md` (why a delivery is a pure text function) and `docs/M4-contracts.md`
(saved phrases) first. One ask from the table:

> "favourites or Moje should remember the tone I picked, not just normal if possible"

Today every tile is bare by contract: `web/app.js`'s `tapLine` calls `say()` with no delivery, and `POST /api/lines`
does not store one. So a line saved while the gear said *smiech* plays flat forever. This milestone gives a line its
own delivery and keeps every existing guarantee: the delivery is still applied to the text at the API boundary, still
part of the cache key, and still never threaded through the worker, the store's render rows or `render_line`.

## Ownership (disjoint)
- **board**: `app/board.py`, `app/api/board.py`, `app/main.py` (the prerender call only), `tests/unit/test_board.py`
- **web**: `web/app.js`, `web/ui.css`
- **verify** (last): full suite plus a TestClient walk; may add `tests/unit/test_m5_smoke.py`.

## Storage (board owner)
- `lines` gains `delivery TEXT` (nullable, `NULL` = bare). Add it in `board.ensure_columns()` beside `favourite`
  and `slot`, so a database from an older image migrates on boot.
- `_entry` gains two keys: `delivery` (the stored id, or `None`) and `delivery_armed` (bool — whether that id is
  **still** armed for this voice). A tone can stop being armed: the Lab re-measures, a re-locked reference clears
  `armed_spices`. That must never break a tile.
- The line id keeps hashing voice+lang+category+text and **not** the delivery. Saving the same text again with a
  different tone re-tones the tile the DM already has; it does not make a second one. That is the behaviour the ask
  describes, and it keeps `add_line` idempotent.

```python
def add_line(voice_id, lang, category, text, source="improv", delivery: str | None = None) -> dict
def set_line(line_id, favourite=None, slot=None, category=None, delivery: str | None = None) -> dict
```
`delivery` is validated against `app.delivery` for that voice before it is written: an unknown or unarmed id is a
`NotArmed` (400 at the API). `"bare"` and `""` clear the column back to `NULL`. `import_bank` never sets one —
bank lines ship bare, and the DM re-tones the ones they care about.

## Rendering a toned line
One helper, used by every path that turns a line into a job, so the rule lives in one place:
```python
def line_text(line: dict, voice: Voice) -> str   # board.py: delivery.apply when still armed, else the bare text
```
- `POST /api/say` is unchanged: the caller sends `delivery` and it is applied as it already is.
- `POST /api/lines/{id}/regenerate` uses `line_text`, so a regenerate keeps the tile's tone.
- `app/main.py::_enqueue_prerender` uses `line_text`, so the pre-render warms the cache for the tone the tile will
  actually play. Nothing else in `main.py` changes.
- A stored tone that is no longer armed renders **bare** rather than failing, and `delivery_armed: false` tells the UI
  to show it greyed. Silence at the table is worse than a flat line.

## API (board owner)
```
POST  /api/lines   {voice, lang, text, category?, delivery?}      -> the line row (with delivery, delivery_armed)
PATCH /api/lines/{id} {favourite?, slot?, category?, delivery?}   -> the line row; "bare" clears the tone
```
Both answer 400 with the reason for an unarmed or unknown tone.

## Web (web owner)
- **Saving keeps the tone.** The star in the improv bar sends the gear's current selection. The star on a last-10
  entry sends the tone that render actually carried: `say()` records `delivery` in `meta[render_id]`, and the strip's
  save reads it, falling back to the bar's current selection when the entry predates it.
- **A tile plays its own tone.** `tapLine` sends `line.delivery`; a line with none stays bare exactly as now. The
  cached-render path is unaffected because the render id already encodes the toned text.
- **A tile shows its tone.** A tile whose `delivery` is set carries a small label in the corner (the Slovak label from
  `/api/deliveries`, e.g. *smiech*), greyed with a line through it when `delivery_armed` is false. A bare tile shows
  nothing, so the board does not become noisy.
- **Long-press carries the tone across.** `editLine` sets the gear to the line's delivery as well as filling the box,
  so a tile lifted into the bar and re-spoken sounds like the tile did.
- **Re-toning a tile.** In the tile's own controls (the same place Delete lives), a "tón" item opens the delivery
  popover for that line and `PATCH`es it; the tile goes back to `pending` on its own because its render id changed,
  and the next tap renders it. No new keyboard binding.
- Nothing else moves: the keyboard map, the roster, the tiles' existing states and the improv bar's layout stay as
  they are.

## Tests
`test_board.py`: the migration adds the column to an old table; `add_line` stores a tone and re-saving the same text
with a different tone re-tones the one tile; an unarmed tone is refused; `"bare"` clears it; `line_text` applies an
armed tone, falls back to bare for an unarmed one, and leaves a bare line untouched; `prerender_plan` order is
unchanged. Verify: a TestClient walk that saves a toned line, reads it back from `/api/board` with
`delivery`/`delivery_armed`, renders it through `/api/say`, and confirms the job text carries the token after the
first word — plus `node --check web/app.js`.
