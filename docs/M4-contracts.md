# M4 — The table's own hands: live tile state, saving phrases, long-press to edit, and the Voice Creator

Builds on M1 (voice core), M2 (service, worker, board, player, web) and M3 (delivery, Writer). Read `docs/REBUILD.md`
§2.5, §4, §5 and §6 first, then `docs/M2-contracts.md` and `docs/M3-contracts.md` for the interfaces you are extending.
Four asks from the table, in the DM's words:

1. "once i render something it should light up like rendered, not stay grey"
2. "allow me to save phrases too"
3. "if i hold press it, have it move to type-a-line instead so i can edit it; regular click plays as it does now"
4. "re-add the voice creator, updated with the new system so it adds the different phrases for the soundboard"

## Ownership (disjoint)
- **board**: `app/board.py`, `app/api/board.py`, `tests/unit/test_board.py` — the saved-phrase category and `POST /api/lines` defaults
- **creator**: `app/creator.py`, `app/api/creator.py`, `app/llm.py` (additive only: `persona`, `phrases`), `app/stt.py`, `tests/unit/test_creator.py`
- **web**: `web/app.js`, `web/ui.css` — items 1, 2 and 3 only
- **creator-web**: `web/creator.html`, `web/creator.js`, and the creator styles at the END of `web/ui.css` inside a clearly marked block (the web owner never touches that block, and you never touch theirs)
- **verify** (last): suite + TestClient smoke with the LLM, the STT and `render_line` stubbed.

## 1. A rendered tile must stay lit (web owner)
Root cause, already diagnosed: `web/app.js`'s `say()` posts `/api/say` **without `line_id`**, so the server never
adopts the render for that line (`worker.adopt_render(job.line_id, ...)` gets `None`). The tile lights up from local
state and goes grey again on the next `/api/board` read.
- `say({text, lineId, ...})` must send `line_id` whenever it has one (tiles, favourites, the T tile, regenerate).
- `POST /api/say` already accepts `line_id`; nothing server-side changes.
- After `job.done` the tile keeps its ring from the board patch, and a re-fetch of `/api/board` must agree — assert
  that in the smoke test by reading the board after the job finishes.
- A tile whose render is `unverified` or `gate-failed` keeps its amber/red ring: "lit" means the render exists, not
  that it passed.

## 2. Saving a phrase (board + web owners)
Backend (board owner):
- `board.SAVED_CATEGORY = {"sk": "Moje", "en": "Mine"}` — where a DM's own line lands. It is never a signature
  category and never seeds favourites.
- `POST /api/lines` keeps its shape but `category` becomes optional and defaults to `SAVED_CATEGORY[lang]`;
  `source` is `"improv"`. Re-saving the same text in the same category returns the existing line (the id already
  hashes voice+lang+category+text), so saving twice is not a duplicate tile.
- `DELETE /api/lines/{id}` removes a line the DM saved (`source != "bank"`), 409 for a bank line. Its renders stay in
  the store; only the tile goes.
Web (web owner):
- A save (star) button in the improv bar, enabled when the box is not empty: saves to the active voice and language,
  then flashes the new tile in its category. `Ctrl+S` does the same.
- Each entry in the last-10 strip gains the same save button, so a good improvised line becomes a tile after it lands.
- A saved tile shows a small dot marking it as the DM's own, and its context action offers Delete (with a confirm
  step); bank tiles never offer it.

## 3. Long-press a tile to edit it (web owner)
- Pointer down on a tile starts a 450 ms timer; releasing before it fires plays the line exactly as now. When it
  fires: the tile's text is copied into the improv box, the box takes focus with the caret at the end, the tile gives
  a short haptic-style pulse (CSS only), and **nothing is spoken**.
- Cancel the timer on pointer move beyond 10 px, on pointer cancel, and on the second pointer of a two-finger touch.
- Use pointer events so it behaves the same with a mouse, a pen and a finger; `contextmenu` on a tile is prevented so
  a long touch does not open the browser menu.
- The keyboard equivalent: `Shift` + the tile's number key (`Shift+1..8`) edits instead of plays. `Shift+T` edits the
  signature line. This must not collide with the existing `Shift+1..9` voice switch: voice switching moves to
  `Alt+1..9`, and the roster labels update to match.

## 4. The Voice Creator (creator + creator-web owners)
A page at `/creator` that takes a clip and produces a table-ready voice: locked reference, calibrated gate and level,
a persona, a full soundboard, and a pre-rendered bank. Everything long runs as a job with progress on `/ws`.

### app/stt.py (creator owner)
`gate.py` already posts a WAV to whisper for CER. Lift that into one place both can use, without changing `gate`'s
public behaviour: `transcribe(pcm: bytes, sr: int, lang: str, timeout: float = 60) -> str | None`. The gate keeps its
2 s table budget by passing its own timeout; the creator can afford 60 s.

### app/creator.py (creator owner)
```python
@dataclass
class Draft:                     # one in-progress voice, persisted under DATA_DIR/creator/<draft_id>/
    id: str; voice_id: str; label: str; lang: str; source_name: str
    clip_path: str; start_s: float; end_s: float
    transcript: str; description: str; f0_band: list[int]
    persona: dict; categories: list[str]; phrases: dict[str, list[str]]
    status: str                  # new|clipped|transcribed|described|locked|calibrated|banked|done|failed
def create(upload: bytes, filename: str, label: str, lang: str) -> Draft
def clip(draft_id: str, start_s: float, end_s: float) -> Draft     # ffmpeg cut, then measure f0 band
def transcribe(draft_id: str) -> Draft                             # whisper fills Draft.transcript
def describe(draft_id: str, description: str) -> Draft             # the Writer fills persona + categories
def write_phrases(draft_id: str, per_category: int = 10) -> Draft   # the Writer fills phrases
def commit(draft_id: str) -> dict                                  # lock -> calibrate -> bank -> pre-render
def get(draft_id: str) -> Draft; def drafts() -> list[Draft]; def discard(draft_id: str) -> None
```
Rules that keep a new voice as trustworthy as Bag:
- The clip is converted the way `voices.lock_reference` expects and **capped at 30 s**; the DM picks the window
  because a crowd reaction or music in the middle poisons the clone (this happened with the shopkeep clip).
- `f0_band` is measured from the clip (median F0 via `librosa.pyin`, band = `[0.65 x median, 1.5 x median]`, clamped
  to 50-420 Hz), not guessed, and is what the golden test later enforces.
- The transcript is whisper's, shown to the DM to correct before the lock. A wrong transcript is the single most
  common cause of a bad clone.
- `commit` runs `voices.lock_reference`, then `voices.calibrate` on the **generated phrases** (30 longest), then
  imports the phrases into the bank as `source="creator"`, seeds favourites and the signature category, and queues
  the pre-render at `batch` priority. It emits `creator.progress {draft_id, step, pct}` through the same `on_event`
  the worker uses, and returns `{voice_id, gate, gain_db, lines, queued}`.
- A voice id must be new, `[a-z0-9_-]{2,24}`, and never one of `bag`, `male`, `female`, `shopkeep`.

### app/llm.py additions (creator owner, additive only)
```python
def persona(label: str, description: str, lang: str) -> dict     # {"sk": ..., "en": ...} + label/desc echo
def phrases(persona: dict, categories: list[str], lang: str, per_category: int = 10) -> dict[str, list[str]]
```
Both use the resident model with the same guards as `fix` (think false, keep_alive -1, num_ctx 4096, one attempt,
JSON-only reply parsed defensively). Every generated line must pass `canon.canonicalize` and `MAX_CHARS`, contain no
`<|...|>` token, and be dropped if it does not; a category that ends up with fewer than three usable lines is
regenerated once, then reported as thin. **The last category is always the signature refusal** for that voice, and
`board.SIGNATURE_CATEGORY` gains the new voice's entry through `board.set_signature(voice_id, lang, category)`
persisted in `voice.yaml` (`signature_category: {sk: ..., en: ...}`) so board logic stays data-driven.

### app/api/creator.py (creator owner)
```
POST   /api/creator/drafts        multipart: file, label, lang           -> Draft
POST   /api/creator/{id}/clip     {start_s, end_s}                       -> Draft
POST   /api/creator/{id}/transcribe                                      -> Draft
PATCH  /api/creator/{id}          {transcript?, description?, label?, voice_id?, categories?} -> Draft
POST   /api/creator/{id}/phrases  {per_category?}                        -> Draft
POST   /api/creator/{id}/commit                                          -> {job_id}
GET    /api/creator/drafts | GET /api/creator/{id} | DELETE /api/creator/{id}
GET    /api/creator/{id}/clip.wav                                        -> the trimmed clip, for the audition player
```
Uploads are capped at 25 MB and to audio content types; 503 with the Slovak brain message when the Writer is not
resident; 400 with the reason for a bad window, a duplicate voice id or a failed conversion.

### Creator page (creator-web owner)
`/creator`, reachable from a `+` card at the end of the roster on the Play page (the web owner adds only that card and
its link). One column, five steps, each with its own Next, no wizard magic:
1. **Clip** — drop a file, waveform-free but with a play button, two number inputs for the window in seconds, a
   Trim button; shows the measured pitch band.
2. **Transcript** — whisper's text in an editable box with the warning that it must match the clip word for word.
3. **Character** — label, voice id, language, and a free-text description; Generate fills persona and the category
   list, both editable.
4. **Phrases** — the generated soundboard, per category, every line editable and removable, an Add line box, and a
   count per category with a warning under three.
5. **Commit** — a summary, then one button that locks, calibrates and banks, with a progress bar fed by
   `creator.progress` and a final card showing the gate numbers and a link back to Play with the new voice active.
State lives on the server; a reload resumes the draft. Reuse the Play page's palette and controls; no new libraries.

## Tests
`test_board.py`: default category on `POST /api/lines`, saving the same line twice is one tile, delete refuses a bank
line. `test_creator.py` (no network, no GPU: stub `app.llm`, `app.stt.transcribe`, `voices.calibrate` and the worker):
the draft state machine and its persistence, the 30 s cap and the f0 band from a synthetic clip, id validation and the
reserved names, phrase filtering (a token-carrying line and an over-long line are dropped, a thin category is
reported), and `commit` calling lock -> calibrate -> import -> queue in that order. Verify: full suite, then a
TestClient smoke that walks a draft from upload to commit with everything stubbed, plus `node --check` on both web
files and a diff review proving the web owner's and creator-web owner's changes did not overlap.
