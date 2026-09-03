# M2 — Render service and the table: module contracts

Builds on M1 (`app/canon, tts_client, gate, master, voices, render, store, config`; do not change their public
interfaces — additive helpers only, in your own files). Read `docs/REBUILD.md` §4–§6 and §8 M2 first. All new code is
Python 3.12 (FastAPI) and no-build web (vendored Preact + htm, plain ES modules). Everything server-side is
single-process: one render worker thread, one WebSocket hub, SQLite (WAL).

## Ownership (disjoint)
- **service**: `app/main.py`, `app/ws.py`, `app/api/__init__.py`, `app/api/health.py`, `app/api/renders.py`, `app/api/voices.py`
- **worker**: `app/worker.py`, `app/jobs.py`, `app/api/say.py`, `app/api/jobs.py`, `tests/unit/test_worker.py`
- **board**: `app/board.py`, `app/api/board.py`, `data/phrases.json` (copy of `server/phrases.json`), `tests/unit/test_board.py`
- **player**: `app/player.py`, `app/api/play.py`, `app/api/remote.py`, `tests/unit/test_player.py`
- **web**: `web/index.html`, `web/speaker.html`, `web/app.js`, `web/speaker.js`, `web/ui.css`, `web/vendor/preact.min.js`, `web/vendor/htm.js` (download pinned builds: preact 10.x `preact.min.js` UMD/ESM and `htm` standalone ESM from jsdelivr; if download is impossible, write a minimal note in `web/vendor/README.md` and use `https://cdn.jsdelivr.net` script tags as fallback)
- **verify** (last): runs the suite, API smoke with FastAPI `TestClient` and mocked `render_line`.

## app/jobs.py (worker owner)
```python
PRIORITY = {"live": 0, "prep": 1, "batch": 2}
@dataclass
class Job: id: str; kind: str; priority: str; voice_id: str; text: str; line_id: str | None; take_no: int
           status: str  # queued|running|done|failed|cancelled
           render_id: str | None; error: str | None; created: float; started: float | None; done: float | None
def new_job(kind, priority, voice_id, text, line_id=None, take_no=0) -> Job   # id = uuid4 hex
```
Persisted in the `jobs` table (M1 schema; add a small migration in `app/jobs.py::ensure_columns()` if columns are
missing: voice_id, text, take_no, render_id, priority). On startup, `queued`/`running` jobs are re-queued.

## app/worker.py (worker owner)
```python
class RenderWorker:
    def __init__(self, on_event: Callable[[str, dict], None]): ...
    def start(self) -> None                       # daemon thread
    def submit(self, job: Job) -> Job             # persists, enqueues (heap by (PRIORITY, created)), emits job.queued
    def cancel(self, job_id: str) -> bool
    def queue_snapshot(self) -> list[dict]        # [{id, kind, priority, voice_id, text[:60], status, position}]
    def wait(self, job_id: str, timeout: float) -> Job | None
```
Behaviour: one job at a time. Before each job and between takes, if a `live` job is queued while a `batch`/`prep` job
runs, the running job is cancelled cooperatively (a `threading.Event` passed to `render_line`'s synth via
`app.render.render_line(voice, text, take_no, n_takes, cancel=event)` — **add an optional `cancel` kwarg to
`render_line` and `tts_client.synth/synth_many`** that aborts the streaming read; this is the one M1 change allowed,
worker owner makes it, keeping defaults backward compatible) and re-queued at the same priority. Cache: if
`store.get_render(render_id(voice, canon.text, take_no))` exists with `gate == "pass"`, the job completes immediately
(`cached: true`) without touching the TTS. On completion: `store.put_render`, `store.set_active_render(line_id, ...)` when
the line has no pinned render, emit `job.done {job_id, render_id, cached, sim, cer, verified, gate, dur_s}`. Errors →
`job.failed {job_id, error}`. Emits `job.started`, `job.progress {stage}`, `queue.changed`.

## app/board.py (board owner)
```python
def import_bank(path: Path = DATA_DIR / "phrases.json") -> int      # upsert_line for every sk/en bag/shopkeep/npc line, source "bank"; idempotent; returns count
def categories(voice_id: str, lang: str) -> list[str]
def board(voice_id: str, lang: str) -> dict   # {categories:[...], lines:[{id,text,category,status,render_id,favourite,slot}], favourites:[line ids by slot 1..8], ten_nie: line_id|None}
def set_line(line_id, favourite: bool | None = None, slot: int | None = None, category: str | None = None) -> dict
def add_line(voice_id, lang, category, text, source="improv") -> dict
def prerender_plan(voice_id: str, lang: str) -> list[str]   # line ids: favourites first, then "Ten nie.", then the rest
```
`status` per line: `ready` (active render exists, gate pass), `unverified`, `gate-failed`, `pending` (no render). Bank
voice mapping: phrases.json keys `bag/shopkeep/npc` → voices `bag`, `shopkeep`, and `npc` lines belong to both `male`
and `female`. Favourites default: the first line of each Bag category for slots 1–7 and "Ten nie." pinned to the
`T` key (not a slot). `lines` table gets columns `favourite INTEGER DEFAULT 0, slot INTEGER` if missing (migration
in `board.py::ensure_columns()`).

## app/player.py (player owner)
Server-side playback state; the browser "speaker" executes it.
```python
class Player:
    def __init__(self, on_event): ...
    def enqueue(self, render_id: str, label: str) -> dict      # FIFO; if idle → play.start immediately
    def stop(self) -> None                                     # clears queue, emits play.stop
    def repeat(self) -> None                                   # re-enqueue last played at the front
    def next(self) -> None                                     # for playlists/suggestions: pop the next pending item
    def ended(self, render_id: str) -> None                    # speaker reports end → play.end, start next
    def state(self) -> dict                                    # {now: {...}|None, queue: [...], last: {...}|None, speaker: bool}
    def claim_speaker(self, client_id: str) -> None; def release_speaker(self, client_id: str) -> None
```
Events: `play.start {render_id, url, label}`, `play.end {render_id}`, `play.stop`, `speaker.presence {connected: bool}`.
If no speaker is claimed, `play.start` still fires (the controlling page plays it locally). Guard: never two
`play.start` without an `end/stop` between them.

## app/ws.py (service owner)
```python
class Hub:
    async def connect(self, ws: WebSocket, client_id: str) -> None; def disconnect(...)
    def broadcast(self, event: str, data: dict) -> None      # thread-safe (call from worker thread via loop.call_soon_threadsafe)
```
`/ws` accepts `?client=<id>&role=play|speaker`; on connect sends `status` (readyz snapshot + player state + queue).
Speaker clients send `{"type":"ended","render_id":...}` and `{"type":"claim"}`.

## API (each file registers an `APIRouter`; `app/main.py` includes them)
```
GET  /api/voices                       -> [{id,label,lang,version,locked,calibrated,energy}]
GET  /api/voices/{id}                  -> voice.yaml as JSON (no persona text)
GET  /api/board?voice=&lang=           -> board()
POST /api/lines {voice,lang,category,text}      PATCH /api/lines/{id} {favourite?,slot?,category?}
POST /api/say {voice,text,lang,priority="live",take_no=0}   -> {job_id, render_id, cached, position}
GET  /api/jobs/{id}    POST /api/jobs/{id}/cancel    GET /api/queue
GET  /api/renders/{id}.wav             -> mastered WAV (Cache-Control: immutable);  GET /api/renders/{id} -> row
POST /api/renders/{id}/pin {line_id}   POST /api/lines/{id}/regenerate -> {job_id}  (take_no + 1)
POST /api/play {render_id,label?}  POST /api/stop  POST /api/repeat  POST /api/next  GET /api/player
POST /api/speaker/claim {client_id}
GET  /remote/{action}?k=KEY            -> action ∈ slot1..slot8 | tennie | next | repeat | stop; KEY from env BAG_REMOTE_KEY (default "bag")
GET  /healthz -> {ok}   GET /readyz -> {tts, tts_warm, stt, llm, speaker, bank_ready, queue_depth}   GET /metrics (Prometheus text; use prometheus_client)
GET  /  -> web/index.html   GET /speaker -> web/speaker.html   /static/* -> web/
```
`/api/say` never blocks on the render: it returns the job; the UI listens on `/ws`. `bank_ready` = every prerender_plan
line for locked voices has a render. Startup order in `app/main.py`: `init_db` → `import_bank` → load voices →
`Hub` → `Player` → `RenderWorker.start()` → enqueue `prerender_plan` for each locked voice at `batch` priority.

## Web (web owner)
`index.html` = **Play** page per REBUILD.md §5 (>=900 px layout; single column under 900 px). Components (Preact+htm,
no build): TopBar (status dots from `/readyz` + `status` event, queue count, red STOP 56 px), Roster (from
`/api/voices`; tap = active voice; `Shift+1..9`), Favourites row (8 tiles keys `1-8`, giant red-bordered **Ten nie.**
key `T`), CategoryTabs + TileGrid (tile = line text, 2-line clamp, state ring: ready/queued/rendering/playing/unverified/
gate-failed), ImprovBar (text with 250 counter, SK/EN pills, **Speak** = `/api/say` live then `/api/play` on
`job.done`; `Enter` speaks), LastTen strip (Replay / Regenerate / Pin), one Banner slot. Tiles: tap → if `ready`
`/api/play` immediately, else `/api/say` and auto-play on `job.done`. Keys: `1-8`, `T`, `R` repeat, `Esc` stop, `Enter`
speak, `Ctrl+P` pin last, `Shift+1..9` voice; ignored while typing except Enter/Esc. Audio: if this client has claimed
the speaker (or no speaker exists) it plays `play.start` urls with one `<audio>` element unlocked by the first user
gesture; otherwise it only shows state. `speaker.html` = a page with one big "Claim speaker" button; after claim it
plays every `play.start`, posts `ended`, shows now-playing, keeps a wake lock if available. Dark palette from the legacy
UI. No inline event handlers; ES modules; `fetch` + one WebSocket with auto-reconnect and exponential backoff.

## Tests
`test_worker.py`: priority order (live before batch), cache short-circuit, cancel of a queued job, re-queue on restart
(fake `render_line`). `test_board.py`: import idempotent count, categories, favourites defaults, status derivation.
`test_player.py`: FIFO, no double play.start, repeat/next/stop, speaker claim. Verify: `pytest tests/unit -q`, TestClient
smoke: `GET /readyz`, `GET /api/board?voice=bag&lang=sk` (after import), `POST /api/say` with `render_line` mocked →
job.done via worker, `GET /api/renders/{id}.wav` 200.
