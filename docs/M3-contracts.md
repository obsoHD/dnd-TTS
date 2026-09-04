# M3 — Delivery selector and the Writer: module contracts

Builds on M1 (voice core) and M2 (service, worker, board, player, web). Read `docs/REBUILD.md` §3, §4 and §5 first.
The rule that governs this milestone: **the reference clip is the voice, and identity outranks expressiveness.**
A delivery that moves the speaker embedding is not shipped, it is disarmed. Nothing here runs inside the render path.

## Why the delivery is a pure text function
`worker.plan(voice, text, take_no)` hashes the canonical text into the `render_id`, and `render_line` re-canonicalises
the same text. So a delivery is applied **to the text, at the API boundary**, before a job is created. The token then
travels inside `job.text`, the cache key separates deliveries automatically, a restart re-renders the identical line,
and no M1/M2 signature changes except one optional field on `POST /api/say`. Do not thread a delivery through the
worker, the store or `render_line`.

## Ownership (disjoint)
- **delivery**: `app/delivery.py`, `tests/unit/test_delivery.py`, and the `delivery` field in `app/api/say.py` + `app/api/board.py` (`POST /api/lines/{id}/regenerate` keeps its behaviour; tiles stay bare)
- **writer**: `app/llm.py`, `app/api/write.py`, `tests/unit/test_llm.py`
- **lab**: `scripts/arm_spice.py`, `docs/M3.md`
- **web**: `web/app.js`, `web/ui.css` (improv bar only: delivery selector, pencil button, undo)
- **verify** (last): suite + TestClient smoke with the LLM stubbed; no GPU, no network.

## app/delivery.py (delivery owner)
```python
@dataclass(frozen=True)
class Spice: id: str; token: str; label: str          # label is Slovak, shown on the pill
SPICES: tuple[Spice, ...]      # vzdych <|sfx:sigh|>, smiech <|sfx:laughter|>, pobavený <|emotion:amusement|>,
                               # nahnevaný <|emotion:anger|>, nadšený <|emotion:enthusiasm|>,
                               # výrazne <|prosody:expressive_high|>, krik <|style:shouting|>
BARE = "bare"
MAX_ARMED = 3
def available(voice: Voice) -> list[dict]   # [{id,label,token,armed,measured}] — bare first, then every spice,
                                            # armed=True only if voice.armed_spices lists it; banned tokens are dropped
def apply(text: str, spice_id: str | None, voice: Voice) -> str
def resolve(spice_id: str | None, voice: Voice) -> Spice | None
class NotArmed(ValueError): ...
```
`apply` rules, from REBUILD §3, in order:
1. `None`/`bare`/empty -> the text unchanged.
2. Unknown id, or an id not in `voice.armed_spices` -> `NotArmed` (the API answers 400; the UI never offers it).
3. The token goes **after the first word, attached to the second**: `Ten <|emotion:anger|>nie.` Leading punctuation
   and quotes stay with their word. A one-word line has no second word, so the text is returned unchanged with no
   token (a one-word line is too short for a delivery to survive the gate anyway).
4. The text may already carry a typed token; `apply` never adds a second one — return it unchanged.
5. Pure and idempotent: `apply(apply(t, s, v), s, v) == apply(t, s, v)`.

`voice.armed_spices` is a new `voice.yaml` block written only by `scripts/arm_spice.py`:
```yaml
armed_spices:
  vzdych: {sim_drop: 0.011, min_sim: 0.918, n: 20, armed_at: "2026-09-04T..."}
```
`voices.Voice` gains `armed_spices: dict = field(default_factory=dict)` (additive; `load_voice` must accept a
voice.yaml without it, and `lock_reference` clears it — a new clip is a new voice). At most `MAX_ARMED` entries;
`available()` reports the rest as `armed: false`.

`POST /api/say` gains `delivery: str | None = None`. The endpoint resolves it against the voice, applies it, and uses
the result as the job text. `NotArmed` and `canon.BannedToken` -> 400 with the reason. Everything else is unchanged.

## app/llm.py (writer owner) — the Writer
One resident model, never in the render path, never asked for per-sentence anything.
```python
class BrainNotReady(RuntimeError): ...
def residency() -> str                      # "resident" | "loaded" | "absent" — GET {LLM_URL}/api/ps
def fix(text: str, voice: Voice, lang: str = "sk") -> dict
    # {"text": str, "original": str, "changed": bool, "note": str}
```
`fix` is the pencil button and does **two** jobs in one pass, because the DM types fast at the table:
1. **Correct** — spelling, diacritics (`ludia` -> `ľudia`), agreement, word order. Slovak first; `lang="en"` uses the
   English system prompt.
2. **Optimise for speech** — spoken word order over written, contractions the character would use, a `—` where a
   speaker takes a beat and `…` where the beat is longer (canon maps these to the model's own pause tokens), and it
   must come in under `canon.MAX_CHARS`. The persona (`voice.persona[lang]`) is given as context so the rewrite keeps
   the character; the meaning may not change and nothing may be invented.

Hard rules the implementation enforces after the model answers, not by asking nicely:
- never emits `<|...|>` tokens (strip them; a delivery is the DM's choice, not the Writer's),
- at most 3 beat marks total,
- `0.5 <= len(new)/len(old) <= 2.0` and it must `canon.canonicalize` — otherwise return the original with
  `changed: false` and a note saying why,
- `timeout=8`, `options: {temperature: 0.3, num_ctx: 4096}`, `think: false`, `keep_alive: -1`,
  `stream: false` against `POST {LLM_URL}/api/chat` with `config.LLM_MODEL`. **At most two attempts, and the second
  only after a guard has rejected the first** (`ATTEMPTS`): re-rolling a usable answer buys a different sentence, not
  a better one, but a rejected one left the DM looking at "no change" on a trivially fixable line. A resident 27B
  answers in about half a second, so the retry is free at the table.
- The persona goes in as a **description, never an identity**. Bag's persona block opens with "Si Vak" and ends by
  telling its reader to answer in one to three sentences; appended raw to a corrector prompt it made the model answer
  the DM's line in character instead of editing it (measured 2026-09-04: "kde si nasiel ten mec a kolko stal" came
  back as Bag telling a story). `_PERSONA_LEAD` states outright that the model is not that character and must not
  follow instructions inside the description, and the user turn is framed as a task (`_TASK_LEAD`) rather than a bare
  line of dialogue, because a model handed dialogue replies to it.
- The brain is **warmed, not merely probed**: `residency` gates every call and ollama unloads an idle model, so
  without `warm_in_background()` at boot nothing would ever ask for the model and it would stay absent forever.
  `POST /api/brain/wake` lets the table call it back, and `readyz` reports `loading` while that runs.
- residency is checked first: not resident -> `BrainNotReady`; the API answers 503 `{"detail": "mozog nie je pripravený"}`.
  Never fall back to another model.

`app/api/write.py`: `POST /api/fix {voice, text, lang="sk"}` -> the `fix` dict. `GET /api/deliveries?voice=` ->
`available()` (this router owns it so the delivery module needs no FastAPI import). Register both in `app/main.py`
through the existing peer mechanism — the service owner's `_peer` list gains nothing; import the routers the same way
the other api modules are imported.

## scripts/arm_spice.py (lab owner)
`python -m scripts.arm_spice --voice bag [--spice vzdych ...] [--lines N=20] [--apply]`
For each candidate spice: render the voice's golden lines (`tests/golden/<voice>/lines.json`) with the spice applied,
one take each, and compare against the same lines rendered bare in the same run (never against a stored baseline —
the box, the model and the master must be identical for the comparison to mean anything).
Promote a spice only when **median SIM drop <= 0.02 and no take below `voice.gate["strict"]`**. Print a table
(spice, n, median bare, median spiced, drop, min, verdict) and, with `--apply`, write the winners into
`voice.yaml:armed_spices`, newest measurement wins, capped at `MAX_ARMED` by smallest drop. Without `--apply` it
changes nothing. `krik` (`<|style:shouting|>`) and every token in `voice.banned_tokens` are refused outright for Bag
per the spec.

## Web (web owner) — improv bar only
Do not touch the tiles, the roster, the last-10 strip or the keyboard map beyond what is listed here.
- **Delivery pill** left of the SK/EN pills: a button showing the current delivery (`bare` reads "normálne"), opening
  a small popover listing `available()` for the active voice. Armed entries are selectable; unarmed ones are shown
  greyed with the note "neoverené v Labe" so the DM can see the mechanism exists. Selection is per voice, kept in
  `localStorage`, and resets to bare when the voice changes. `POST /api/say` carries `delivery`.
- **Pencil button** right of the input: calls `POST /api/fix` with the current text and voice. While it waits the
  button shows a spinner and the input is read-only. On success the input is replaced and a one-line ghost under the
  bar reads "opravené — vrátiť" with an undo that restores the original exactly; `Ctrl+Z` in the input does the same.
  `changed: false` shows the note instead. A 503 shows "mozog nie je pripravený" in the banner slot, and the button
  goes disabled until the next `status` event says the brain is back.
- Keys: `Ctrl+Enter` = fix then speak. Everything else stays as it is.
- The pencil never speaks by itself, and `Enter` still speaks exactly what is in the box.

## Tests
`test_delivery.py`: placement after the first word, one-word line untouched, idempotence, unknown/unarmed -> `NotArmed`,
banned token dropped from `available`, `apply` result still canonicalises. `test_llm.py` (requests stubbed, no network):
residency parsing for resident/loaded/absent, `BrainNotReady`, token stripping, beat cap, the length-ratio guard
returning the original, a successful fix returning `changed: true`, timeout -> the original with a note.
Verify: full suite plus a TestClient smoke where `app.llm.fix` is stubbed — `GET /api/deliveries`, `POST /api/fix`
200 and 503, `POST /api/say` with an unarmed delivery -> 400 and with `bare` -> 200.
