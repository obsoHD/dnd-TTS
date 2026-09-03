# Mr. Bag — Final Rebuild Spec

Repo `C:\Users\admin\dnd-TTS`, rebuilt in place (`server/` deleted at the end of M2). Stack unchanged: Higgs Audio v3 via SGLang-Omni on GPU 1 (`:8010`), ollama `huihui_ai/qwen3.8-abliterated:27b` resident on GPU 0, whisper large-v3 at lifeos `:8443/stt`, FastAPI app in Docker on `:8020`, no-build web UI. Slovak first, EN toggle.

## 1. Principles

1. **The reference clip is the voice.** `bag_ref.wav` rendered bare is the sound the user likes. Nothing between clip and speaker varies per line.
2. **One delivery: bare.** Expression lives in the words. No modes, no director, no machine-chosen tags.
3. **Cached beats generated.** The table is pinned-WAV playback; improv has a hard 4 s budget with an instant filler.
4. **Identity is the only measured quality.** Speaker similarity selects takes; whisper CER only detects truncation; pitch and pace are never optimised.
5. **One tap, eyes back on the players.** Two Lab knobs per voice (Energy, Gate); zero knobs on the Play screen.
6. **Degrade, never surprise.** No silent model swap, no seed swap on ASR failure, no self-widening gate.

## 2. Voice pipeline

### 2.1 Reference
- `data/voices/<id>/ref.wav`: mono 24 kHz 16-bit, head/tail silence trimmed at -42 dBFS, peak -3 dBFS, **no EQ/compression/denoise**, 12-20 s (cap 30). Bag v1 = current `bag_ref.wav` byte-for-byte with its verbatim transcript; the Lab has an A/B slot for a second window of the same narrator later (adopting it = version bump + bank re-render).
- Immutable: `sha256` recorded in `voice.yaml`; a mismatch refuses to render until re-lock (`voice_version += 1`). Path never changes, so SGLang's ref-code LRU and RadixAttention prefix cache stay warm (`SGLANG_OMNI_HIGGS_REF_CODE_CACHE=1` stays).
- One `references[]` entry per request. No multi-ref, no previous-chunk continuity reference.
- `ref.emb.npy`: ECAPA embedding (mean over 3 s windows), computed once at lock.

### 2.2 Sampling (frozen per voice)
`temperature 0.7, top_k 50, top_p 1.0, max_new_tokens = min(1024, 120 + 12*syllables)`, `seed = golden_seed + take_no`, input = canonical text only, no leading tokens. Golden seed is picked by ear in the Lab (10 lines x seeds 0-9, ties by mean SIM). `--enable-deterministic-inference` is not used; determinism comes from the cache.

### 2.3 Canonical text (`app/canon.py`, versioned)
NFC; collapse whitespace; normalise quotes; ensure terminal punctuation; strip any `<|...|>` typed by the user; `Brá**cho` -> `Bráácho` (max 3 stars); ` — ` -> ` <|prosody:pause|> `, `…`/`...` -> ` <|prosody:long_pause|> ` (max 3 pause tokens per line, extras dropped). **Hard cap 250 chars per line**; the improv bar refuses longer text (400 `too_long`, live counter); Prep splits long paragraphs at sentence ends into separate line rows. No joined WAVs anywhere.

### 2.4 Tokens
`configs/tokens.json` seeded from the verified list in `docs/milestone-0.md`: emotion `amusement, anger, confusion, contemplation, enthusiasm, relief, sadness, surprise`; prosody `expressive_high, pause, long_pause, pitch_high`; sfx `laughter, sigh, cough, crying, humming, screaming, sneeze, sniff`; style `shouting`. `scripts/verify_tokens.py` re-checks it against the mounted `tokenizer.json` at build. Anything else (`pitch_low`, `speed_*`, `whispering`, `singing`, `expressive_low`, `bitterness`...) does not exist and is refused with 400.

### 2.5 Gate (per take, in order)
1. **Sanity:** peak >= -30 dBFS; duration in `[0.12*syl, 0.45*syl + 2.0]` s; no internal silence > 1.5 s.
2. **Speaker similarity (selector):** SpeechBrain ECAPA `spkrec-ecapa-voxceleb`, CPU, 16 kHz; cosine vs `ref.emb.npy`. Among sanity passers pick **max SIM**.
3. **CER on the SIM winner only:** whisper large-v3, `_norm_for_cer` (fold diacritics, collapse repeats). Reject if CER > 0.15 or word-count ratio outside 0.75-1.25, then try the next take by SIM. Whisper down or > 2 s: serve with `verified=false` (amber); the seed never changes because of STT.

**Calibration** (`bag-cli calibrate`, Lab button): 30 bank lines x 3 takes = 90 SIM values; `baseline = median`, `p10`; `strict = max(baseline - 0.04, p10)`, `loose = baseline - 0.08`. Bag = strict. Paper thresholds (ECAPA 0.25 etc.) are never used.

**Rejection sampling:** N=2 concurrent takes (N=3 when > 120 chars; SGLang batches them, 1.07 s measured); if none pass, one retry round with seeds +3..+5; if still none, serve best-SIM flagged `gate=failed` (red dot, Regenerate offered). The table is never blocked.

### 2.6 Cache and idempotency
```
render_id  = sha256(voice_id | voice_version | recipe_version | canonical_text | take_no)
recipe_version = sha256(sampler + gate + canon_version + tokens.json)
master_version = sha256(master block of voice.yaml + master.py VERSION)
```
- Each render stores the accepted **raw take** (`<id>.raw.wav`) and the mastered file (`<id>.<master_version>.wav`). An Energy change re-masters from raw in the background; the GPU is never touched.
- A recipe bump re-renders the bank at `batch` priority and swaps `lines.active_render_id` **only when the new take passes the gate**; the old file stays until then. No holes on the board.
- Cache hit (`/api/say` on a known voice+text) returns in < 50 ms without the GPU. Regenerate = `take_no + 1`; Pin sets `active_render_id` and survives restarts.
- Pre-render at boot and after any lock: every bank line of every locked voice (active language first, favourites first), 6 fillers (`Hmm…`, `No…`, `Počkaj…`, `Tak…`, `Ehm…`, `Moment…`), signature lines.

### 2.7 Master chain: the "voice changer" (`app/master.py`, deterministic, identical for every take of a voice)

| stage | Bag value (Energy 65) |
|---|---|
| edge trim | onset -50 dBFS keep 40 ms lead; keep 250 ms tail; **no internal silence editing** |
| HPF | 50 Hz, 12 dB/oct (80 Hz thinned the 74-113 Hz fundamental) |
| presence bell | +2.6 dB @ 3 kHz, Q 1.0 |
| air shelf | +1.6 dB @ 10 kHz |
| compressor | 3:1, attack 5 ms, release 100 ms, threshold set at calibration for ~4.5 dB median GR |
| tempo | 1.078x, `pedalboard.time_stretch(preserve_formants=True, high_quality=True)`; cap 1.10 |
| pitch / formant | 0 st / 0 % for Bag; NPCs max +/-1 st, +/-3 %, never stacked with tempo > 1.05 |
| fixed gain | one per-voice constant from calibration so the 30-line set median = -18 LUFS (pyloudnorm); **no per-line normalisation** |
| room | shared by all voices: `Reverb(room_size 0.22, damping 0.65, wet 0.08, dry 0.92)` |
| limiter | -1.0 dBTP, then deterministic peak scale-down |
| fades | 8 ms in, 60 ms out after the tail |

Energy `E` (0-100) is the only knob: `presence = 0.04*E dB`, `air = 0.025*E dB`, `GR target = 0.07*E dB`, `tempo = 1 + 0.0012*E`. Gate knob: strict/loose. De-esser, expander, exciter, VC/RVC: not in v1 (M5 decides on VC from the golden report).

## 3. Delivery presets

| preset | tokens | default |
|---|---|---|
| `bare` | none | every voice, every line |
| beats (from punctuation only) | ` — ` -> `<|prosody:pause|>`, `…` -> `<|prosody:long_pause|>` | on, max 3/line |
| spice `vzdych` | `<|sfx:sigh|>` | off |
| spice `smiech` | `<|sfx:laughter|>` | off |
| spice `pobavený` | `<|emotion:amusement|>` | off |
| spice `nahnevaný` | `<|emotion:anger|>` | off |
| spice `nadšený` | `<|emotion:enthusiasm|>` | off |
| spice `výrazne` | `<|prosody:expressive_high|>` | off |
| spice `krik` | `<|style:shouting|>` | off, forbidden on Bag |

Rules: a spice is emitted **after the first word, attached to the second** (`Ten <|emotion:anger|>nie.`), never leading. Max 3 armed per voice. Arming happens only in the Lab: 20 golden lines with the spice, promote only if median SIM drop <= 0.02 and no take below `strict`. **Bag ships with zero armed spices**; `pitch_high` and `shouting` are banned on Bag. The LLM never selects presets; the DM applies an armed spice by long-pressing Speak. Tempo dial, fine-speed slider, song mode, 15 modes and the director are gone.

**The Writer** (`app/llm.py`, resident 27B, `keep_alive -1`, `think false`, `num_ctx 4096`): `suggest` (3 candidate lines, few-shot from the bank, may write `—`/`…`), `ask` (answer a typed player line in persona), `fix` (`SK_FIX_SYSTEM`, minimal edit, kept only if length ratio 0.5-2.0). Never in the render path, never returns per-sentence anything. Not resident -> `BrainNotReady`, UI shows "mozog nie je pripravený", nothing falls back to `qwen3:14b`.

## 4. Table features

| feature | table moment | behaviour |
|---|---|---|
| Soundboard | player reaches into the Bag | category tabs (Bag: Pozdrav kámoša, Urážka partie, Chvastanie po záchrane, Odmietnutie predmetu, Bojový pokrik, Sarkastická poznámka, Namrzené povzbudenie, Ten nie.) with **tiles showing the line text**; all 80 SK Bag lines cached at boot |
| Favourites + Ten nie. | top 8 | 8 slots keys `1-8`; giant red-bordered `Ten nie.` tile always visible, key `T` |
| Improv bar | DM types a line | Enter renders and plays; on cache miss a cached filler plays <= 300 ms, ambience ducks -10 dB, the line queues behind; thin 4 s progress bar on the DM screen only; typed lines are `bare` |
| Suggest / Ask | DM is blank / a player spoke | 3 candidates shown as tiles, **all prefetch-rendered** so tapping is instant; save-to-bank icon; never auto-plays |
| Dictation | Slovak diacritics on a tablet | hold backtick or mic: whisper fills the box, DM still taps Speak. No end-to-end talk-to-the-Bag (over budget) |
| Roster | NPC enters | cards (initials/portrait, name, 2-line note, signature line); tap = plays signature + becomes active speaker; long-press activates silently; `Shift+1..9` |
| Scenes | party enters the dungeon | name + roster subset + note (feeds Suggest) + optional ambience file; one tap, 1 s crossfade |
| Session memory | session 2 | `!fact` typed in the bar is stored; last 20 played lines + facts injected into Suggest/Ask |
| Queue / Stop / Repeat | two taps at once | server-side playback FIFO, never overlaps; `Esc` stops < 100 ms; `R` repeats last |
| Last-10 strip | take came out wrong | Replay, Regenerate (`take_no+1`), Pin; amber/red dots |
| Script strip | the night before | `[bag] ... / [kupec] ...` lines parsed, rendered at `prep` priority, played with `Space` = next |
| LAN remote / pedal | hands full of dice | `GET /remote/{slot1..8,tennie,next,repeat,stop}?k=KEY`; pedal L repeat, M next, R stop |
| Speaker page | DM on the couch | `/speaker` owns audio (one click unlocks autoplay), ducks ambience; tablets are remotes; state lives on the server |

## 5. UX

Four pages, plain ES modules + vendored Preact/htm (offline), one WebSocket each: **Play** `/`, **Prep** `/prep`, **Lab** `/lab`, **Speaker** `/speaker`.

**Play (>= 900 px):** top bar = scene pill, status dots (TTS / whisper / brain / speaker), queue count, red **STOP** (56 px). Left 260 px = active speaker card + roster. Centre = favourites row (96 px tiles with key numerals, `Ten nie.` at the end), category tabs, tile grid (>= 88x88 px, 16 px text, 2-line clamp). Sticky bottom = improv bar (mic, text field with 250 counter, SK/EN, Speak, Suggest, Ask, Fix) + last-10 strip. Suggest candidates appear above the bar.
**Tablet (< 900 px):** single column, roster as a horizontal strip, 2 tiles per row, sticky bar with the mic as the biggest control, no hover affordances.
**States:** `ready`, `queued` (dimmed + position), `rendering` (ring + seconds), `playing` (accent border + progress), `unverified` (amber), `gate-failed` (red), `error` (tap to retry). One banner at a time: "TTS down: cached lines only", "mozog nie je pripravený: Suggest/Ask disabled", "whisper down: lines unverified", "no speaker connected (playing here)". Boot shows tiles filling in, favourites first.
**Keys:** `1-8` favourites, `T` Ten nie., `Space` next (script/suggestions; ignored while typing), `R` repeat, `Esc` stop, `Enter` speak, `Ctrl+Enter` suggest, backtick hold = mic, `Shift+1..9` voice, `Ctrl+P` pin last.
**Prep:** script textarea with live parse, Render all with progress, roster/scene editors, bank editor (add line, favourite, slot). **Lab:** per voice waveform + transcript, Calibrate (SIM histogram, baseline, thresholds, gain, golden seed audition), Energy, Gate, spice arming panel, Run golden, last-100 renders with SIM/CER badges, New voice wizard (upload -> convert -> editable whisper transcript -> seed audition -> calibrate -> lock).

## 6. Architecture

**Services:** `tts` (SGLang-Omni, GPU 1, `:8010`, image pinned by digest, healthcheck on `/health`); `app` (host network `:8020`: API, WS hub, one render worker, SQLite, static web; CPU torch + speechbrain only, no torchaudio/parselmouth); external ollama `:11434` (pinned via `scripts/register_broker.sh`) and whisper `:8443`.

**Pipeline:** `validate -> canonicalize -> render_id -> cache -> enqueue -> synth (N takes) -> gate -> master -> store -> notify`. One worker over a persisted priority queue (`live` > `prep` > `batch`); a job fans out its takes but two jobs never run at once; a `live` job aborts the in-flight `batch` HTTP request and re-queues it.

**Endpoints:**
```
GET  /api/voices                     GET  /api/voices/{id}
POST /api/voices  (multipart: audio, name, lang)      POST /api/voices/{id}/calibrate
POST /api/voices/{id}/lock           PUT  /api/voices/{id}/master {energy, gate, spices[]}
DELETE /api/voices/{id}              POST /api/voices/{id}/spice-test {tag} -> {median_drop, min_sim}
GET  /api/board?voice=&lang=         -> categories[], lines[{id,text,status,render_id}], favourites[]
POST /api/lines {voice,lang,category,text}   PATCH /api/lines/{id} {favourite,slot,category}
POST /api/say {voice, text, lang, priority, spice?}  -> {job_id, render_id, cached}
GET  /api/jobs/{id}   POST /api/jobs/{id}/cancel   GET /api/queue
GET  /api/renders/{id}.wav           POST /api/renders/{id}/pin   POST /api/lines/{id}/regenerate
POST /api/play {render_id}   POST /api/stop   POST /api/repeat   POST /api/next   POST /api/speaker/claim
POST /api/suggest {voice,category,scene,lang} -> {candidates[{text,render_id,job_id}]}
POST /api/ask {voice,text,scene,lang}         POST /api/fix {text,lang}
POST /api/transcribe (multipart audio, lang)  -> {text}
GET/PUT /api/scenes   POST /api/scenes/{id}/activate   POST /api/session/facts {text}
POST /api/script/parse {text}   POST /api/script/render {lines[]} -> {playlist_id}
GET  /remote/{slot1..slot8|tennie|next|repeat|stop}?k=
WS   /ws  events: status, job.{queued,started,progress,done,failed}, queue.changed,
          play.{start,end}, bank.progress, speaker.presence
GET  /healthz   GET /readyz {tts,tts_warm,stt,llm:resident|absent,speaker,bank_ready,queue_depth}   GET /metrics
```

**`data/voices/<id>/voice.yaml`:**
```yaml
id: bag            label: Mr. Bag       lang: sk      version: 1
reference: {file: ref.wav, sha256: "...", transcript: "Popravia? Dostane tretí obed. ..."}
sampler: {temperature: 0.7, top_k: 50, top_p: 1.0, max_new_tokens: 1024}
golden_seed: 4
gate: {mode: strict, baseline: 0.0, p10: 0.0, strict: 0.0, loose: 0.0, cer_max: 0.15, calibrated_at: null}
master: {energy: 65, hpf_hz: 50, presence: {hz: 3000, q: 1.0}, air_hz: 10000,
         comp: {ratio: 3, attack_ms: 5, release_ms: 100, threshold_db: null},
         tempo_cap: 1.10, pitch_st: 0, formant_pct: 0, gain_db: null, target_lufs: -18,
         room: {size: 0.22, damping: 0.65, wet: 0.08}, limiter_dbtp: -1.0}
spices: [{name: nahnevaný, tag: "emotion:anger", armed: false, median_drop: null}]
persona: {sk: "...", en: "..."}    signature_line_id: null    fillers: [...]
f0_band: [74, 113]                 # golden-test sanity only
```

**SQLite `data/app.db`:**
```
voices(id PK, version, ref_sha, golden_seed, sim_baseline, sim_strict, sim_loose, gain_db, energy, locked_at)
lines(id PK, voice_id, lang, category, text, source[bank|improv|dictate|script|suggest], active_render_id, favourite, slot, created)
renders(id PK, line_id, voice_id, voice_version, recipe_version, master_version, take_no, seed, sim, cer, dur_s, lufs, verified, gate[pass|failed], raw_path, path, created)
jobs(id PK, kind, priority, status, line_id, error, created, started, done)
scenes(id PK, name, voice_ids JSON, note, ambience)      playlists(id PK, name, line_ids JSON)
session(id PK, started, facts JSON, said JSON)          speaker(client_id PK, claimed_at)
```
Bank ids = `sha1(lang|char|category|text)` so re-imports are idempotent.

**Repo layout:**
```
app/   main.py config.py canon.py tts_client.py gate.py master.py render.py worker.py store.py
       voices.py llm.py stt.py board.py scenes.py ws.py api/{voices,board,say,play,brain,scenes,script,remote,health}.py
web/   index.html prep.html lab.html speaker.html app.js speaker.js ui.css vendor/{preact.min.js,htm.js}
configs/ tokens.json  personas/{bag,npc,shopkeep}.{sk,en}.txt  voices/{bag,male,female,shopkeep}.yaml
data/  phrases.json (bank, committed)  voices/  renders/  ambience/  app.db
tests/ unit/{test_canon,test_render_id,test_gate,test_master,test_script}.py  golden/bag/{lines.json,approved/*.wav}  run_golden.py
docker/ Dockerfile (tts) app.Dockerfile compose.yml
scripts/ bag_cli.py verify_tokens.py register_broker.sh lifeos_pause.sh lifeos_resume.sh serve.sh
tools/ m0_test.py clone.py
```

**Golden tests** (`make golden`, live stack, every voice/config change; 30 SK lines for Bag, 10 per other voice, approved WAVs committed): SIM >= `strict`; CER <= 0.15 and word ratio 0.75-1.25; F0 median (librosa pyin) inside `f0_band`; integrated loudness within +/-2 LU of -18 LUFS; duration/syllable 0.12-0.35 s; `master()` byte-identical on the same raw bytes; `render_id` and served bytes identical across a container restart. Unit tests (no GPU): canon, render_id stability, gate decisions on fixtures, spice placement, script parser. **Metrics:** `prometheus-fastapi-instrumentator` + `bag_stage_seconds{stage=synth|sim|asr|master}`, `bag_queue_depth{priority}`, `bag_cache_hits_total`, `bag_cache_misses_total`, `bag_sim{voice}`, `bag_gate_rejects_total{reason}`, `bag_llm_resident`, `bag_play_latency_seconds`.

## 7. Deleted from the current code

- `server/elongation.py` entirely (MMS alignment, PSOLA, `_splice`, `_xfade_join`, `_room_tone`, `elongate`, `parse_marks`, `word_bounds`).
- `orchestrator.py`: `_f0`, `_f0_median`, `band`/`MAX_TRIES` gate, `voice_gen`, `_voice_gen_unlocked`, `_score`/`BEST_OF`, `MODES`, `DEFAULT_MODE`, `MODE_SPS`, `MODE_BAND_HI`, `MODE_PAUSE`, `EN_RATE_SCALE`, `_lead_for`, `_place_tags`, `_beats`, `_cue`, `_pace_sentences`, `PACE_MULT`, `TEMPO_*`, `_direct`, `DIRECTOR_MODES_SK`, `_plan_parts`, `_join_parts`, continuity `extra_refs`/`_cont_*.wav`, `_trim` internal `silenceremove`, per-line `pyln.normalize`, HPF 80, `Compressor(1.8)`, `SPACES` hall/bag/dry, `_peak_normalize`/`audioop`, `_CACHE`, `_pick_model`/`LLM_FALLBACKS`, `/modes`, `/line`, `/linetext`, `/say`, `/respond`, `/converse` end-to-end, `/script` thread pool + joined WAV, `/status` (undefined `_llm_on_gpu`), `X-Bag-*` headers, request fields `song/direct/paces/modes/pauses/tempo/speed/space/emotion/seed`, env `BAG_REF`, `BAG_BROKER_URL`, `BAG_TARGET_WPS`, `BAG_MAX_TRIES`, `BAG_BEST_OF`, `BAG_ASR_CER_MAX`.
- `server/web/index.html` entirely; `docker/docker-compose.yml`, `docker/orchestrator.Dockerfile`, `scripts/gruff_test.py`, `scripts/serve_orchestrator.sh`; deps `praat-parselmouth`, `torchaudio`.
- **Kept** (moved): `phrases.json`, persona prompts + `SK_RULES` + `SK_FIX_SYSTEM` + `_improv_line` few-shot, `_llm_reply` minus `_pick_model`, `_stt`, `_asr_cer`, `_norm_for_cer`, `_syllables`, `_wav`, `_SENT`, `_stretch_np`, `_resolve_voice`, `_SCRIPT_LINE`, `SPACES["room"]`, limiter + fades, `/voices/create` convert + transcript, `compose.yml` GPU split, broker/lifeos scripts, `m0_test.py`, `clone.py`, the dark palette.

## 8. Build plan

**M1 — Voice lock, CLI only.** `canon, tts_client, gate, master, voices, store`, `configs/tokens.json`, `bag_cli.py render|calibrate|golden`, Bag golden set. *Accept:* calibration writes `voice.yaml` thresholds and gain; user picks a golden seed and judges the 10 audition lines "the same person"; golden 30/30 pass SIM strict, CER, F0 band, loudness, duration; p95 cache-miss render (2 takes + gate + master) <= 3.0 s on the box; `master` byte-identical; whisper stopped mid-run yields `verified=false` with unchanged seeds; blind A/B of 10 pairs vs the pre-director build: user prefers or ties on >= 8; image has no torchaudio/parselmouth.

**M2 — Render service and the table.** FastAPI, SQLite, worker with priorities/preemption/persistence, WS, bank pre-render, Play + Speaker pages, favourites, Ten nie., queue/Stop/Repeat, last-10 Pin/Regenerate, hotkeys, `/remote`, tablet layout; old `server/` deleted. *Accept:* cold boot pre-renders all Bag SK lines + fillers unattended; second `say` of the same text `cached:true` < 50 ms; `live` job starts within 500 ms during pre-render; tap-to-first-sample <= 150 ms from a tablet; Stop < 100 ms; two rapid taps never overlap; pinned line and queue survive `docker restart`; locked tablet never stops audio; every `/remote/*` works from curl.

**M3 — Improv, brain, mic.** Improv bar with filler-then-line, `/api/suggest` (prefetch-rendered candidates), `/api/ask`, `/api/fix`, `/api/transcribe`, session facts, degraded banners. *Accept:* typed cache-miss line: filler <= 300 ms, real line audible <= 4 s p95; Suggest = one write + one fix call, 3 Slovak candidates playable instantly; dictation fills the box < 2 s for a 5 s clip; with the 27B unloaded the board works, Suggest/Ask disabled with banner, no request reaches `qwen3:14b`; `/readyz` correct.

**M4 — Prep.** Script parse/render into the strip with `Space`, roster and scene editors, ambience bed with ducking, bank editor. *Accept:* 40-line script renders unattended with progress and plays Next -> Next; scene switch crossfades within 1 s and swaps the board; a `!fact` appears in the next suggestion; ambience ducks -10 dB on `play.start` and restores after `play.end`.

**M5 — Lab, hardening, VC decision.** Lab page (calibration UI, Energy re-master from raw, Gate, spice arming, golden report, New-voice wizard), `/metrics`, compose healthcheck + digest pin, README/docs. *Accept:* new NPC from a 20 s clip to locked voice < 5 min; Energy change re-masters the bank without GPU; a spice with median SIM drop > 0.02 cannot be armed; `make golden` green for every shipped voice; `docker compose up -d` from clean reaches `readyz` unattended. If the golden report shows p10 SIM below strict for any voice, open a seed-vc post-stage experiment on GPU 1, adopted only if it beats master-only by >= 0.03 median SIM with no CER regression.
