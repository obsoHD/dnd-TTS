# M1 — Voice lock: module contracts

Builders implement exactly these interfaces (Python 3.12, package `app/`, no torchaudio,
no parselmouth). Read `docs/REBUILD.md` §1–§2.7 and §6 first. Corrections to the spec:

- **Tokens.** Control tokens are parsed by the SGLang server, not present in `tokenizer.json`.
  The authoritative list is the cookbook inside the image (`/sgl-workspace/sglang-omni/docs/cookbook/higgs_tts.md`):
  emotions `elation amusement enthusiasm determination pride contentment affection relief contemplation confusion
  surprise awe longing arousal anger fear disgust bitterness sadness shame helplessness`; styles `singing shouting whispering`;
  sfx `cough laughter crying screaming burping humming sigh sniff sneeze`; prosody `speed_very_slow speed_slow speed_fast
  speed_very_fast pause long_pause pitch_low pitch_high expressive_high expressive_low`. `configs/tokens.json` lists ALL of
  them; a per-voice `banned` list (Bag: every style, `pitch_*`, `speed_*`, `expressive_low`) is enforced by `canon`.
- **Speaker embedding.** Use `resemblyzer` (`VoiceEncoder("cpu")`, `preprocess_wav`); it is installed and measured on the
  box (0.08 s/clip; Bag takes 0.91–0.96 vs reference, other voices 0.45–0.55, shouting 0.70). Import guard:
  `webrtcvad` comes from the `webrtcvad-wheels` package.
- **Sampler default** (evidence from A/B on Slovak CER): `temperature 0.75, top_k 40, top_p 0.95`.

All audio in memory is **int16 mono PCM bytes + sample rate** (24 000 from the TTS). Helpers may convert to float32
internally. Every function is pure/deterministic unless it talks to a service.

## app/config.py
```python
DATA_DIR: Path            # env BAG_DATA (default /data in the container, ./data locally)
TTS_URL: str              # env BAG_TTS_URL default http://127.0.0.1:8010
STT_URL: str              # env BAG_STT_URL default https://127.0.0.1:8443/stt  (self-signed; verify=False)
LLM_URL: str              # env BAG_LLM_URL default http://127.0.0.1:11434
LLM_MODEL: str            # env BAG_LLM_MODEL default huihui_ai/qwen3.8-abliterated:27b
def voice_dir(voice_id: str) -> Path        # DATA_DIR/voices/<id>
def renders_dir(voice_id: str) -> Path      # DATA_DIR/renders/<id>
```

## app/canon.py
```python
CANON_VERSION = "1"
MAX_CHARS = 250
class TooLong(ValueError): ...
class BannedToken(ValueError): ...
@dataclass
class Canon:
    text: str            # what the model receives (tokens inline)
    spoken: str          # text with control tokens stripped (for CER / display)
    syllables: int
    warnings: list[str]
def canonicalize(text: str, lang: str = "sk", banned: set[str] = frozenset()) -> Canon
```
Rules: NFC; collapse whitespace; normalise quotes/dashes; ensure terminal punctuation; strip any `<|...|>` the user typed
unless it is in `configs/tokens.json` AND not banned (banned → `BannedToken`); `Brá**cho` → `Bráácho` (1 extra vowel for
1–2 stars, 2 for 3+; strip stars); ` — ` / ` - ` → ` <|prosody:pause|> `; `…` / `...` → ` <|prosody:long_pause|> `;
max 3 pause tokens per line (extras dropped, warning); `len(spoken) > MAX_CHARS` → `TooLong`.
`syllables`: Slovak-aware count (vowel nuclei; `ia ie iu ô` once; syllabic r/l between consonants; vowel-less clitics
`z v k s` count 0) — port from `server/orchestrator.py::_syllables` with the clitic fix.

## app/tts_client.py
```python
@dataclass
class Sampler: temperature: float = 0.75; top_k: int = 40; top_p: float = 0.95
def synth(text: str, ref_path: str, ref_text: str, seed: int, sampler: Sampler, max_new_tokens: int,
          timeout: float = 120) -> tuple[bytes, int]                       # POST TTS_URL/v1/audio/speech, stream pcm
def synth_many(text, ref_path, ref_text, seeds: list[int], sampler, max_new_tokens) -> list[tuple[int, bytes, int]]
                                                                            # concurrent (ThreadPool), same order as seeds
def health() -> bool                                                        # GET TTS_URL/health
```
Body fields: `model "/model"`, `stream true`, `response_format "pcm"`, `input`, `references [{audio_path, text}]`,
`temperature top_k top_p seed max_new_tokens`, `voice "default"`. `ref_path` is the path AS THE TTS CONTAINER SEES IT
(`/refs/...`); the caller passes it.

## app/gate.py
```python
class SpeakerGate:
    def __init__(self, ref_wav: str | Path): ...       # embeds the reference once (cache .emb.npy next to it)
    def similarity(self, pcm: bytes, sr: int) -> float # cosine, resemblyzer
def sanity(pcm: bytes, sr: int, syllables: int) -> tuple[bool, str]
    # peak >= -30 dBFS; duration in [0.12*syl, 0.45*syl + 2.0] s; no internal silence > 1.5 s (-45 dBFS, 20 ms frames)
def cer(pcm: bytes, sr: int, spoken: str, lang: str) -> float | None
    # whisper via STT_URL multipart (audio=wav, lang); None if STT down/slow (>2 s); normalisation: lowercase, fold
    # Slovak diacritics, drop punctuation, collapse repeated letters; difflib ratio → 1 - ratio
def word_ratio(spoken: str, heard: str) -> float
@dataclass
class TakeScore: seed: int; sim: float; sane: bool; reason: str; cer: float | None; verified: bool
def select(gate: SpeakerGate, takes: list[tuple[int, bytes, int]], canon: Canon, lang: str, strict: float,
           cer_max: float = 0.15) -> tuple[int | None, list[TakeScore]]
    # order: sanity → sim (desc) → CER on the best only (then next by sim if it fails); returns index of the chosen take
    # (or best-by-sim with gate failed if none passes) and all scores
```

## app/master.py
```python
VERSION = "1"
DEFAULT = dict(energy=65, hpf_hz=50, presence_hz=3000, presence_q=1.0, air_hz=10000, comp_ratio=3.0,
               comp_attack_ms=5, comp_release_ms=100, comp_threshold_db=-24.0, tempo=1.0, tempo_cap=1.10,
               pitch_st=0.0, gain_db=0.0, room_size=0.22, room_damping=0.65, room_wet=0.08, limiter_dbtp=-1.0,
               fade_in_ms=8, fade_out_ms=60, lead_ms=40, tail_ms=250)
def energy_to_params(energy: int, base: dict) -> dict   # presence=0.04*E dB, air=0.025*E dB, comp GR target=0.07*E dB, tempo=min(cap, 1+0.0012*E)
def master(raw_pcm: bytes, sr: int, params: dict) -> bytes
```
Chain, in order, with `pedalboard`: edge trim (onset at -50 dBFS keep `lead_ms`, keep `tail_ms`), `HighpassFilter(hpf_hz)`,
`PeakFilter(presence_hz, presence_db, presence_q)`, `HighShelfFilter(air_hz, air_db)`, `Compressor(threshold, ratio,
attack, release)`, tempo via `pedalboard.time_stretch` only if `tempo != 1.0`, fixed `Gain(gain_db)`, `Reverb(room_size,
damping, wet_level, dry_level=1-wet)`, `Limiter(threshold_db=limiter_dbtp)`, then deterministic peak scale-down to
-1 dBFS if needed, then fades. **No per-line loudness normalisation.** Byte-identical output for identical input+params.
Also: `def measure(pcm, sr) -> dict(peak_dbfs, lufs, dur_s)` using `pyloudnorm`.

## app/voices.py
```python
@dataclass
class Voice:
    id: str; label: str; lang: str; version: int
    ref_file: str; ref_sha256: str; ref_transcript: str; ref_tts_path: str   # /refs/<id>/ref.wav as the TTS sees it
    sampler: Sampler; golden_seed: int
    gate: dict          # mode strict|loose, baseline, p10, strict, loose, cer_max, calibrated_at
    master: dict        # master params (see master.DEFAULT) incl. energy
    banned_tokens: list[str]; persona: dict; fillers: list[str]; f0_band: list[int]
def load_voice(voice_id: str) -> Voice                   # data/voices/<id>/voice.yaml
def save_voice(v: Voice) -> None
def lock_reference(voice_id: str, wav_in: Path, transcript: str) -> Voice
    # ffmpeg → mono 24 kHz 16-bit, trim head/tail at -42 dBFS, peak -3 dBFS, cap 30 s, NO EQ; sha256; version += 1
def calibrate(v: Voice, lines: list[str], n_takes: int = 3) -> Voice
    # renders lines × takes (seeds golden_seed+take), sim per take → baseline=median, p10; strict=max(baseline-0.04,p10),
    # loose=baseline-0.08; gain_db so the set's median integrated loudness after master() = -18 LUFS; writes voice.yaml
```

## app/render.py
```python
RECIPE_VERSION: str          # sha256 of (sampler defaults + gate rules + CANON_VERSION + tokens.json) computed at import
@dataclass
class RenderResult:
    render_id: str; voice_id: str; text: str; canon: Canon; seed: int; take_no: int
    raw_pcm: bytes; pcm: bytes; sr: int; sim: float; cer: float | None; verified: bool; gate: str  # pass|failed
    scores: list[TakeScore]; timings: dict
def render_id(voice: Voice, canon_text: str, take_no: int) -> str
    # sha256(voice.id | voice.version | RECIPE_VERSION | canon_text | take_no)
def render_line(voice: Voice, text: str, take_no: int = 0, n_takes: int = 2) -> RenderResult
    # canonicalize (banned=voice.banned_tokens) → seeds = [golden_seed + take_no*8 + i for i in range(n)] (3 takes if
    # len(spoken) > 120) → synth_many → gate.select → master → RenderResult; one retry round (seeds +3..+5) if none passes
```

## app/store.py
SQLite at `DATA_DIR/app.db`, WAL. Tables exactly as REBUILD.md §6 (voices, lines, renders, jobs, scenes, playlists,
session, speaker). `def db() -> sqlite3.Connection`, `def init_db()`, `def put_render(r: RenderResult, line_id: str | None,
raw_path: str, path: str)`, `def get_render(render_id) -> dict | None`, `def upsert_line(voice_id, lang, category, text,
source) -> line_id` (id = sha1(lang|voice|category|text)), `def set_active_render(line_id, render_id)`.
Files: `renders_dir(voice)/<render_id>.raw.wav` and `<render_id>.<master.VERSION>.wav` (write via `wave`).

## scripts/bag_cli.py
`python -m scripts.bag_cli render --voice bag --text "..." [--take N] [--out x.wav]` → prints render_id, seed, sim, cer,
timings; `calibrate --voice bag [--lines data/golden/bag/lines.json]`; `audition --voice bag --seeds 0-9 --lines 10`
(renders 10 lines × seeds into `DATA_DIR/audition/<voice>/seed<k>_<i>.wav` for listening);
`golden --voice bag` (renders `tests/golden/<voice>/lines.json`, asserts SIM >= strict, CER <= 0.15, word ratio
0.75–1.25, F0 median in f0_band (librosa.pyin), loudness within ±2 LU of -18, duration/syllable 0.12–0.35 s; prints a
table and exits non-zero on failure).

## Tests (`tests/unit`, pytest, no network)
`test_canon.py` (drawl spelling, pause tokens, cap, banned tokens, syllables incl. clitics), `test_render_id.py`
(stable across processes; changes with take/version/text), `test_master.py` (byte-identical on repeat; energy 0 vs 100
changes output; tempo cap), `test_gate.py` (`sanity` on synthetic tones/silence; `select` ordering with a fake gate).

## Infra
`docker/app.Dockerfile`: python:3.12-slim + ffmpeg; pip `fastapi uvicorn[standard] pydantic pyyaml requests numpy
pedalboard pyloudnorm librosa webrtcvad-wheels resemblyzer` + CPU torch (`--index-url https://download.pytorch.org/whl/cpu`).
`docker/compose.yml`: replace the `orch` service with `app` (same host networking, `${HOME}/dnd-tts/refs:/refs`,
`${HOME}/dnd-tts/data:/data`, `${HOME}/dnd-tts/cache:/cache`, env `BAG_DATA=/data`), keep `tts` unchanged.
`configs/tokens.json`: the full list above with categories. `scripts/verify_tokens.py`: greps the cookbook doc inside
the tts image (`docker exec bag-tts grep -oE '<\|[a-z_]+:[a-z_]+\|>' .../higgs_tts.md`) and diffs against tokens.json.
`data/voices/bag/voice.yaml` seed: id bag, label "Mr. Bag", lang sk, version 1, ref_file ref.wav (copied from
`/refs/bag_ref.wav`), transcript from `server/orchestrator.py` VOICES["bag"]["text"], ref_tts_path `/refs/bag/ref.wav`,
sampler defaults, golden_seed 0, gate uncalibrated, master DEFAULT with energy 65, banned tokens as above, persona from
`server/orchestrator.py` (BAG_PERSONA / BAG_PERSONA_EN incl. SK_RULES), fillers `["Hmm…", "No…", "Počkaj…", "Tak…",
"Ehm…", "Moment…"]`, f0_band [60, 155].
