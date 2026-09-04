"""The Voice Creator: a clip in, a table-ready voice out (M4 contract, section 4).

The previous build let anyone drop a wav in and start speaking, and the voices
drifted. This module exists so that every new voice walks the same road Bag did,
in the same order, with a human in it at the two places a machine gets it wrong:

* the **window** is picked by the DM, because a crowd reaction or music in the
  middle of the clip poisons the clone (that is what happened to the shopkeep);
* the **transcript** is whisper's first draft and then the DM's, because a wrong
  reference transcript is the single most common cause of a bad clone.

Everything else is measured, never guessed. ``f0_band`` comes out of the clip's
own pitch, so the golden test later checks the voice against what the voice
actually is. The gate thresholds and the fixed gain come out of ``voices.calibrate``
run on the character's own generated lines, so the voice is measured on the kind
of sentence it will speak at the table.

A draft is a directory under ``DATA_DIR/creator/<draft_id>/`` holding the upload,
the trimmed clip and ``draft.json``. State lives there and nowhere else, so a
reload -- or a container restart -- resumes the draft exactly where the DM left it.

Nothing here imports FastAPI; ``app/api/creator.py`` is the only HTTP face, and
``commit`` takes the worker and the event sink as arguments so this module can be
tested without either.
"""
from __future__ import annotations

import json
import logging
import re
import shutil
import subprocess
import unicodedata
import uuid
import wave
from contextlib import closing
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path

import numpy as np
import yaml

from app import board, config, llm, store, stt, voices

log = logging.getLogger(__name__)

CLIP_MAX_S = 30.0                  # the DM picks the window; longer would be cut by lock_reference anyway
CLIP_SR = voices.REF_SR            # what lock_reference converts to, so the audition is what gets locked
MAX_UPLOAD_BYTES = 25 * 1024 * 1024
CALIBRATION_LINES = 30             # same count new_voice.py uses: enough SIM samples, ~90 takes of GPU time
LINE_SOURCE = "creator"            # not "bank": the DM generated these, and Prep may edit them
PRERENDER_KIND = "bank"            # the job kind app/main.py queues the board with at boot
STT_TIMEOUT_S = 60.0               # no one is mid-scene; a 30 s clip is worth waiting for

# librosa.pyin's search range. Wider than the band it produces: the median has
# to be found before it can be trusted, and a bass voice sits under 60 Hz.
F0_FMIN, F0_FMAX = 50.0, 500.0
F0_BAND_LO, F0_BAND_HI = 0.65, 1.5
F0_CLAMP = (50, 420)
DEFAULT_F0_BAND = [F0_CLAMP[0], F0_CLAMP[1]]

VOICE_ID_RE = re.compile(r"^[a-z0-9_-]{2,24}$")
# Bag and the three seeded voices. A new voice may never take one of these ids:
# it would overwrite a locked reference and every render_id that hangs off it.
RESERVED_VOICE_IDS = frozenset({"bag", "male", "female", "shopkeep"})
_DRAFT_ID_RE = re.compile(r"^[0-9a-f]{32}$")
_SLUG_STRIP = re.compile(r"[^a-z0-9]+")
_SUFFIX_OK = re.compile(r"^\.[a-z0-9]{1,8}$")

STATUSES = ("new", "clipped", "transcribed", "described", "locked", "calibrated", "banked", "done", "failed")
# (step, percent) in the order commit runs them; the Creator page's progress bar
# reads nothing else, so the order here is the order the DM sees.
COMMIT_STEPS = (("lock", 10), ("calibrate", 35), ("bank", 70), ("queue", 90), ("done", 100))


class CreatorError(RuntimeError):
    """Anything the Creator cannot do with a draft as it stands."""


class DraftNotFound(KeyError):
    """No draft directory with that id; the API maps this to 404."""


class BadWindow(ValueError):
    """The clip window is empty, negative or outside the upload."""


class BadVoiceId(ValueError):
    """The voice id is malformed, reserved, or already taken by a locked voice."""


class ConversionFailed(CreatorError):
    """ffmpeg could not turn the upload into a reference clip."""


class SttDown(CreatorError):
    """Whisper could not answer. The gate treats that as "unverified", but here
    the transcript *is* the request, so the DM is told instead of handed silence."""


@dataclass
class Draft:
    """One voice being built. Persisted whole as ``draft.json`` on every change.

    ``result`` is not in the sketch in the contract; it holds what ``commit``
    returned so that a reload after the commit still shows the gate numbers the
    final card is made of. Everything else is exactly the contract's shape.
    """

    id: str
    voice_id: str
    label: str
    lang: str
    source_name: str
    clip_path: str
    start_s: float
    end_s: float
    transcript: str
    description: str
    f0_band: list[int]
    persona: dict
    categories: list[str]
    phrases: dict[str, list[str]]
    status: str
    result: dict = field(default_factory=dict)

    def as_dict(self) -> dict:
        return asdict(self)


# -- draft storage -----------------------------------------------------------


def _root() -> Path:
    """Read at call time, not at import, so a test's ``BAG_DATA`` reload lands."""
    return config.DATA_DIR / "creator"


def _dir(draft_id: str) -> Path:
    """The draft's directory, after proving the id is an id.

    WHY the regex: this value arrives straight from a URL path, and a draft id
    is the only thing between it and the filesystem.
    """
    if not _DRAFT_ID_RE.match(draft_id or ""):
        raise DraftNotFound(draft_id)
    return _root() / draft_id


def _save(draft: Draft) -> Draft:
    """Atomic write: a crash mid-save must not leave a draft that cannot load.

    The status is checked here because this is the one door every step goes
    through, and a mistyped status would only show up much later as a Creator
    page stuck on a step that does not exist.
    """
    if draft.status not in STATUSES:
        raise ValueError(f"neznámy stav konceptu {draft.status!r}")
    folder = _dir(draft.id)
    folder.mkdir(parents=True, exist_ok=True)
    tmp = folder / "draft.json.tmp"
    tmp.write_text(json.dumps(draft.as_dict(), ensure_ascii=False, indent=2), encoding="utf-8")
    tmp.replace(folder / "draft.json")
    return draft


def get(draft_id: str) -> Draft:
    path = _dir(draft_id) / "draft.json"
    if not path.is_file():
        raise DraftNotFound(draft_id)
    data = json.loads(path.read_text(encoding="utf-8"))
    known = {f.name for f in fields(Draft)}
    return Draft(**{k: v for k, v in data.items() if k in known})


def drafts() -> list[Draft]:
    """Every draft, newest first, skipping any directory that will not load: a
    half-written draft must not take the Creator page down with it."""
    root = _root()
    if not root.is_dir():
        return []
    out = []
    for path in sorted(root.glob("*/draft.json"), key=lambda p: p.stat().st_mtime, reverse=True):
        try:
            out.append(get(path.parent.name))
        except (ValueError, KeyError, OSError):
            log.warning("unreadable draft %s", path.parent.name)
    return out


def discard(draft_id: str) -> None:
    """Delete the draft and its audio. A committed voice is untouched: it lives
    under ``voices/<id>/`` and stopped being a draft the moment it locked."""
    shutil.rmtree(_dir(draft_id), ignore_errors=True)


# -- step 1: the upload and its window ---------------------------------------


def _suffix(filename: str) -> str:
    """The upload's extension, sanitised, because ffmpeg picks its demuxer from
    the name and the name comes from a browser."""
    suffix = Path(filename or "").suffix.lower()
    return suffix if _SUFFIX_OK.match(suffix) else ".bin"


def slug(label: str) -> str:
    """A voice id the DM would have typed themselves: ascii, lower, hyphenated."""
    folded = unicodedata.normalize("NFKD", label or "").encode("ascii", "ignore").decode().lower()
    out = _SLUG_STRIP.sub("-", folded).strip("-")[:24].strip("-")
    return out if len(out) >= 2 else "hlas"


def suggest_voice_id(label: str) -> str:
    """A free id derived from the label, so a fresh draft starts valid."""
    base = slug(label)
    for n in range(1, 100):
        candidate = base if n == 1 else f"{base[:22]}-{n}"
        if candidate not in RESERVED_VOICE_IDS and not config.voice_dir(candidate).exists():
            return candidate
    raise BadVoiceId(f"nepodarilo sa odvodiť voľné id z {label!r}")


def check_voice_id(voice_id: str) -> str:
    """The three rules a new voice id must pass, checked together so the API can
    answer with one reason. A taken id is refused rather than merged: it would
    overwrite a locked reference and silently invalidate every cached render."""
    if not VOICE_ID_RE.match(voice_id or ""):
        raise BadVoiceId("id hlasu musí byť 2-24 znakov z a-z, 0-9, _ a -")
    if voice_id in RESERVED_VOICE_IDS:
        raise BadVoiceId(f"id {voice_id!r} je vyhradené")
    if config.voice_dir(voice_id).exists():
        raise BadVoiceId(f"hlas {voice_id!r} už existuje")
    return voice_id


def create(upload: bytes, filename: str, label: str, lang: str) -> Draft:
    """Take the DM's file and open a draft around it. Nothing is converted yet:
    the window comes first, because converting the whole file would only produce
    audio the DM is about to throw most of away."""
    if not upload:
        raise ValueError("prázdny súbor")
    if len(upload) > MAX_UPLOAD_BYTES:
        raise ValueError(f"súbor je väčší ako {MAX_UPLOAD_BYTES // (1024 * 1024)} MB")
    label = " ".join((label or "").split()) or "Nový hlas"
    draft_id = uuid.uuid4().hex
    folder = _root() / draft_id
    folder.mkdir(parents=True, exist_ok=True)
    (folder / f"source{_suffix(filename)}").write_bytes(upload)
    return _save(Draft(id=draft_id, voice_id=suggest_voice_id(label), label=label,
                       lang=lang or "sk", source_name=filename or "upload", clip_path="",
                       start_s=0.0, end_s=0.0, transcript="", description="",
                       f0_band=list(DEFAULT_F0_BAND), persona={}, categories=[],
                       phrases={}, status="new"))


def _source(draft: Draft) -> Path:
    matches = sorted(_dir(draft.id).glob("source.*"))
    if not matches:
        raise CreatorError("nahraný súbor sa nenašiel")
    return matches[0]


def _convert(src: Path, start_s: float, dur_s: float) -> bytes:
    """ffmpeg -> mono 24 kHz int16 PCM of exactly the chosen window.

    Same target format ``voices.lock_reference`` converts to, so what the DM
    auditions is what gets locked. No EQ, no denoise, no normalisation: the clip
    *is* the voice, and lock_reference does its own trim and peak scaling.
    """
    cmd = ["ffmpeg", "-nostdin", "-v", "error", "-ss", f"{start_s:.3f}", "-i", str(src),
           "-t", f"{dur_s:.3f}", "-ac", "1", "-ar", str(CLIP_SR), "-f", "s16le", "pipe:1"]
    try:
        done = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True)
    except FileNotFoundError as e:
        raise ConversionFailed("ffmpeg nie je dostupný") from e
    except subprocess.CalledProcessError as e:
        raise ConversionFailed(f"ffmpeg zlyhal: {e.stderr.decode('utf-8', 'replace').strip()[:200]}") from e
    if not done.stdout:
        raise ConversionFailed("v tomto okne nie je žiadny zvuk")
    return done.stdout


def clip(draft_id: str, start_s: float, end_s: float) -> Draft:
    """Cut the chosen window, then measure the voice's own pitch band from it.

    The window is capped at 30 s rather than refused: ``lock_reference`` truncates
    there anyway, and silently locking a different clip than the one the DM
    auditioned is exactly the kind of drift this rebuild exists to stop -- so the
    cap is applied here, written back into the draft, and shown.

    Re-clipping clears the transcript on purpose: a transcript describes one
    window, and carrying it over to another is how a clone gets a reference whose
    words do not match its audio.
    """
    draft = get(draft_id)
    start, end = float(start_s), float(end_s)
    if not (np.isfinite(start) and np.isfinite(end)) or start < 0 or end <= start:
        raise BadWindow("okno musí začínať pred koncom a nesmie byť záporné")
    end = min(end, start + CLIP_MAX_S)
    pcm = _convert(_source(draft), start, end - start)
    out = _dir(draft_id) / "clip.wav"
    store.write_wav(out, pcm, CLIP_SR)
    draft.clip_path = str(out)
    draft.start_s, draft.end_s = round(start, 3), round(end, 3)
    draft.f0_band = f0_band(pcm, CLIP_SR)
    draft.transcript = ""
    draft.status = "clipped"
    return _save(draft)


def _pyin(y: np.ndarray, sr: int) -> tuple[np.ndarray, np.ndarray]:
    """librosa's probabilistic YIN behind one seam.

    Its own import costs seconds and pulls in numba, so it happens here, only
    when a clip is actually being measured -- and a test can replace this one
    function instead of the whole library.
    """
    import librosa  # slow import; only a new clip needs it

    f0, voiced, _ = librosa.pyin(y, fmin=F0_FMIN, fmax=F0_FMAX, sr=sr)
    return f0, voiced


def median_f0(pcm: bytes, sr: int) -> float:
    """Median voiced F0 in Hz; 0.0 when nothing voiced was found (music, noise,
    a window the DM mis-picked). Median, not mean, because one octave error on a
    single frame would drag a mean across the whole band."""
    y = np.frombuffer(pcm, dtype=np.int16).astype(np.float32) / 32768.0
    if y.size == 0:
        return 0.0
    f0, voiced = _pyin(y, sr)
    f0 = np.asarray(f0, dtype=np.float64)
    keep = np.asarray(voiced, dtype=bool) & ~np.isnan(f0)
    return float(np.median(f0[keep])) if keep.any() else 0.0


def f0_band(pcm: bytes, sr: int) -> list[int]:
    """The golden test's pitch window for this voice, measured not guessed.

    Asymmetric around the median (0.65x below, 1.5x above) because a speaker's
    pitch climbs far further when they raise their voice than it drops when they
    settle, and clamped to 50-420 Hz so a mis-measured clip cannot hand the
    golden test a band that would pass anything.
    """
    median = median_f0(pcm, sr)
    if median <= 0.0:
        log.warning("no voiced pitch in the clip; keeping the default f0 band")
        return list(DEFAULT_F0_BAND)
    lo = max(F0_CLAMP[0], int(round(F0_BAND_LO * median)))
    hi = min(F0_CLAMP[1], int(round(F0_BAND_HI * median)))
    return [lo, hi] if lo < hi else list(DEFAULT_F0_BAND)


# -- step 2: the transcript --------------------------------------------------


def transcribe(draft_id: str) -> Draft:
    """Whisper's first draft of what the clip says. The DM corrects it before the
    lock; this is only the typing saved, never the last word."""
    draft = get(draft_id)
    clip_file = Path(draft.clip_path or "")
    if not clip_file.is_file():
        raise CreatorError("najprv orež klip")
    pcm, sr = _read_clip(clip_file)
    heard = stt.transcribe(pcm, sr, draft.lang, timeout=STT_TIMEOUT_S)
    if heard is None:
        raise SttDown("whisper neodpovedal; prepis napíš ručne")
    draft.transcript = " ".join(heard.split())
    draft.status = "transcribed"
    return _save(draft)


def _read_clip(path: Path) -> tuple[bytes, int]:
    with closing(wave.open(str(path), "rb")) as w:
        return w.readframes(w.getnframes()), w.getframerate()


# -- step 3 and 4: the character and its soundboard --------------------------


def describe(draft_id: str, description: str) -> Draft:
    """The Writer turns the DM's sentence about the character into the persona
    that will ride in every later ``fix`` prompt, plus the soundboard's tabs.
    The last category is the signature refusal; ``commit`` relies on that order."""
    draft = get(draft_id)
    draft.description = " ".join((description or "").split())
    if not draft.description:
        raise ValueError("opis postavy je prázdny")
    written = llm.persona(draft.label, draft.description, lang=draft.lang)
    draft.persona = {"sk": written["sk"], "en": written["en"]}
    draft.categories = list(written["categories"])
    draft.status = "described"
    return _save(draft)


def write_phrases(draft_id: str, per_category: int = 10) -> Draft:
    """Fill the soundboard. Regenerating replaces the whole board on purpose: a
    DM who edited single lines and then pressed Generate again would otherwise
    get a silent merge of two different drafts of the character."""
    draft = get(draft_id)
    if not draft.categories:
        raise CreatorError("najprv vygeneruj postavu a kategórie")
    draft.phrases = llm.phrases(draft.persona, draft.categories, lang=draft.lang,
                                per_category=per_category)
    return _save(draft)


def set_phrases(draft_id: str, phrases: dict[str, list[str]]) -> Draft:
    """The DM's own edits to the board: every line re-checked exactly the way a
    generated one was, so hand-typed text cannot smuggle in a control token or a
    line the renderer would refuse later."""
    draft = get(draft_id)
    cleaned: dict[str, list[str]] = {}
    for category, lines in (phrases or {}).items():
        name = " ".join(str(category).split())
        if not name:
            continue
        kept = [line for line in (_clean_line(text, draft.lang) for text in lines or []) if line]
        cleaned[name] = list(dict.fromkeys(kept))
    draft.phrases = cleaned
    return _save(draft)


def _clean_line(text: object, lang: str) -> str | None:
    """One line the DM typed, or None when it could never become a tile."""
    return llm.usable_line(text, lang)


def update(draft_id: str, *, transcript: str | None = None, label: str | None = None,
           voice_id: str | None = None, categories: list[str] | None = None) -> Draft:
    """The DM's own edits to what whisper and the Writer filled in.

    The transcript is stored exactly as typed apart from whitespace: the DM is
    correcting whisper word for word against the clip, and a helpful rewrite
    here would defeat the whole point of showing it. The voice id is validated
    now rather than at commit, so a taken name is a red field on step 3 instead
    of a failure after the GPU has already spent three minutes calibrating.
    """
    draft = get(draft_id)
    if voice_id is not None:
        draft.voice_id = check_voice_id(" ".join(voice_id.split()).lower())
    if label is not None:
        draft.label = " ".join(label.split()) or draft.label
    if transcript is not None:
        draft.transcript = " ".join(transcript.split())
        if draft.transcript and draft.status == "clipped":
            draft.status = "transcribed"
    if categories is not None:
        names = list(dict.fromkeys(" ".join(str(name).split()) for name in categories))
        draft.categories = [name for name in names if name]
    return _save(draft)


def thin_categories(draft: Draft) -> list[str]:
    """Categories with fewer than three lines. The Creator page warns on these;
    they are not an error, because three real lines beat ten padded ones."""
    return [name for name, lines in draft.phrases.items() if len(lines) < llm.MIN_LINES]


# -- step 5: commit ----------------------------------------------------------


def calibration_lines(draft: Draft, limit: int = CALIBRATION_LINES) -> list[str]:
    """The character's own longest lines.

    WHY longest first (the same rule ``scripts/new_voice.py`` follows): the gain
    solver needs takes the loudness meter can gate at all -- under ~400 ms it
    reads -inf -- and a similarity baseline measured on "Hej." describes nothing.
    """
    texts = {line for lines in draft.phrases.values() for line in lines}
    return sorted(texts, key=len, reverse=True)[:limit]


def commit(draft_id: str, *, worker=None, on_event=None) -> dict:
    """Lock, calibrate, bank, queue -- in that order, because each step needs the
    one before it: nothing may render until the reference is locked, the gate
    thresholds mean nothing until they are measured against that reference, and
    a pre-render queued before the lines exist would have nothing to render.

    Progress goes out as ``creator.progress {draft_id, step, pct}`` through the
    same ``on_event`` the worker uses, so the Creator page's bar rides the one
    WebSocket the rest of the app already has. ``worker`` and ``on_event`` are
    keyword-only and optional so this whole sequence can be run in a test with
    neither, exactly as the contract's ``commit(draft_id)`` signature reads.

    Returns ``{voice_id, gate, gain_db, lines, queued}``.
    """
    draft = get(draft_id)
    _ready_to_commit(draft)
    try:
        v = _lock(draft, on_event)
        v = _calibrate(draft, v, on_event)
        lines = _bank(draft, on_event)
        queued = _queue(draft, worker, on_event)
    except Exception:
        draft.status = "failed"
        _save(draft)
        _progress(on_event, draft, "failed", 100)
        raise
    draft.status = "done"
    draft.result = {"voice_id": v.id, "gate": dict(v.gate), "gain_db": v.master.get("gain_db"),
                    "lines": lines, "queued": queued}
    _save(draft)
    _progress(on_event, draft, "done", 100)
    return dict(draft.result)


def _ready_to_commit(draft: Draft) -> None:
    """Everything that would fail halfway through, checked before anything is
    written: a commit that dies after the lock leaves a half-built voice on disk."""
    check_voice_id(draft.voice_id)
    if not Path(draft.clip_path or "").is_file():
        raise CreatorError("chýba orezaný klip")
    if not draft.transcript.strip():
        raise CreatorError("prepis referencie je prázdny")
    if not calibration_lines(draft):
        raise CreatorError("žiadne repliky na kalibráciu")


def _step_pct(step: str) -> int:
    return dict(COMMIT_STEPS).get(step, 0)


def _progress(on_event, draft: Draft, step: str, pct: int | None = None) -> None:
    """A listener's failure is logged, never propagated: a browser that dropped
    its socket must not abort a commit that is already writing to disk."""
    if on_event is None:
        return
    try:
        on_event("creator.progress", {"draft_id": draft.id, "step": step,
                                      "pct": _step_pct(step) if pct is None else pct})
    except Exception:  # noqa: BLE001
        log.exception("creator.progress listener failed on %s", step)


def _lock(draft: Draft, on_event) -> voices.Voice:
    """Make the clip this voice's immutable reference, then write what only the
    Creator knows: the label, the language, the measured pitch band and the
    persona. Saved after the lock because ``lock_reference`` writes voice.yaml."""
    _progress(on_event, draft, "lock")
    v = voices.lock_reference(draft.voice_id, Path(draft.clip_path), draft.transcript)
    v.label = draft.label
    v.lang = draft.lang
    v.f0_band = list(draft.f0_band)
    v.persona = dict(draft.persona)
    voices.save_voice(v)
    draft.status = "locked"
    _save(draft)
    return v


def _calibrate(draft: Draft, v: voices.Voice, on_event) -> voices.Voice:
    """Measure the voice against itself on its own longest lines. This is the
    step that turns a locked clip into a voice with a gate, and it is why a
    Creator voice is as trustworthy as Bag rather than merely as loud."""
    _progress(on_event, draft, "calibrate")
    v = voices.calibrate(v, calibration_lines(draft))
    draft.status = "calibrated"
    _save(draft)
    return v


def _bank(draft: Draft, on_event) -> int:
    """Import the generated lines as this voice's bank, put the first line of
    each ordinary category on a favourites key, and register the last category
    as the signature refusal so the giant T tile has something to show."""
    _progress(on_event, draft, "bank")
    ensure_source()
    signature = draft.categories[-1] if draft.categories else ""
    ordinary: list[str] = []
    count = 0
    for category in draft.categories:
        first = True
        for text in draft.phrases.get(category, []):
            line = board.add_line(draft.voice_id, draft.lang, category, text, source=LINE_SOURCE)
            count += 1
            if first and category != signature:
                ordinary.append(line["id"])
            first = False
    for slot, line_id in enumerate(ordinary[:board.DEFAULT_SLOTS], start=1):
        board.set_line(line_id, slot=slot)
    if signature:
        set_signature(draft.voice_id, draft.lang, signature)
    draft.status = "banked"
    _save(draft)
    return count


def _queue(draft: Draft, worker, on_event) -> int:
    """Pre-render the whole board at ``batch`` priority, favourites first (that
    is what ``prerender_plan`` orders), so the DM can walk back to Play and find
    the tiles filling in instead of a grid of pending greys."""
    _progress(on_event, draft, "queue")
    if worker is None:
        return 0
    from app.jobs import new_job  # the worker owner's module; bound here, not at import

    texts = {line["id"]: line["text"] for line in board.board(draft.voice_id, draft.lang)["lines"]}
    queued = 0
    for line_id in board.prerender_plan(draft.voice_id, draft.lang):
        text = texts.get(line_id)
        if text is None:
            continue
        worker.submit(new_job(PRERENDER_KIND, "batch", draft.voice_id, text, line_id=line_id))
        queued += 1
    return queued


# -- the two places M4 widens what M1/M2 already wrote ------------------------


def set_signature(voice_id: str, lang: str, category: str) -> None:
    """Tell the board which category holds this voice's refusal.

    Prefers ``board.set_signature`` when the board owner has shipped it, because
    the board owns that table; the fallback does exactly what the contract
    describes -- the in-memory entry plus ``signature_category`` in voice.yaml --
    so a voice created before that lands still gets its T tile.

    Written after every ``save_voice`` in the commit, because ``save_voice``
    serialises the ``Voice`` dataclass and would drop a key it has no field for.
    """
    setter = getattr(board, "set_signature", None)
    if setter is not None:
        setter(voice_id, lang, category)
        return
    board.SIGNATURE_CATEGORY.setdefault(lang, {})[board.BANK_KEY.get(voice_id, voice_id)] = category
    path = voices.voice_yaml(voice_id)
    if not path.exists():
        return
    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    data.setdefault("signature_category", {})[lang] = category
    path.write_text(yaml.safe_dump(data, sort_keys=False, allow_unicode=True), encoding="utf-8")


def ensure_source(source: str = LINE_SOURCE) -> None:
    """Let ``lines.source`` hold ``creator``.

    The M1 schema pins ``source`` to a CHECK list written before the Voice
    Creator existed, and SQLite cannot widen a CHECK in place. The table is
    rebuilt from **its own stored DDL**, so the constraint is derived rather than
    copied and cannot drift from ``store.SCHEMA``; the indexes come back from
    ``store.init_db()``, which is idempotent. Same spirit as
    ``jobs.ensure_columns``: an existing database is brought up to what this
    milestone needs without losing a row.
    """
    marker = "CHECK(source IN ("
    with closing(store.db()) as con, con:
        row = con.execute("SELECT sql FROM sqlite_master WHERE type='table' AND name='lines'").fetchone()
        ddl = "" if row is None else str(row["sql"] or "")
        start = ddl.find(marker)
        if not ddl or start < 0 or f"'{source}'" in ddl:
            return
        close = ddl.index(")", start + len(marker))
        widened = re.sub(r"^CREATE\s+TABLE\s+(IF\s+NOT\s+EXISTS\s+)?[\"'`\[]?lines[\"'`\]]?",
                         "CREATE TABLE lines_new", ddl[:close] + f",'{source}'" + ddl[close:],
                         count=1, flags=re.I)
        con.execute(widened)
        con.execute("INSERT INTO lines_new SELECT * FROM lines")
        con.execute("DROP TABLE lines")
        con.execute("ALTER TABLE lines_new RENAME TO lines")
    store.init_db()
    log.info("lines.source widened to accept %r", source)


__all__ = ["Draft", "CreatorError", "DraftNotFound", "BadWindow", "BadVoiceId", "ConversionFailed",
           "SttDown", "create", "clip", "transcribe", "describe", "write_phrases", "set_phrases",
           "update", "commit", "get", "drafts", "discard", "check_voice_id", "suggest_voice_id", "f0_band",
           "median_f0", "thin_categories", "calibration_lines", "set_signature", "ensure_source"]
