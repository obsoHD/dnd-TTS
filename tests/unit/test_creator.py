"""The Voice Creator, with everything outside this process replaced.

Nothing here reaches the network, the GPU box, whisper, ollama or ffmpeg: the
conversion is one stubbed function returning a synthetic sine, ``app.stt`` and
``app.llm`` are replaced per test, ``voices.calibrate`` records its arguments
instead of rendering, and the worker is a list of submitted jobs.

What is worth pinning here is exactly what makes a Creator voice as trustworthy
as Bag: the draft survives a reload, the window is capped before it reaches the
lock, the pitch band is measured from the clip instead of guessed, a reserved or
taken id is refused, the Writer's lines are filtered rather than believed, and
``commit`` runs lock -> calibrate -> import -> queue in that order, because every
step there depends on the one before it.
"""
from __future__ import annotations

import importlib
import json
import math
from pathlib import Path

import numpy as np
import pytest
import yaml

from app import board, canon, config, creator, llm, store, voices
from app.tts_client import Sampler
from app.voices import Voice

SR = creator.CLIP_SR
LABEL = "Krčmár Bruno"
LANG = "sk"
CATEGORIES = ["Vítanie hostí", "Účet za pivo", "Nalej ešte", "Toto ti nenalejem"]
PHRASES = {
    "Vítanie hostí": ["Vitaj, pocestný.", "Sadni si k ohňu.", "Dnes máme čerstvé pivo."],
    "Účet za pivo": ["To robí tri strieborné.", "Platíš ty, alebo tá elfka?", "Účet rastie, kamoš."],
    "Nalej ešte": ["Ešte jedno?", "Nalejem, ale posledné.", "Máš na to ešte hrdlo?"],
    "Toto ti nenalejem": ["Toto ti nenalejem.", "Dnes už nie.", "Choď domov, chlape."],
}


# -- fixtures ----------------------------------------------------------------


@pytest.fixture
def data_dir(tmp_path, monkeypatch):
    """Point the whole app at a temp DATA_DIR by env, the way the container does."""
    monkeypatch.setenv("BAG_DATA", str(tmp_path))
    importlib.reload(config)
    yield tmp_path
    monkeypatch.undo()
    importlib.reload(config)


@pytest.fixture
def fresh_db(data_dir) -> Path:
    store.init_db()
    return data_dir


@pytest.fixture
def no_ffmpeg(monkeypatch):
    """``_convert`` replaced by a synthetic 150 Hz clip of exactly the asked-for
    window, so the window arithmetic is tested without an ffmpeg on the box."""
    calls: list[dict] = []

    def convert(src: Path, start_s: float, dur_s: float) -> bytes:
        calls.append({"src": Path(src), "start_s": start_s, "dur_s": dur_s})
        return sine(dur_s)

    monkeypatch.setattr(creator, "_convert", convert)
    return calls


@pytest.fixture
def flat_pitch(monkeypatch):
    """``librosa.pyin`` replaced by a steady 150 Hz reading; the library is a
    slow optional import and the band arithmetic is what this suite is about."""
    monkeypatch.setattr(creator, "_pyin", lambda y, sr: pyin_at(150.0, y))
    return 150.0


def sine(seconds: float, hz: float = 150.0) -> bytes:
    """int16 mono PCM of a pure tone: enough for pitch, length and file writing."""
    n = max(1, int(seconds * SR))
    t = np.arange(n, dtype=np.float32) / SR
    return (0.3 * np.sin(2 * math.pi * hz * t) * 32767).astype(np.int16).tobytes()


def pyin_at(hz: float, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    frames = max(1, y.size // 512)
    return np.full(frames, hz), np.ones(frames, dtype=bool)


def make_draft(**over) -> creator.Draft:
    """A draft as it stands just before ``commit``, without walking the steps."""
    data = dict(id="0" * 32, voice_id="krcmar", label=LABEL, lang=LANG, source_name="bruno.wav",
                clip_path="", start_s=0.0, end_s=12.0, transcript="Vitaj, pocestný.",
                description="Zhovorčivý krčmár.", f0_band=[98, 225],
                persona={"sk": "Krčmár.", "en": "An innkeeper."}, categories=list(CATEGORIES),
                phrases={k: list(v) for k, v in PHRASES.items()}, status="described")
    return creator.Draft(**{**data, **over})


def make_voice(voice_id: str = "krcmar") -> Voice:
    return Voice(id=voice_id, label=LABEL, lang=LANG, version=1, ref_file="ref.wav",
                 ref_sha256="0" * 64, ref_transcript="Vitaj, pocestný.",
                 ref_tts_path=f"/refs/{voice_id}/ref.wav", sampler=Sampler(), golden_seed=0,
                 gate=dict(voices.UNCALIBRATED_GATE), master={"energy": 65})


def started(data_dir, no_ffmpeg, flat_pitch, seconds: float = 40.0) -> creator.Draft:
    """A fresh draft whose upload is ``seconds`` long."""
    return creator.create(sine(seconds), "bruno.wav", LABEL, LANG)


class FakeWorker:
    """The worker as ``commit`` uses it: one method, and a record of the jobs."""

    def __init__(self) -> None:
        self.jobs: list = []

    def submit(self, job):
        self.jobs.append(job)
        return job


# -- the draft state machine and its persistence -----------------------------


def test_create_opens_a_draft_that_reloads_byte_for_byte(data_dir, no_ffmpeg, flat_pitch):
    draft = started(data_dir, no_ffmpeg, flat_pitch)
    assert draft.status == "new"
    assert draft.voice_id == "krcmar-bruno"      # slugged from the label, diacritics folded
    assert (data_dir / "creator" / draft.id / "source.wav").is_file()
    assert creator.get(draft.id) == draft
    assert [d.id for d in creator.drafts()] == [draft.id]


def test_the_steps_walk_the_contract_status_ladder(data_dir, no_ffmpeg, flat_pitch, monkeypatch):
    monkeypatch.setattr(creator.stt, "transcribe", lambda *a, **kw: "Vitaj,  pocestný.")
    monkeypatch.setattr(creator.llm, "persona", lambda label, desc, lang="sk": {
        "sk": "Krčmár.", "en": "An innkeeper.", "label": label, "description": desc,
        "categories": list(CATEGORIES)})
    monkeypatch.setattr(creator.llm, "phrases", lambda *a, **kw: {k: list(v) for k, v in PHRASES.items()})

    draft = started(data_dir, no_ffmpeg, flat_pitch)
    assert creator.clip(draft.id, 1.0, 6.0).status == "clipped"
    assert creator.transcribe(draft.id).transcript == "Vitaj, pocestný."
    assert creator.get(draft.id).status == "transcribed"
    described = creator.describe(draft.id, "Zhovorčivý krčmár, hovorí pomaly.")
    assert described.status == "described"
    assert described.categories == CATEGORIES
    assert creator.write_phrases(draft.id).phrases == PHRASES
    # Every step went to disk, not just to the object the caller was handed.
    on_disk = json.loads((data_dir / "creator" / draft.id / "draft.json").read_text(encoding="utf-8"))
    assert on_disk["transcript"] == "Vitaj, pocestný."
    assert on_disk["persona"]["en"] == "An innkeeper."


def test_reclipping_drops_the_transcript(data_dir, no_ffmpeg, flat_pitch, monkeypatch):
    """A transcript describes one window. Carrying it to another is how a clone
    gets a reference whose words do not match its audio."""
    monkeypatch.setattr(creator.stt, "transcribe", lambda *a, **kw: "Vitaj, pocestný.")
    draft = started(data_dir, no_ffmpeg, flat_pitch)
    creator.clip(draft.id, 0.0, 5.0)
    creator.transcribe(draft.id)
    assert creator.clip(draft.id, 8.0, 12.0).transcript == ""


def test_transcribe_says_so_when_whisper_is_down(data_dir, no_ffmpeg, flat_pitch, monkeypatch):
    monkeypatch.setattr(creator.stt, "transcribe", lambda *a, **kw: None)
    draft = started(data_dir, no_ffmpeg, flat_pitch)
    creator.clip(draft.id, 0.0, 5.0)
    with pytest.raises(creator.SttDown):
        creator.transcribe(draft.id)


def test_unknown_and_malformed_draft_ids_are_not_found(data_dir):
    with pytest.raises(creator.DraftNotFound):
        creator.get("f" * 32)
    with pytest.raises(creator.DraftNotFound):
        creator.get("../../etc")          # the id is all that stands between a URL and the disk


def test_discard_removes_the_draft_and_its_audio(data_dir, no_ffmpeg, flat_pitch):
    draft = started(data_dir, no_ffmpeg, flat_pitch)
    creator.discard(draft.id)
    assert not (data_dir / "creator" / draft.id).exists()
    assert creator.drafts() == []


# -- the clip: the 30 s cap and the measured band ----------------------------


def test_the_window_is_capped_at_thirty_seconds(data_dir, no_ffmpeg, flat_pitch):
    draft = started(data_dir, no_ffmpeg, flat_pitch, seconds=60.0)
    clipped = creator.clip(draft.id, 5.0, 55.0)
    assert clipped.start_s == 5.0
    assert clipped.end_s == 35.0                       # 5 + CLIP_MAX_S, written back so the DM sees it
    assert no_ffmpeg[-1]["dur_s"] == pytest.approx(creator.CLIP_MAX_S)
    assert Path(clipped.clip_path).is_file()


@pytest.mark.parametrize("start, end", [(-1.0, 5.0), (5.0, 5.0), (7.0, 2.0)])
def test_a_bad_window_is_refused(data_dir, no_ffmpeg, flat_pitch, start, end):
    draft = started(data_dir, no_ffmpeg, flat_pitch)
    with pytest.raises(creator.BadWindow):
        creator.clip(draft.id, start, end)


def test_ffmpeg_failure_is_a_conversion_error(data_dir, monkeypatch, flat_pitch):
    monkeypatch.setattr(creator, "_convert", lambda *a: (_ for _ in ()).throw(
        creator.ConversionFailed("ffmpeg zlyhal")))
    draft = creator.create(sine(4.0), "bruno.wav", LABEL, LANG)
    with pytest.raises(creator.ConversionFailed):
        creator.clip(draft.id, 0.0, 2.0)


def test_the_band_is_measured_from_the_clip(data_dir, no_ffmpeg, flat_pitch):
    draft = started(data_dir, no_ffmpeg, flat_pitch)
    clipped = creator.clip(draft.id, 0.0, 4.0)
    assert clipped.f0_band == [98, 225]                # 0.65 x 150 and 1.5 x 150, rounded


@pytest.mark.parametrize("median, band", [
    (60.0, [50, 90]),        # 0.65 x 60 = 39, clamped up to the floor
    (400.0, [260, 420]),     # 1.5 x 400 = 600, clamped down to the ceiling
    (150.0, [98, 225]),
])
def test_the_band_is_clamped_to_the_human_range(monkeypatch, median, band):
    monkeypatch.setattr(creator, "_pyin", lambda y, sr: pyin_at(median, y))
    assert creator.f0_band(sine(1.0), SR) == band


def test_an_unvoiced_clip_keeps_the_default_band(monkeypatch):
    """Music or crowd noise measures nothing; the golden test then falls back to
    the wide default rather than to a band that would pass anything."""
    monkeypatch.setattr(creator, "_pyin", lambda y, sr: (np.full(4, np.nan), np.zeros(4, dtype=bool)))
    assert creator.f0_band(sine(1.0), SR) == creator.DEFAULT_F0_BAND


def test_the_band_of_a_real_synthetic_clip_contains_its_pitch():
    """The one test that runs librosa itself, when the box has it: it proves the
    seam feeds pyin what pyin expects, which a stub cannot."""
    pytest.importorskip("librosa")
    lo, hi = creator.f0_band(sine(2.0, hz=150.0), SR)
    assert lo < 150 < hi


# -- the voice id ------------------------------------------------------------


@pytest.mark.parametrize("voice_id", sorted(creator.RESERVED_VOICE_IDS))
def test_reserved_voice_ids_are_refused(data_dir, voice_id):
    with pytest.raises(creator.BadVoiceId):
        creator.check_voice_id(voice_id)


@pytest.mark.parametrize("voice_id", ["", "a", "A-Voice", "má-diakritiku", "x" * 25, "spaced id"])
def test_malformed_voice_ids_are_refused(data_dir, voice_id):
    with pytest.raises(creator.BadVoiceId):
        creator.check_voice_id(voice_id)


def test_a_taken_voice_id_is_refused_and_the_suggestion_steps_around_it(data_dir):
    voices.save_voice(make_voice("krcmar-bruno"))
    with pytest.raises(creator.BadVoiceId):
        creator.check_voice_id("krcmar-bruno")
    assert creator.suggest_voice_id(LABEL) == "krcmar-bruno-2"


def test_update_validates_the_id_before_the_gpu_is_spent(data_dir, no_ffmpeg, flat_pitch):
    draft = started(data_dir, no_ffmpeg, flat_pitch)
    assert creator.update(draft.id, voice_id="bruno").voice_id == "bruno"
    with pytest.raises(creator.BadVoiceId):
        creator.update(draft.id, voice_id="bag")


# -- what the Writer produces is filtered, not believed -----------------------


def stub_writer(monkeypatch, replies: list[str]) -> list[dict]:
    """``llm._ask`` answering from a canned list, one reply per call."""
    calls: list[dict] = []

    def ask(system: str, user: str, timeout: float) -> str:
        calls.append({"system": system, "user": user, "timeout": timeout})
        return replies[min(len(calls) - 1, len(replies) - 1)]

    monkeypatch.setattr(llm, "residency", lambda: "resident")
    monkeypatch.setattr(llm, "_ask", ask)
    return calls


def test_a_token_line_and_an_over_long_line_are_dropped(monkeypatch):
    reply = json.dumps({"lines": ["Vitaj, pocestný.",
                                  "Ten <|emotion:anger|>nie.",       # a delivery is the DM's choice
                                  "a" * (canon.MAX_CHARS + 1),       # canon would refuse it at render
                                  "Sadni si k ohňu."]}, ensure_ascii=False)
    stub_writer(monkeypatch, [reply])
    board_lines = llm.phrases({"sk": "Krčmár."}, ["Vítanie hostí"], lang="sk")
    assert board_lines["Vítanie hostí"] == ["Vitaj, pocestný.", "Sadni si k ohňu."]


def test_duplicates_and_fenced_json_survive_the_parser(monkeypatch):
    reply = '```json\n{"lines": ["Ešte jedno?", "ešte jedno?", "Nalejem."]}\n```'
    stub_writer(monkeypatch, [reply])
    assert llm.phrases({"sk": "K."}, ["Nalej ešte"], lang="sk")["Nalej ešte"] == ["Ešte jedno?", "Nalejem."]


def test_a_thin_category_is_asked_once_more_and_then_reported(monkeypatch, data_dir):
    reply = json.dumps({"lines": ["Dnes už nie.", "Choď domov."]}, ensure_ascii=False)
    calls = stub_writer(monkeypatch, [reply])
    written = llm.phrases({"sk": "K."}, ["Toto ti nenalejem"], lang="sk")
    assert len(calls) == 2                                   # one attempt, then one regeneration
    assert len(written["Toto ti nenalejem"]) == 2
    draft = make_draft(phrases=written)
    assert creator.thin_categories(draft) == ["Toto ti nenalejem"]


def test_a_category_the_brain_never_answered_is_empty_not_fatal(monkeypatch):
    import requests

    monkeypatch.setattr(llm, "residency", lambda: "resident")
    monkeypatch.setattr(llm, "_ask", lambda *a: (_ for _ in ()).throw(requests.Timeout("slow")))
    assert llm.phrases({"sk": "K."}, ["Vítanie hostí"], lang="sk") == {"Vítanie hostí": []}


def test_persona_needs_text_and_at_least_two_categories(monkeypatch):
    stub_writer(monkeypatch, ['{"sk": "", "en": "", "categories": []}'])
    with pytest.raises(llm.WriterFailed):
        llm.persona(LABEL, "Zhovorčivý krčmár.", lang="sk")


def test_persona_keeps_the_model_order_so_the_last_category_is_the_refusal(monkeypatch):
    stub_writer(monkeypatch, [json.dumps({"sk": "Krčmár.", "en": "Innkeeper.",
                                          "categories": CATEGORIES}, ensure_ascii=False)])
    written = llm.persona(LABEL, "Zhovorčivý krčmár.", lang="sk")
    assert written["categories"][-1] == "Toto ti nenalejem"
    assert written["label"] == LABEL


def test_the_dms_own_edits_go_through_the_same_guard(data_dir, no_ffmpeg, flat_pitch):
    draft = started(data_dir, no_ffmpeg, flat_pitch)
    edited = creator.set_phrases(draft.id, {"Vítanie hostí": ["Vitaj.", "Ten <|sfx:sigh|>nie.", "Vitaj."]})
    assert edited.phrases == {"Vítanie hostí": ["Vitaj."]}


# -- commit ------------------------------------------------------------------


@pytest.fixture
def committable(fresh_db, monkeypatch, no_ffmpeg, flat_pitch):
    """A draft one call away from commit, with the GPU and the board's global
    signature table replaced. Returns (draft, order, calibrated, worker)."""
    monkeypatch.setattr(board, "SIGNATURE_CATEGORY",
                        {lang: dict(entries) for lang, entries in board.SIGNATURE_CATEGORY.items()})
    order: list[str] = []
    calibrated: list[list[str]] = []

    def lock_reference(voice_id: str, wav_in: Path, transcript: str) -> Voice:
        order.append("lock")
        v = make_voice(voice_id)
        v.ref_transcript = transcript
        voices.save_voice(v)
        return v

    def calibrate(v: Voice, lines: list[str], n_takes: int = 3) -> Voice:
        order.append("calibrate")
        calibrated.append(list(lines))
        v.gate.update(baseline=0.93, strict=0.89, loose=0.85, calibrated_at="2026-09-04T10:00:00")
        v.master["gain_db"] = -1.5
        voices.save_voice(v)
        return v

    real_add_line = board.add_line

    def add_line(*args, **kw):
        if "import" not in order:
            order.append("import")
        return real_add_line(*args, **kw)

    monkeypatch.setattr(creator.voices, "lock_reference", lock_reference)
    monkeypatch.setattr(creator.voices, "calibrate", calibrate)
    monkeypatch.setattr(board, "add_line", add_line)

    draft = creator.create(sine(20.0), "bruno.wav", LABEL, LANG)
    creator.clip(draft.id, 0.0, 12.0)
    draft = creator.update(draft.id, voice_id="krcmar", transcript="Vitaj, pocestný.")
    draft.persona = {"sk": "Krčmár.", "en": "An innkeeper."}
    draft.categories = list(CATEGORIES)
    draft.phrases = {k: list(v) for k, v in PHRASES.items()}
    draft.status = "described"
    creator._save(draft)

    worker = FakeWorker()

    def submit(job):
        if "queue" not in order:
            order.append("queue")
        return worker.jobs.append(job) or job

    worker.submit = submit
    return draft, order, calibrated, worker


def test_commit_runs_lock_calibrate_import_queue_in_that_order(committable):
    draft, order, calibrated, worker = committable
    events: list[dict] = []
    result = creator.commit(draft.id, worker=worker,
                            on_event=lambda name, data: events.append({"name": name, **data}))

    assert order == ["lock", "calibrate", "import", "queue"]
    assert result["voice_id"] == "krcmar"
    assert result["lines"] == sum(len(v) for v in PHRASES.values())
    assert result["queued"] == result["lines"]
    assert result["gain_db"] == -1.5
    assert result["gate"]["strict"] == 0.89
    assert [e["step"] for e in events] == ["lock", "calibrate", "bank", "queue", "done"]
    assert {e["name"] for e in events} == {"creator.progress"}
    assert [e["pct"] for e in events] == [10, 35, 70, 90, 100]
    assert all(e["draft_id"] == draft.id for e in events)


def test_commit_calibrates_on_the_generated_lines_longest_first(committable):
    draft, order, calibrated, worker = committable
    creator.commit(draft.id, worker=worker)
    lines = calibrated[0]
    assert lines == sorted(lines, key=len, reverse=True)
    assert set(lines) <= {line for group in PHRASES.values() for line in group}
    assert len(lines) <= creator.CALIBRATION_LINES


def test_commit_banks_the_board_and_lights_the_favourites_row(committable):
    draft, order, calibrated, worker = committable
    creator.commit(draft.id, worker=worker)

    view = board.board("krcmar", LANG)
    assert view["categories"] == CATEGORIES
    # The signature category never seeds a favourite; it owns the T tile instead.
    slotted = [line for line in view["lines"] if line["slot"]]
    assert [line["category"] for line in sorted(slotted, key=lambda x: x["slot"])] == CATEGORIES[:-1]
    assert view["ten_nie"] is not None
    assert board.get_line(view["ten_nie"])["category"] == "Toto ti nenalejem"
    assert board.signature_category("krcmar", LANG) == "Toto ti nenalejem"


def test_committed_lines_are_marked_as_the_creators(committable):
    """``source='creator'`` is outside the M1 CHECK list, so the widening
    migration has to have run for any of this to be storable at all."""
    draft, order, calibrated, worker = committable
    creator.commit(draft.id, worker=worker)
    with store.db() as con:
        sources = {row[0] for row in con.execute("SELECT DISTINCT source FROM lines WHERE voice_id='krcmar'")}
    assert sources == {"creator"}


def test_commit_persists_its_result_so_a_reload_still_shows_the_numbers(committable):
    draft, order, calibrated, worker = committable
    result = creator.commit(draft.id, worker=worker)
    reloaded = creator.get(draft.id)
    assert reloaded.status == "done"
    assert reloaded.result == result


def test_commit_writes_the_signature_category_into_voice_yaml(committable, monkeypatch):
    """The fallback path, for a board that has not shipped ``set_signature`` yet:
    the entry has to survive a restart, and ``voice.yaml`` is where it lives."""
    monkeypatch.delattr(board, "set_signature", raising=False)
    draft, order, calibrated, worker = committable
    creator.commit(draft.id, worker=worker)
    data = yaml.safe_load(voices.voice_yaml("krcmar").read_text(encoding="utf-8"))
    assert data["signature_category"] == {LANG: "Toto ti nenalejem"}


def test_commit_refuses_a_reserved_id_before_touching_the_voice(committable):
    """``update`` refuses it at the edge, so this smuggles it straight into the
    stored draft: ``commit`` must check again before it locks anything, because
    a draft written by an older build is not the DM's promise that it is valid."""
    draft, order, calibrated, worker = committable
    stored = creator.get(draft.id)
    stored.voice_id = "bag"
    creator._save(stored)
    with pytest.raises(creator.BadVoiceId):
        creator.commit(draft.id, worker=worker)
    assert order == []
    assert not config.voice_dir("bag").exists()


def test_commit_refuses_a_draft_without_a_transcript(committable):
    draft, order, calibrated, worker = committable
    creator.update(draft.id, transcript="   ")
    with pytest.raises(creator.CreatorError):
        creator.commit(draft.id, worker=worker)
    assert order == []


def test_a_failed_commit_marks_the_draft_and_says_so(committable, monkeypatch):
    draft, order, calibrated, worker = committable
    monkeypatch.setattr(creator.voices, "calibrate",
                        lambda v, lines, n_takes=3: (_ for _ in ()).throw(
                            voices.CalibrationError("no take passed sanity")))
    events: list[dict] = []
    with pytest.raises(voices.CalibrationError):
        creator.commit(draft.id, worker=worker,
                       on_event=lambda name, data: events.append(data))
    assert creator.get(draft.id).status == "failed"
    assert events[-1]["step"] == "failed"


def test_commit_without_a_worker_banks_but_queues_nothing(committable):
    """The contract's bare ``commit(draft_id)``: everything on disk happens, and
    the pre-render simply is not queued because there is nothing to queue onto."""
    draft, order, calibrated, worker = committable
    result = creator.commit(draft.id)
    assert order == ["lock", "calibrate", "import"]
    assert result["queued"] == 0
    assert result["lines"] > 0
