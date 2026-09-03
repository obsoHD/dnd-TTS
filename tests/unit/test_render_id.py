"""render_id is the cache key for every served file, so it must be a pure
function of (voice, version, recipe, canonical text, take) and nothing else:
identical across processes and restarts, different whenever any input differs.
The render_line tests drive the orchestration (take counts, seed ladder, retry
round, gate verdict) with the TTS and the speaker gate replaced by fakes: no
GPU, no network."""
from __future__ import annotations

import hashlib
import importlib
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from app import canon, config, render, store, voices
from app.gate import TakeScore
from app.tts_client import Sampler
from app.voices import Voice

REPO = Path(__file__).resolve().parents[2]
SR = 24_000
SHORT = "Ten nie."
LONG = ("Vy nie ste družina, vy ste kolektívna diagnóza a ja som ten, kto vás bude musieť "
        "zase zachraňovať, lebo inak by ste sa nedožili ani rána.")


def tone(seconds: float = 1.0, hz: float = 110.0) -> bytes:
    t = np.arange(int(SR * seconds)) / SR
    return (0.5 * 32767 * np.sin(2 * np.pi * hz * t)).astype(np.int16).tobytes()


def make_voice(**overrides) -> Voice:
    base = dict(id="bag", label="Mr. Bag", lang="sk", version=1, ref_file="ref.wav", ref_sha256="0" * 64,
                ref_transcript="Popravia? Dostane tretí obed.", ref_tts_path="/refs/bag/ref.wav",
                sampler=Sampler(), golden_seed=0, gate=dict(voices.UNCALIBRATED_GATE),
                master={**render.master.DEFAULT, "energy": 65}, banned_tokens=[], persona={}, fillers=[],
                f0_band=[60, 155])
    return Voice(**{**base, **overrides})


def compute(text: str = SHORT, take_no: int = 0, version: int = 1) -> str:
    """Also the entry point for the cross-process check below."""
    return render.render_id(make_voice(version=version), canon.canonicalize(text).text, take_no)


def test_render_id_stable_across_processes():
    code = "from tests.unit.test_render_id import compute; print(compute())"
    other = subprocess.run([sys.executable, "-c", code], cwd=REPO, capture_output=True, text=True, check=True)
    assert other.stdout.strip() == compute()


def test_render_id_changes_with_take_version_text():
    base = compute()
    assert compute(take_no=1) != base
    assert compute(version=2) != base
    assert compute(text="Ten áno.") != base
    assert compute() == base


def test_render_id_is_the_documented_hash():
    text = canon.canonicalize(SHORT).text
    key = f"bag|1|{render.RECIPE_VERSION}|{text}|0"
    assert compute() == hashlib.sha256(key.encode("utf-8")).hexdigest()


def test_recipe_version_is_a_sha256():
    assert len(render.RECIPE_VERSION) == 64
    int(render.RECIPE_VERSION, 16)


@pytest.fixture
def data_dir(tmp_path, monkeypatch):
    """Point the whole app at a temp DATA_DIR by env, the way the container does."""
    monkeypatch.setenv("BAG_DATA", str(tmp_path))
    importlib.reload(config)
    yield tmp_path
    monkeypatch.undo()
    importlib.reload(config)


@pytest.fixture
def locked_voice(data_dir) -> Voice:
    ref = config.voice_dir("bag") / "ref.wav"
    store.write_wav(ref, tone(), SR)
    return make_voice(ref_sha256=voices.sha256_file(ref))


@pytest.fixture
def fake_services(monkeypatch):
    """Replace the two service calls with fakes that record the seed ladder and
    score takes from a table of similarities; the gate verdict follows the real
    contract (``reason == "ok"`` iff the take passed)."""
    calls: list[list[int]] = []
    sims: dict[int, float] = {}

    def synth_many(text, ref_path, ref_text, seeds, sampler, max_new_tokens):
        calls.append(list(seeds))
        return [(seed, tone(), SR) for seed in seeds]

    def select(gate, takes, c, lang, strict, cer_max=0.15):
        scores = []
        for seed, _, _ in takes:
            sim = sims.get(seed, 0.9)
            scores.append(TakeScore(seed=seed, sim=sim, sane=True,
                                    reason="ok" if sim >= strict else f"sim {sim} < strict {strict}"))
        best = max(range(len(scores)), key=lambda i: scores[i].sim)
        return best, scores

    monkeypatch.setattr(render.tts_client, "synth_many", synth_many)
    monkeypatch.setattr(render.gate, "select", select)
    monkeypatch.setattr(render, "_speaker_gate", lambda ref, sha: object())
    return calls, sims


def test_render_line_short_line_two_takes(locked_voice, fake_services):
    calls, _ = fake_services
    r = render.render_line(locked_voice, SHORT)
    assert calls == [[0, 1]]
    assert r.gate == "pass" and r.timings["rounds"] == 1
    assert r.render_id == render.render_id(locked_voice, r.canon.text, 0)
    assert r.recipe_version == render.RECIPE_VERSION and r.voice_version == 1
    assert r.master_version == render.master_version(locked_voice)
    assert r.pcm and r.raw_pcm and r.sr == SR
    assert r.lufs is not None and np.isfinite(r.lufs)


def test_render_line_long_line_three_takes(locked_voice, fake_services):
    calls, _ = fake_services
    assert len(canon.canonicalize(LONG).spoken) > render.LONG_LINE_CHARS
    render.render_line(locked_voice, LONG)
    assert calls == [[0, 1, 2]]


def test_render_line_take_no_moves_the_seed_ladder(locked_voice, fake_services):
    calls, _ = fake_services
    render.render_line(locked_voice, SHORT, take_no=2)
    assert calls == [[16, 17]]


def test_render_line_retries_then_serves_best_failed(locked_voice, fake_services):
    calls, sims = fake_services
    locked_voice.gate["strict"] = 0.8
    sims.update({0: 0.5, 1: 0.6, 3: 0.55, 4: 0.7, 5: 0.65})
    r = render.render_line(locked_voice, SHORT)
    assert calls == [[0, 1], [3, 4, 5]]
    assert r.gate == "failed" and r.timings["rounds"] == 2
    assert r.seed == 4 and r.sim == 0.7
    assert [s.seed for s in r.scores] == [0, 1, 3, 4, 5]


def test_render_line_refuses_a_changed_reference(locked_voice, fake_services):
    store.write_wav(config.voice_dir("bag") / "ref.wav", tone(hz=220.0), SR)
    with pytest.raises(voices.ReferenceMismatch):
        render.render_line(locked_voice, SHORT)
