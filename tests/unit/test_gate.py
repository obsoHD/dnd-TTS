"""Gate decisions on synthetic fixtures: no GPU, no network, no resemblyzer.

The STT is stubbed at ``requests.post``; speaker similarity comes from a fake
gate keyed by take bytes, so the selection logic is tested in isolation from
the models it normally drives.
"""
from __future__ import annotations

import os
import sys
import types
import wave
from dataclasses import dataclass
from io import BytesIO
from pathlib import Path

import numpy as np
import pytest
import requests

from app import gate, stt, store

SR = 24_000
SPOKEN = "Ten nie."          # 2 syllables


def _sine(seconds: float, dbfs: float = -6.0, hz: float = 220.0) -> bytes:
    t = np.arange(int(SR * seconds)) / SR
    amplitude = 10 ** (dbfs / 20)
    return (np.sin(2 * np.pi * hz * t) * amplitude * 32767).astype(np.int16).tobytes()


def _silence(seconds: float) -> bytes:
    return bytes(2 * int(SR * seconds))


def _pcm_of_wav(blob: bytes) -> bytes:
    with wave.open(BytesIO(blob), "rb") as w:
        return w.readframes(w.getnframes())


@dataclass
class _Canon:
    spoken: str = SPOKEN
    syllables: int = 2


class _Response:
    def __init__(self, text: str) -> None:
        self._text = text

    def raise_for_status(self) -> None:
        return None

    def json(self) -> dict:
        return {"text": self._text}


class FakeGate:
    """similarity() looked up by take bytes, so tests state each take's SIM outright."""

    def __init__(self, sims: dict[bytes, float]) -> None:
        self.sims = sims

    def similarity(self, pcm: bytes, sr: int) -> float:
        return self.sims[pcm]


def _takes(sim_by_seed: dict[int, float]) -> tuple[list[tuple[int, bytes, int]], FakeGate]:
    """One sane 1 s take per seed, each at its own pitch so the bytes differ."""
    takes, sims = [], {}
    for k, (seed, sim) in enumerate(sim_by_seed.items()):
        pcm = _sine(1.0, hz=220.0 + 110.0 * k)
        takes.append((seed, pcm, SR))
        sims[pcm] = sim
    return takes, FakeGate(sims)


def _stub_stt(monkeypatch: pytest.MonkeyPatch, replies: list) -> list[dict]:
    """requests.post stub: replies are consumed in call order (str -> transcript,
    Exception -> raised); returns the recorded call kwargs."""
    calls: list[dict] = []

    def post(url: str, **kwargs) -> _Response:
        calls.append({"url": url, **kwargs})
        reply = replies[min(len(calls), len(replies)) - 1]
        if isinstance(reply, Exception):
            raise reply
        return _Response(reply)

    # The gate posts through app.stt, the one whisper client it shares with the
    # Voice Creator, so the stub goes there and still proves the gate's own
    # timeout and payload reach the endpoint.
    monkeypatch.setattr(stt.requests, "post", post)
    monkeypatch.setattr(stt, "_url", lambda: "https://stt.test/stt")
    return calls


# ------------------------------------------------------------------ sanity
class TestSanity:
    def test_one_second_sine_passes(self):
        assert gate.sanity(_sine(1.0), SR, syllables=4) == (True, "ok")

    def test_silence_fails_on_peak(self):
        ok, reason = gate.sanity(_silence(1.0), SR, syllables=4)
        assert not ok
        assert reason.startswith("peak")

    def test_quiet_take_fails_on_peak(self):
        ok, reason = gate.sanity(_sine(1.0, dbfs=-36.0), SR, syllables=4)
        assert not ok
        assert reason.startswith("peak")

    def test_two_second_internal_gap_fails(self):
        pcm = _sine(0.5) + _silence(2.0) + _sine(0.5)           # 3 s, within [0.72, 4.7]
        ok, reason = gate.sanity(pcm, SR, syllables=6)
        assert not ok
        assert reason.startswith("internal silence 2.00")

    def test_short_internal_gap_is_a_pause_not_a_defect(self):
        pcm = _sine(0.5) + _silence(1.0) + _sine(0.5)
        assert gate.sanity(pcm, SR, syllables=6) == (True, "ok")

    def test_edge_silence_is_ignored(self):
        pcm = _silence(2.0) + _sine(1.0) + _silence(2.0)         # 5 s, within [0.96, 5.6]
        assert gate.sanity(pcm, SR, syllables=8) == (True, "ok")

    def test_too_short_for_syllables_fails(self):
        ok, reason = gate.sanity(_sine(1.0), SR, syllables=20)   # needs >= 2.4 s
        assert not ok
        assert reason.startswith("duration")

    def test_too_long_for_syllables_fails(self):
        ok, reason = gate.sanity(_sine(4.0), SR, syllables=2)    # allows <= 2.9 s
        assert not ok
        assert reason.startswith("duration")

    def test_empty_take_fails_on_peak(self):
        assert gate.sanity(b"", SR, syllables=2)[1].startswith("peak")

    def test_dangling_byte_is_tolerated(self):
        assert gate.sanity(_sine(1.0) + b"\x01", SR, syllables=4) == (True, "ok")


# ------------------------------------------------------------ normalisation
def test_norm_for_cer_folds_tokens_diacritics_punctuation_and_repeats():
    text = "Ten <|prosody:pause|> nie… Bráácho, ľúbim ťa!"
    assert gate._norm_for_cer(text) == "ten nie bracho lubim ta"


def test_word_ratio_counts_normalised_words():
    assert gate.word_ratio("Ten nie.", "ten nie") == 1.0
    assert gate.word_ratio("Ten nie, brácho.", "ten") == pytest.approx(1 / 3)
    assert gate.word_ratio("", "anything") == 1.0


# -------------------------------------------------------------------- cer
def test_cer_returns_none_when_requests_raises(monkeypatch):
    _stub_stt(monkeypatch, [requests.ConnectionError("stt down")])
    assert gate.cer(_sine(1.0), SR, SPOKEN, "sk") is None


def test_cer_returns_none_on_timeout(monkeypatch):
    _stub_stt(monkeypatch, [requests.Timeout("slow")])
    assert gate.cer(_sine(1.0), SR, SPOKEN, "sk") is None


def test_cer_measures_normalised_character_distance(monkeypatch):
    _stub_stt(monkeypatch, ["ten nie"])
    assert gate.cer(_sine(1.0), SR, "Ten nie.", "sk") == 0.0

    _stub_stt(monkeypatch, ["ten"])
    assert gate.cer(_sine(1.0), SR, "Ten nie.", "sk") == pytest.approx(1 - 2 * 3 / (7 + 3), abs=1e-3)


def test_cer_posts_wav_multipart_within_table_budget(monkeypatch):
    calls = _stub_stt(monkeypatch, ["ten nie"])
    pcm = _sine(1.0)
    gate.cer(pcm, SR, SPOKEN, "sk")
    (call,) = calls
    assert call["url"] == "https://stt.test/stt"
    assert call["verify"] is False
    assert call["timeout"] == 2.0
    assert call["data"] == {"lang": "sk"}
    name, blob, mime = call["files"]["audio"]
    assert name.endswith(".wav") and mime == "audio/wav"
    assert _pcm_of_wav(blob) == pcm


# ----------------------------------------------------------------- select
def test_select_picks_best_sim_and_checks_cer_on_it_only(monkeypatch):
    takes, fake = _takes({11: 0.80, 12: 0.93, 13: 0.88})
    calls = _stub_stt(monkeypatch, ["ten nie"])

    idx, scores = gate.select(fake, takes, _Canon(), "sk", strict=0.5)

    assert idx == 1
    assert [s.seed for s in scores] == [11, 12, 13]
    assert scores[1] == gate.TakeScore(seed=12, sim=0.93, sane=True, reason="ok", cer=0.0, verified=True)
    assert [(s.cer, s.verified) for s in (scores[0], scores[2])] == [(None, False), (None, False)]
    assert len(calls) == 1
    assert _pcm_of_wav(calls[0]["files"]["audio"][1]) == takes[1][1]


def test_select_skips_insane_takes_even_with_higher_sim(monkeypatch):
    takes, fake = _takes({11: 0.80, 12: 0.90})
    silent = _silence(1.0)
    takes.append((13, silent, SR))
    fake.sims[silent] = 0.99
    _stub_stt(monkeypatch, ["ten nie"])

    idx, scores = gate.select(fake, takes, _Canon(), "sk", strict=0.5)

    assert idx == 1
    assert scores[2].sane is False and scores[2].reason.startswith("peak")
    assert scores[2].sim == 0.99          # SIM is still measured for calibration


def test_select_moves_to_next_by_sim_when_cer_fails(monkeypatch):
    takes, fake = _takes({11: 0.80, 12: 0.93, 13: 0.88})
    calls = _stub_stt(monkeypatch, ["hmm hmm hmm hmm", "ten nie"])

    idx, scores = gate.select(fake, takes, _Canon(), "sk", strict=0.5)

    assert idx == 2
    assert scores[1].cer > 0.15 and not scores[1].verified and scores[1].reason.startswith("cer")
    assert scores[2].cer == 0.0 and scores[2].verified and scores[2].reason == "ok"
    assert scores[0].cer is None
    assert [_pcm_of_wav(c["files"]["audio"][1]) for c in calls] == [takes[1][1], takes[2][1]]


def test_select_never_swaps_seed_when_stt_is_down(monkeypatch):
    takes, fake = _takes({11: 0.80, 12: 0.93, 13: 0.88})
    calls = _stub_stt(monkeypatch, [requests.ConnectionError("stt down")])

    idx, scores = gate.select(fake, takes, _Canon(), "sk", strict=0.5)

    assert idx == 1
    assert scores[1].cer is None and scores[1].verified is False and scores[1].reason == "ok"
    assert len(calls) == 1                # no second seed is tried because of a missing verdict


def test_select_below_strict_returns_best_sim_without_stt(monkeypatch):
    takes, fake = _takes({11: 0.80, 12: 0.93, 13: 0.88})
    calls = _stub_stt(monkeypatch, ["ten nie"])

    idx, scores = gate.select(fake, takes, _Canon(), "sk", strict=0.95)

    assert idx == 1
    assert scores[1].reason == "sim 0.930 < strict 0.950" and not scores[1].verified
    assert calls == []


def test_select_all_cer_fail_returns_best_sim_gate_failed(monkeypatch):
    takes, fake = _takes({11: 0.80, 12: 0.93, 13: 0.88})
    calls = _stub_stt(monkeypatch, ["hmm hmm hmm hmm"])

    idx, scores = gate.select(fake, takes, _Canon(), "sk", strict=0.5)

    assert idx == 1
    assert all(s.reason.startswith("cer") and not s.verified for s in scores)
    assert len(calls) == 3


def test_select_prefers_a_sane_take_over_an_insane_one_when_nothing_passes(monkeypatch):
    takes, fake = _takes({11: 0.60})
    silent = _silence(1.0)
    takes.append((12, silent, SR))
    fake.sims[silent] = 0.99
    _stub_stt(monkeypatch, ["ten nie"])

    idx, _ = gate.select(fake, takes, _Canon(), "sk", strict=0.9)

    assert idx == 0


def test_select_empty_takes():
    assert gate.select(FakeGate({}), [], _Canon(), "sk", strict=0.5) == (None, [])


# ------------------------------------------------------------ SpeakerGate
@pytest.fixture
def fake_resemblyzer(monkeypatch) -> dict[int, list[float]]:
    """A resemblyzer stand-in whose embeddings are looked up by preprocessed
    length: a path preprocesses to 100 samples, an array to itself."""
    embeddings: dict[int, list[float]] = {}

    class VoiceEncoder:
        def __init__(self, device=None, **kwargs) -> None:
            self.device = device

        def embed_utterance(self, wav: np.ndarray) -> np.ndarray:
            return np.asarray(embeddings[wav.size], dtype=np.float32)

    def preprocess_wav(source, source_sr=None) -> np.ndarray:
        if isinstance(source, (str, Path)):
            return np.zeros(100, dtype=np.float32)
        return np.asarray(source, dtype=np.float32)

    module = types.ModuleType("resemblyzer")
    module.VoiceEncoder = VoiceEncoder
    module.preprocess_wav = preprocess_wav
    monkeypatch.setitem(sys.modules, "resemblyzer", module)
    monkeypatch.setitem(sys.modules, "webrtcvad", types.ModuleType("webrtcvad"))
    return embeddings


def test_speaker_gate_is_lazy_about_resemblyzer(monkeypatch, tmp_path):
    monkeypatch.setitem(sys.modules, "resemblyzer", None)
    with pytest.raises(ImportError, match="resemblyzer"):
        gate.SpeakerGate(tmp_path / "ref.wav")


def test_speaker_gate_caches_reference_embedding_and_scores_cosine(tmp_path, fake_resemblyzer):
    ref = tmp_path / "ref.wav"
    ref.write_bytes(store.wav_bytes(_sine(1.0), SR))
    fake_resemblyzer[100] = [1.0, 0.0]            # the reference
    fake_resemblyzer[SR] = [1.0, 0.0]             # a 1 s take: same speaker
    fake_resemblyzer[SR // 2] = [0.0, 1.0]        # a 0.5 s take: someone else

    g = gate.SpeakerGate(ref)

    cache = tmp_path / "ref.emb.npy"
    assert np.allclose(np.load(cache), [1.0, 0.0])
    assert g.similarity(_sine(1.0), SR) == pytest.approx(1.0)
    assert g.similarity(_sine(0.5), SR) == pytest.approx(0.0)

    fake_resemblyzer[100] = [0.0, 1.0]            # a fresh embedding would now differ...
    assert gate.SpeakerGate(ref).similarity(_sine(1.0), SR) == pytest.approx(1.0)   # ...the cache is used

    stale = cache.stat().st_mtime - 10
    os.utime(cache, (stale, stale))               # a re-locked ref.wav is newer than its cache
    assert gate.SpeakerGate(ref).similarity(_sine(1.0), SR) == pytest.approx(0.0)  # recomputed


def test_speech_seconds_excludes_pauses():
    pcm = _sine(1.0) + _silence(0.8) + _sine(0.5)
    assert gate.speech_seconds(pcm, SR) == pytest.approx(1.5, abs=0.05)
    assert gate.speech_seconds(b"", SR) == 0.0
