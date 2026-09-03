"""app.master: pure DSP on synthetic signals. No network, no GPU, no fixtures on disk."""
from __future__ import annotations

import numpy as np
import pytest

from app import master as m

SR = 24000
E65 = m.energy_to_params(65, m.DEFAULT)


def _pcm(x: np.ndarray) -> bytes:
    return (np.clip(x, -1.0, 1.0) * 32767).astype(np.int16).tobytes()


def _voiced(seconds: float, amp: float = 0.3, f0: float = 110.0) -> np.ndarray:
    """Harmonic buzz with syllable-rate amplitude modulation: enough dynamics for
    the compressor to act on and a clear -50 dBFS edge for the trim to find."""
    t = np.arange(int(seconds * SR)) / SR
    y = sum(np.sin(2 * np.pi * f0 * k * t) / k for k in range(1, 9))
    y *= 1.0 + 0.4 * np.sin(2 * np.pi * 3.0 * t)
    return (amp * y / np.max(np.abs(y))).astype(np.float32)


def _padded_voiced(pad_s: float = 0.5, amp: float = 0.3) -> bytes:
    """2 s of voice between two 0.5 s stretches of -75 dBFS noise (well under the
    -50 dBFS trim floor): the trim must keep 40 ms lead and 250 ms tail only."""
    noise = np.random.default_rng(0).normal(0.0, 10 ** (-75 / 20), int(pad_s * SR)).astype(np.float32)
    return _pcm(np.concatenate([noise, _voiced(2.0, amp=amp), noise]))


def _tone(seconds: float, dbfs: float, hz: float = 1000.0) -> bytes:
    t = np.arange(int(seconds * SR)) / SR
    return _pcm((10 ** (dbfs / 20)) * np.sin(2 * np.pi * hz * t))


def _seconds(pcm: bytes) -> float:
    return len(pcm) / 2 / SR


@pytest.fixture(scope="module")
def raw() -> bytes:
    return _padded_voiced()


# ---------------------------------------------------------------- energy knob
def test_energy_65_matches_the_spec_table():
    assert E65["presence_db"] == pytest.approx(2.6)
    assert E65["air_db"] == pytest.approx(1.625)
    assert E65["comp_gr_db"] == pytest.approx(4.55)
    assert E65["comp_threshold_db"] == pytest.approx(-24.0)   # DEFAULT is calibrated at 65
    assert E65["tempo"] == pytest.approx(1.078)


def test_tempo_is_capped_and_compression_tracks_energy():
    e0, e100 = m.energy_to_params(0, m.DEFAULT), m.energy_to_params(100, m.DEFAULT)
    assert e100["tempo"] == pytest.approx(m.DEFAULT["tempo_cap"])      # 1.12 clamped to 1.10
    assert m.energy_to_params(100, {**m.DEFAULT, "tempo_cap": 1.05})["tempo"] == pytest.approx(1.05)
    assert e0["tempo"] == pytest.approx(1.0)
    assert e0["presence_db"] == e0["air_db"] == 0.0
    assert e100["comp_threshold_db"] < -24.0 < e0["comp_threshold_db"]  # more energy, more reduction


def test_energy_to_params_is_pure_and_idempotent():
    base = dict(m.DEFAULT)
    once = m.energy_to_params(30, base)
    assert base == m.DEFAULT
    assert m.energy_to_params(30, once) == once


# ---------------------------------------------------------------- the chain
def test_master_is_byte_identical_on_repeat(raw):
    assert m.master(raw, SR, E65) == m.master(raw, SR, E65)


def test_energy_0_and_100_master_differently(raw):
    e0, e100 = m.energy_to_params(0, m.DEFAULT), m.energy_to_params(100, m.DEFAULT)
    assert m.master(raw, SR, e0) != m.master(raw, SR, e100)
    # still different with tempo pinned: EQ and compression alone must move the bytes
    assert m.master(raw, SR, {**e0, "tempo": 1.0}) != m.master(raw, SR, {**e100, "tempo": 1.0})


def test_tempo_above_cap_is_clamped(raw):
    capped = m.master(raw, SR, {**E65, "tempo": E65["tempo_cap"]})
    assert m.master(raw, SR, {**E65, "tempo": 1.5}) == capped
    plain = m.master(raw, SR, {**E65, "tempo": 1.0})
    assert _seconds(capped) == pytest.approx(_seconds(plain) / E65["tempo_cap"], rel=0.02)


def test_edge_trim_keeps_lead_and_tail_only(raw):
    out = m.master(raw, SR, {**E65, "tempo": 1.0})
    dur = _seconds(out)
    assert 2.0 <= dur <= 2.5
    assert dur == pytest.approx(2.0 + 0.040 + 0.250, abs=0.03)   # voice + lead + tail, not 3 s


def test_unexpanded_block_is_filled_from_its_energy(raw):
    """A voice's master block passed untouched masters like the expanded one at
    its own tempo: the EQ/compression gap is derived, the written tempo honoured."""
    assert m.master(raw, SR, m.DEFAULT) == m.master(raw, SR, {**E65, "tempo": m.DEFAULT["tempo"]})


def test_peak_never_exceeds_minus_1_dbfs():
    hot = _padded_voiced(amp=0.999)
    out = m.master(hot, SR, m.energy_to_params(100, m.DEFAULT))
    assert m.measure(out, SR)["peak_dbfs"] <= -1.0 + 0.01


def test_fades_touch_zero_at_both_edges(raw):
    y = np.frombuffer(m.master(raw, SR, E65), dtype=np.int16)
    assert y[0] == 0 and y[-1] == 0
    assert np.max(np.abs(y)) > 1000


def test_empty_input_gives_empty_output():
    assert m.master(b"", SR, E65) == b""


# ---------------------------------------------------------------- measure
def test_measure_reads_a_minus_12_dbfs_tone():
    r = m.measure(_tone(2.0, -12.0), SR)
    assert r["peak_dbfs"] == pytest.approx(-12.0, abs=0.05)
    assert np.isfinite(r["lufs"]) and -18.0 < r["lufs"] < -12.0
    assert r["dur_s"] == pytest.approx(2.0)


def test_measure_reports_silence_and_short_clips_as_minus_inf():
    assert m.measure(_tone(1.0, -120.0), SR)["lufs"] == float("-inf")
    assert m.measure(_tone(0.2, -12.0), SR)["lufs"] == float("-inf")
