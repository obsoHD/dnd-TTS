"""Milestone 0b — measure Higgs Audio v3 on a 5090: Slovak quality + latency.

Streams PCM from the SGLang-Omni /v1/audio/speech endpoint, times the first
audio byte (TTFA) and the whole generation, computes RTF against the produced
audio duration, and writes a WAV for a human to judge. Stdlib only, so it runs
anywhere with no install.

    python3 m0_test.py --host http://127.0.0.1:8000
"""
from __future__ import annotations

import argparse
import json
import time
import urllib.request
import wave

MODEL = "bosonai/higgs-audio-v3-tts-4b"
SR = 24000          # Higgs v3 output rate (model card); overridden by header if present
CH = 1
WIDTH = 2           # s16le

# in-character Slovak test lines. Bag is INT 12 / WIS 14 / CHA 18, sarcastic.
CASES = [
    ("neutral", {
        "input": "Ahoj. Som Vak. A áno — opäť som mal pravdu, ako vždy.",
        "voice": "default"}),
    ("expressive", {
        # delivery tokens go at the START of input per the model card
        "input": "<|emotion:amusement|><|style:mocking|>Naozaj? Toto je tvoj veľký plán? "
                 "<|sfx:laughter|>Cha cha. Vážne, kto tu vždy zachráni ten prekliaty deň?",
        "voice": "default"}),
    ("cloned", {
        "input": "Nie. Nie ten. Tie veci nie sú tvoje, kamoš.",
        "voice": "default",
        "references": [{"audio_path": "/refs/piper_probe.wav",
                        "text": "Ahoj, som Vak a opäť som mal pravdu."}]}),
]


def run_case(host: str, name: str, extra: dict) -> dict:
    body = {"model": MODEL, "stream": True, "response_format": "pcm",
            "temperature": 0.8, "top_k": 50, "max_new_tokens": 2048}
    body.update(extra)
    req = urllib.request.Request(host.rstrip("/") + "/v1/audio/speech",
                                 data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    t0 = time.time()
    ttfa = None
    pcm = bytearray()
    with urllib.request.urlopen(req, timeout=120) as r:
        sr = int(r.headers.get("x-sample-rate") or r.headers.get("sample-rate") or SR)
        for chunk in iter(lambda: r.read(4096), b""):
            if chunk:
                if ttfa is None:
                    ttfa = time.time() - t0
                pcm.extend(chunk)
    gen = time.time() - t0
    dur = len(pcm) / (sr * CH * WIDTH) if pcm else 0.0
    out = f"/out/m0_{name}.wav"
    return {"name": name, "ttfa": ttfa, "gen": gen, "audio_s": dur,
            "rtf": (gen / dur if dur else None), "bytes": len(pcm),
            "sr": sr, "pcm": bytes(pcm), "out": out}


def save_wav(path: str, pcm: bytes, sr: int) -> None:
    with wave.open(path, "wb") as w:
        w.setnchannels(CH)
        w.setsampwidth(WIDTH)
        w.setframerate(sr)
        w.writeframes(pcm)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="http://127.0.0.1:8000")
    ap.add_argument("--outdir", default="/out")
    args = ap.parse_args()

    print(f"{'case':<12}{'TTFA':>9}{'gen':>9}{'audio':>9}{'RTF':>8}   file")
    print("-" * 62)
    for name, extra in CASES:
        try:
            r = run_case(args.host, name, extra)
        except Exception as exc:                             # noqa: BLE001
            print(f"{name:<12}  FAILED: {type(exc).__name__}: {exc}")
            continue
        path = f"{args.outdir}/m0_{name}.wav"
        if r["pcm"]:
            save_wav(path, r["pcm"], r["sr"])
        ttfa = f"{r['ttfa']:.2f}s" if r["ttfa"] else "—"
        rtf = f"{r['rtf']:.3f}" if r["rtf"] else "—"
        print(f"{name:<12}{ttfa:>9}{r['gen']:>8.2f}s{r['audio_s']:>8.2f}s{rtf:>8}   {path}")
    print("\nRTF < 1.0 = faster than real time. TTFA is time to first audio.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
