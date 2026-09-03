"""Gruff 'Ted'-style voice test for Bag — clone deepened male references and
push anger/amusement + sigh/laughter, using ONLY control tokens the model
actually ships (no pitch_low — gruffness comes from the reference timbre)."""
import json, time, urllib.request, wave

HOST = "http://127.0.0.1:8010"
MODEL = "/model"
REF_TXT = "Hey, Adam here. Let's create something that feels real, sounds human, and connects every time."
SR, CH, W = 24000, 1, 2

CASES = [
    ("gruff_male", "/refs/male-voice.wav",
     "<|emotion:anger|><|sfx:sigh|> Nie. To si nevezmeš. Tie prekliate veci nie su tvoje, kamos."),
    ("gruff_mid", "/refs/gruff_ref.wav",
     "<|emotion:amusement|><|sfx:laughter|> Cha cha. A kto zas zachranil ten prekliaty den? Ja. Vzdy ja."),
    ("gruff_deep", "/refs/gruff_deep.wav",
     "<|emotion:anger|> Vazne? Toto je tvoj velky plan? Ja som len prekliaty vak a aj tak mam lepsie napady nez ty."),
]

def run(name, ref, text):
    body = {"model": MODEL, "stream": True, "response_format": "pcm",
            "temperature": 0.8, "top_k": 50, "max_new_tokens": 2048,
            "voice": "default", "input": text,
            "references": [{"audio_path": ref, "text": REF_TXT}]}
    req = urllib.request.Request(HOST + "/v1/audio/speech",
        data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
    t0 = time.time(); ttfa = None; pcm = bytearray()
    with urllib.request.urlopen(req, timeout=120) as r:
        sr = int(r.headers.get("x-sample-rate") or SR)
        for c in iter(lambda: r.read(4096), b""):
            if c:
                if ttfa is None: ttfa = time.time() - t0
                pcm.extend(c)
    gen = time.time() - t0; dur = len(pcm)/(sr*CH*W) if pcm else 0
    with wave.open(f"/home/obso/dnd-tts/out/{name}.wav", "wb") as w:
        w.setnchannels(CH); w.setsampwidth(W); w.setframerate(sr); w.writeframes(bytes(pcm))
    print(f"{name:<12} TTFA {ttfa:.2f}s  audio {dur:.1f}s  RTF {gen/dur:.3f}")

for n, ref, t in CASES:
    try: run(n, ref, t)
    except Exception as e: print(f"{n}: FAILED {type(e).__name__}: {e}")
