"""Clone a voice and make Bag say a line. This is how you audition a real voice.

    python3 clone.py --ref /refs/my_bag.wav \
        --ref-text "the exact words spoken in my_bag.wav" \
        --line "<|emotion:anger|> Nie. To si nevezmes, kamos."

WHY A TRANSCRIPT MATTERS. Testing showed the clone is faithful when the
reference is a real, natural recording AND its transcript is correct — a Slovak
Piper reference at 202 Hz cloned to exactly 202 Hz. But borrowed clips with a
guessed transcript flip gender randomly under sampling (a 121 Hz male reference
came out anywhere from 128 Hz to 296 Hz). So: record yourself, and give the
exact words. A deep, gruff performance clones to a deep, gruff voice.

RECORDING TIPS for Mr. Bag:
  * 20-30 seconds, performed in character (gruff, sarcastic, the works).
  * Quiet room, one voice, no music.
  * Write down exactly what you said and pass it as --ref-text.
  * Drop the WAV in ~/dnd-tts/refs/ and point --ref at /refs/<name>.wav.

Lower --temp (0.4-0.6) makes delivery more faithful to the reference; higher is
more expressive but drifts more.
"""
import argparse, json, urllib.request, wave


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="http://127.0.0.1:8010")
    ap.add_argument("--ref", required=True, help="container path, e.g. /refs/my_bag.wav")
    ap.add_argument("--ref-text", required=True, help="exact transcript of the reference")
    ap.add_argument("--line", required=True, help="what Bag should say (may start with control tokens)")
    ap.add_argument("--out", default="/home/obso/dnd-tts/out/clone.wav")
    ap.add_argument("--temp", type=float, default=0.5)
    ap.add_argument("--max-tokens", type=int, default=1024)
    a = ap.parse_args()

    body = {"model": "/model", "stream": True, "response_format": "pcm",
            "temperature": a.temp, "top_k": 40, "max_new_tokens": a.max_tokens,
            "voice": "default", "input": a.line,
            "references": [{"audio_path": a.ref, "text": a.ref_text}]}
    req = urllib.request.Request(a.host + "/v1/audio/speech",
                                 data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    pcm = bytearray()
    with urllib.request.urlopen(req, timeout=120) as r:
        sr = int(r.headers.get("x-sample-rate") or 24000)
        for c in iter(lambda: r.read(4096), b""):
            pcm.extend(c)
    with wave.open(a.out, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sr)
        w.writeframes(bytes(pcm))
    print(f"wrote {a.out}  ({len(pcm)//(sr*2)}s @ {sr}Hz)")


if __name__ == "__main__":
    main()
