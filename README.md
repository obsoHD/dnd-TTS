# Bag — a talking sentient D&D magic item

A local, offline voice system that gives a physical prop bag a real-time
speaking voice at the table. **Bag** is a sentient, sarcastic, foul-mouthed
magic item bonded to one player since birth (INT 12 / WIS 14 / CHA 18). He
insults enemies, mocks the party, takes credit for everything, and speaks
Slovak.

Runs entirely on the LAN — an inference box (2× RTX 5090) does all the work; a
MacBook Air is a thin push-to-talk client. No cloud.

## Stack (decided)

| part | choice |
|---|---|
| TTS | `bosonai/higgs-audio-v3-tts-4b` — Slovak in its production tier, zero-shot voice cloning, inline emotion/style/sfx control tokens |
| ASR | NVIDIA Parakeet TDT 0.6B v3 (Slovak), faster-whisper large-v3 fallback |
| LLM (Bag's brain) | Qwen3 ~30B MoE, AWQ 4-bit |
| serving | SGLang-Omni → OpenAI-compatible `/v1/audio/speech`, streaming PCM |
| orchestration | plain Python asyncio + FastAPI |
| GPU split | GPU 0 = LLM · GPU 1 = TTS + ASR (no tensor-parallel) |

The first voice is **Mr. Bag**; the design keeps voices swappable so other NPCs
can get their own references later.

## Status

**Milestone 0 — PASSED.** sm_120 (Blackwell) verified, Slovak generates, and
warm latency is **sub-second time-to-first-audio at ~6× real time** on one 5090.
See [`docs/milestone-0.md`](docs/milestone-0.md) for the measured numbers and how
to reproduce.

Next: orchestrator (`/say`, `/respond`, `/stream`), the catchphrase cache, Bag's
persona + item-card mechanics as tool calls, and the MacBook push-to-talk client.

## Layout

```
docker/Dockerfile     derived sglang-omni image (adds the light deps it ships without)
scripts/serve.sh      launch the Higgs TTS server, pinned to GPU 1
scripts/m0_test.py    latency + Slovak sample harness (stdlib only)
docs/milestone-0.md   Milestone 0 report
```

Models, generated audio, and venvs are gitignored — see `.gitignore`.
