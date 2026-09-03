# Milestone 0 — verification (2026-09-03)

Do these before building anything. Both passed.

## 0a — sm_120 (Blackwell) wheels build and run

**PASSED, with a live proof rather than a version check.** The inference box's
ComfyUI already runs `torch 2.8.0+cu128`, whose arch list includes `sm_120`, and
it generates on both RTX 5090s daily (`get_device_capability(0)` → `(12, 0)`).
Driver is CUDA 13.2; Docker 29.7.1 with the NVIDIA CDI runtime. The proven-good
toolchain is **cu128**.

The residual unknown was the *serving layer* — SGLang-Omni's own FlashInfer
attention kernels and CUDA-graph capture, which are the parts most likely to
break on Blackwell. Serving the model settled it: the server loaded the weights,
allocated the KV cache, and **captured CUDA graphs on the FlashInfer backend
without error** on GPU 1. sm_120 is not a problem here.

## 0b — Slovak generation + real latency on a 5090

Served `bosonai/higgs-audio-v3-tts-4b` via SGLang-Omni, generated three Slovak
lines, measured on-box. Published H100 figures are RTF 0.217 / sub-second TTFA;
we match or beat them on a 5090.

**Warm (representative):**

| case | TTFA | RTF | notes |
|---|---|---|---|
| neutral (default voice) | **0.87 s** | 0.165 | plain Slovak line |
| expressive | **0.95 s** | 0.155 | `<|emotion:amusement|>` + `<|sfx:laughter|>` |
| cloned | **0.53 s** | 0.169 | zero-shot from a Slovak reference |

**Cold (first call after load):** TTFA 3.8 s — pure warm-up; the second call
drops to sub-second. Warm the model at boot and keep it resident.

- **Sub-second TTFA: met** (0.53–0.95 s warm).
- **RTF ~0.16: ~6× faster than real time**, well inside the real-time requirement.
- GPU pin held: **GPU 1 at 28.3 GB, GPU 0 untouched** — the LLM's card is free.

## Control tokens the model actually ships

Verified against the tokenizer (the cookbook lists some that do not exist here —
notably there is **no `pitch_low`**, only `pitch_high`, so a deeper/gruffer voice
must come from the reference clip, not a token):

- **emotion:** amusement, anger, confusion, contemplation, enthusiasm, relief,
  sadness, surprise
- **prosody:** expressive_high, pause, long_pause, pitch_high
- **sfx:** laughter, sigh, cough, crying, humming, screaming, sneeze, sniff
- **style:** shouting

Place delivery tokens at the START of the input string.

## Gotchas found (all fixed in `docker/Dockerfile`)

1. The `lmsysorg/sglang-omni:dev` image ships the sglang-omni **source** but not
   its light deps installed → `ModuleNotFoundError: msgpack`.
2. Its bare `pip` targets a different interpreter than the runtime
   `/usr/bin/python` — install with `python -m pip`.
3. A full `pip install -e .` re-resolves torch/sglang and conflicts with the
   image's pins → install `--no-deps` and add only the missing light libs.
4. The CLI is `sglang_omni.cli:app`, invoked as `python -m sglang_omni.cli`, not
   a global `sgl-omni` binary.

## Verdict

Green light. Slovak works, it's fast enough, sm_120 is fine, the expressive
control that motivated the model choice is present. Proceed to the orchestrator —
but the definitive Bag voice needs a real reference recording (a gruff in-
character performance), which the clone path already accepts as
`references:[{audio_path, text}]`.
