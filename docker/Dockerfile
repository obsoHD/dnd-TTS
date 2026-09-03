# Higgs Audio v3 TTS server, built on SGLang-Omni.
#
# The upstream lmsysorg/sglang-omni:dev image ships the sglang-omni SOURCE at
# /sgl-workspace/sglang-omni but NOT its (lighter) Python deps installed, so the
# `sgl-omni` CLI fails with ModuleNotFoundError: msgpack. Two gotchas fixed here:
#
#   1. Install with `python -m pip`, not the bare `pip` — the image's `pip`
#      points at a DIFFERENT interpreter than the runtime `/usr/bin/python`.
#   2. Install the package with --no-deps: a full `-e .` re-resolves torch and
#      sglang and conflicts with the image's pins (ResolutionImpossible). The
#      heavy deps are already present; only the light ones are missing.
#
# If the server later reports another missing module, add it to the pip line.
FROM lmsysorg/sglang-omni:dev

RUN cd /sgl-workspace/sglang-omni && python -m pip install --no-deps -e . && \
    python -m pip install msgpack pyzmq

WORKDIR /sgl-workspace/sglang-omni

# the CLI is a console-script entry (sglang_omni.cli:app), invoked as a module
# so it resolves from the workspace without a global install.
ENTRYPOINT ["python", "-m", "sglang_omni.cli"]
