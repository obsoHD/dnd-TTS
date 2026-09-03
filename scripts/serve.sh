#!/usr/bin/env bash
# Launch the Higgs Audio v3 TTS server, pinned to GPU 1 (leaving GPU 0 for the
# LLM). Model is mounted from a local dir so nothing re-downloads; the refs dir
# holds voice-clone reference clips and MUST be the --allowed-local-media-path.
#
#   ./serve.sh            # serves on host :8010 (container :8000)
#
# Build the image first:  docker build -t bag-tts:local docker/
set -euo pipefail

MODEL_DIR="${MODEL_DIR:-$HOME/dnd-tts/models/higgs-v3}"
REFS_DIR="${REFS_DIR:-$HOME/dnd-tts/refs}"
PORT="${PORT:-8010}"
GPU="${GPU:-1}"

docker rm -f bag-tts 2>/dev/null || true
docker run -d --name bag-tts \
  --gpus all -e CUDA_VISIBLE_DEVICES="$GPU" \
  -e SGLANG_OMNI_HIGGS_REF_CODE_CACHE=1 \
  -p "${PORT}:8000" \
  -v "${MODEL_DIR}:/model:ro" \
  -v "${REFS_DIR}:/refs:ro" \
  bag-tts:local \
  serve --model-path /model --allowed-local-media-path /refs \
        --host 0.0.0.0 --port 8000

echo "bag-tts starting on :${PORT} (GPU ${GPU}). Warm-load takes a few minutes."
echo "watch:   docker logs -f bag-tts"
echo "ready when the log says 'Application startup complete'."
