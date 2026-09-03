#!/usr/bin/env bash
# Build + run the Bag orchestrator. Host networking so it can reach the TTS
# container on 127.0.0.1:8010 and expose its own UI on :8020.
set -euo pipefail
cd "$(dirname "$0")/.."

docker build -f docker/orchestrator.Dockerfile -t bag-orchestrator .

docker rm -f bag-orch 2>/dev/null || true
mkdir -p "$HOME/dnd-tts/cache"
docker run -d --name bag-orch \
  --network host \
  --restart unless-stopped \
  -v "$HOME/dnd-tts/refs:/refs:ro" \
  -v "$HOME/dnd-tts/cache:/cache" \
  -e BAG_TTS_URL="http://127.0.0.1:8010" \
  -e BAG_REF="/refs/bag_ref.wav" \
  bag-orchestrator

echo "Bag UI → http://$(hostname -I | awk '{print $1}'):8020/"
