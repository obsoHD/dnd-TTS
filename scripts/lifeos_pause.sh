#!/usr/bin/env bash
# Pause lifeos's hold on GPU 0 so Bag's brain stays resident (reversible with
# lifeos_resume.sh). Does three things:
#   1) stops the lifeos assistant containers (they reload hermes4-70b on use)
#   2) unloads hermes4-70b from ollama right now
#   3) puts Bag above genllm in the broker registry (backup kept) and restarts
#      only the broker container
set -euo pipefail
DIR=/opt/generation/apps/orchestrator
STOP="${LIFEOS_CONTAINERS:-lifeos lifeos-habitat}"

for c in $STOP; do
  if docker ps --format '{{.Names}}' | grep -qx "$c"; then docker stop "$c" >/dev/null && echo "stopped $c"; fi
done
curl -s -X POST http://127.0.0.1:11434/api/generate -d '{"model":"hermes4-70b-lifeos:latest","keep_alive":0,"prompt":""}' >/dev/null || true
echo "unloaded hermes4-70b"

ts=$(date +%s)
cp "$DIR/apps.json" "$DIR/apps.json.bak.$ts"
python3 - "$DIR/apps.json" <<'PY'
import json, sys
p = sys.argv[1]; d = json.load(open(p))
for a in d["apps"]:
    if a["id"] == "bag":
        a["priority"] = 1
    elif a["id"] == "genllm":
        a["priority"] = 50          # paused: released before Bag, never protected
json.dump(d, open(p, "w"), ensure_ascii=False, indent=2)
print("registry:", [(a["id"], a["priority"]) for a in d["apps"]])
PY
cd "$DIR" && docker compose up -d >/dev/null && echo "broker restarted"
curl -s http://127.0.0.1:11434/api/ps | python3 -c "import sys,json;print('resident:',[m['name'] for m in json.load(sys.stdin).get('models',[])] or 'none')"
