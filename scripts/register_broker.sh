#!/usr/bin/env bash
# Register Bag with the lifeos VRAM broker ("Mission Control", :8090) so it is
# not evicted when other apps ask for VRAM.
#
# Two changes to /opt/generation/apps/orchestrator (backed up first):
#   1) apps.json — add Bag at priority 5 (just below the assistant). The broker
#      releases apps HIGHEST-priority-number first, and only ones that expose a
#      release endpoint; Bag has none, so like the assistant it is never asked
#      to release. It also shows up on the Mission Control dashboard.
#   2) docker-compose.yml — add Bag's LLM to OLLAMA_KEEP. This is the real fix:
#      when the assistant frees VRAM the broker unloads idle Ollama models that
#      are NOT in OLLAMA_KEEP — which was silently killing Bag's brain between
#      turns. Keeping it listed pins it (in RAM; it's CPU-loaded here, so this
#      does not compete for the assistant's VRAM).
#
# Idempotent. Recreates ONLY the genctl-broker container.
set -euo pipefail
DIR=/opt/generation/apps/orchestrator
MODEL="${1:-llama3.1:8b-instruct-q8_0}"
ts=$(date +%s)

cp "$DIR/apps.json"          "$DIR/apps.json.bak.$ts"
cp "$DIR/docker-compose.yml" "$DIR/docker-compose.yml.bak.$ts"
echo "backed up -> *.bak.$ts"

python3 - "$DIR/apps.json" <<'PY'
import json, sys
p = sys.argv[1]; d = json.load(open(p))
d["apps"] = [a for a in d["apps"] if a.get("id") != "bag"]
bag = {"id": "bag", "name": "Bag — D&D voice (TTS + orchestrator)", "priority": 5,
       "status_url": "http://localhost:8020/healthz",
       "user_notice": "Bag hlas beží — chránený počas hry."}
d["apps"] = ([d["apps"][0], bag] + d["apps"][1:]) if d["apps"] and \
            d["apps"][0].get("id") == "genllm" else [bag] + d["apps"]
json.dump(d, open(p, "w"), ensure_ascii=False, indent=2)
print("registry:", [(a["id"], a["priority"]) for a in d["apps"]])
PY

python3 - "$DIR/docker-compose.yml" "$MODEL" <<'PY'
import sys, re
p, model = sys.argv[1], sys.argv[2]
s = open(p).read()
default = "qwen3:14b,qwen2.5vl:7b"
if "OLLAMA_KEEP" in s:
    def repl(m):
        vals = [v.strip() for v in m.group(1).split(",") if v.strip()]
        if model not in vals:
            vals.append(model)
        return "- OLLAMA_KEEP=" + ",".join(vals)
    s = re.sub(r"- OLLAMA_KEEP=([^\n]*)", repl, s)
else:
    s = re.sub(r"(- OLLAMA_URL=[^\n]*\n)",
               r"\1      - OLLAMA_KEEP=" + default + "," + model + "\n", s)
open(p, "w").write(s)
print("OLLAMA_KEEP now includes", model)
PY

cd "$DIR" && docker compose up -d
sleep 2
echo "=== broker sees ==="
curl -s http://localhost:8090/broker/state | \
  python3 -c "import sys,json;d=json.load(sys.stdin);print('apps:',[a['id'] for a in d['apps']]);print('ollama_keep protects Bag: check compose')"
echo "done — Bag registered, brain pinned."
