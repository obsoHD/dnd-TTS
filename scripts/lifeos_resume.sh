#!/usr/bin/env bash
# Undo lifeos_pause.sh: restore the registry priorities (genllm 1, bag 5),
# restart the broker, start the lifeos containers again.
set -euo pipefail
DIR=/opt/generation/apps/orchestrator
START="${LIFEOS_CONTAINERS:-lifeos lifeos-habitat}"

python3 - "$DIR/apps.json" <<'PY'
import json, sys
p = sys.argv[1]; d = json.load(open(p))
for a in d["apps"]:
    if a["id"] == "bag":
        a["priority"] = 5
    elif a["id"] == "genllm":
        a["priority"] = 1
json.dump(d, open(p, "w"), ensure_ascii=False, indent=2)
print("registry:", [(a["id"], a["priority"]) for a in d["apps"]])
PY
cd "$DIR" && docker compose up -d >/dev/null && echo "broker restarted"
for c in $START; do
  if docker ps -a --format '{{.Names}}' | grep -qx "$c"; then docker start "$c" >/dev/null && echo "started $c"; fi
done
