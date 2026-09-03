#!/bin/sh
# Seed the phrase bank into the data volume once. WHY here and not in the app:
# /data is the operator's copy (editable, backed up); the image only carries the
# shipped default, and overwriting it on every boot would discard their edits.
set -e
if [ ! -f /data/phrases.json ] && [ -f /app/seed/phrases.json ]; then
  cp /app/seed/phrases.json /data/phrases.json
  echo "seeded /data/phrases.json from the image"
fi
exec "$@"
