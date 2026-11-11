#!/usr/bin/env bash
# Incremental PhononDB recompute + legacy gate while the MDR download runs (design 2026-11-11 step 1c).
# Every 30 min: recompute newly downloaded archives, re-run the gate (counts only). Stops after the
# downloader has finished and every archive is ingested. Detached: setsid nohup bash <this> &
set -u
cd "$(dirname "$0")/../../.."
STATE_DIR=execution/spectral_store_phonondb
LOG=$STATE_DIR/ingest.log
mkdir -p "$STATE_DIR"
echo "$(date -u +%FT%TZ) start pid $$" >> "$LOG"
while true; do
  python tools/data/spectral_store/ingest_phonon_sources.py phonondb --workers 8 >> "$LOG" 2>&1 || {
    echo "$(date -u +%FT%TZ) ERROR phonondb ingest failed; stopping" >> "$LOG"; exit 1; }
  python tools/data/spectral_store/ingest_phonon_sources.py gate 2>&1 | grep -v "Warning\|val = self" >> "$LOG" || true
  status=$(python -c "import json;print(json.load(open('$STATE_DIR/state.json'))['status'])")
  zips=$(ls data/spectral_store_v1/raw_phonondb/zips/*.zip 2>/dev/null | wc -l)
  echo "$(date -u +%FT%TZ) heartbeat download=$status zips=$zips" >> "$LOG"
  echo "{\"task\": \"phonondb_ingest\", \"pid\": $$, \"download_status\": \"$status\", \"zips\": $zips, \"heartbeat\": \"$(date -u +%FT%TZ)\"}" > "$STATE_DIR/ingest_state.json"
  if [ "$status" != "running" ]; then
    python tools/data/spectral_store/ingest_phonon_sources.py phonondb --workers 8 >> "$LOG" 2>&1
    python tools/data/spectral_store/ingest_phonon_sources.py gate 2>&1 | grep -v "Warning\|val = self" >> "$LOG"
    echo "$(date -u +%FT%TZ) finished" >> "$LOG"
    sed -i 's/"heartbeat"/"status": "finished", "heartbeat"/' "$STATE_DIR/ingest_state.json"
    exit 0
  fi
  sleep 1800
done
