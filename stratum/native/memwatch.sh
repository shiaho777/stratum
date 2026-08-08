#!/bin/bash
# Hard memory watchdog. Kills the target the instant free memory drops
# below SAFE_FREE_PCT or its RSS exceeds SAFE_RSS_GB. Protects against
# the Metal-wires-whole-mmap OOM that froze the machine twice.
TARGET_PAT="$1"
SAFE_FREE_PCT=22
SAFE_RSS_GB=18
for i in $(seq 1 120); do
  PID=$(pgrep -f "$TARGET_PAT" | head -1)
  if [ -z "$PID" ]; then echo "[watch] target gone, exiting"; exit 0; fi
  FREE=$(memory_pressure 2>/dev/null | awk '/free percentage/{gsub("%","",$NF);print int($NF)}')
  RSSKB=$(ps -o rss= -p "$PID" 2>/dev/null | tr -d ' ')
  RSSGB=$((RSSKB/1024/1024))
  echo "[watch t=${i}] free=${FREE}% rss=${RSSGB}GB"
  if [ -n "$FREE" ] && [ "$FREE" -lt "$SAFE_FREE_PCT" ]; then
    echo "[watch] FREE ${FREE}% < ${SAFE_FREE_PCT}% -> KILL $PID"; kill -9 "$PID"; exit 1
  fi
  if [ -n "$RSSGB" ] && [ "$RSSGB" -gt "$SAFE_RSS_GB" ]; then
    echo "[watch] RSS ${RSSGB}GB > ${SAFE_RSS_GB}GB -> KILL $PID"; kill -9 "$PID"; exit 1
  fi
  sleep 0.5
done
echo "[watch] timeout"
