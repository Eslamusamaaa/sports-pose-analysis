#!/bin/bash
# Resumable, parallel download of SwimXYZ labels (Zenodo 8399376, CC-BY 4.0) with md5 verification.
cd "$(dirname "$0")"
fetch() {
  local name=$1 size=$2 md5=${3#md5:} url=$4
  for attempt in 1 2 3 4 5 6 7 8; do
    if [ -f "$name" ] && [ "$(stat -c %s "$name")" = "$size" ]; then break; fi
    echo "[$(date +%T)] $name attempt $attempt"
    curl -sS -L -C - --retry 5 --retry-delay 10 -o "$name" "$url"
  done
  if [ "$(md5sum "$name" | cut -d' ' -f1)" = "$md5" ]; then echo "[$(date +%T)] OK  $name"; else echo "[$(date +%T)] BAD $name"; fi
}
while read -r name size sum url; do fetch "$name" "$size" "$sum" "$url" & done < manifest.txt
wait
echo DONE
