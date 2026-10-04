#!/usr/bin/env bash
# Thử lại link Google Drive chính thức của UAVDT UAV-benchmark-M (quota Drive reset theo thời gian).
# Không dùng mirror, không form. Mỗi 20 phút, tối đa 12 lần.
CODE="$(cd "$(dirname "$0")" && pwd)"
DEST="${THS_DATASETS:-$THS_DATASETS}/UAVDT/_zips"  # luật dataset 28-09-2026: zip ngoài Google Drive
mkdir -p "$DEST" && cd "$DEST" || exit 1
S="$CODE/dl_gdrive_uavdt.sh"
declare -A IDS=( [UAV-benchmark-M.zip]=1m8KA6oPIRK_Iwt9TYFquC87vBc_8wRVc )
for attempt in $(seq 1 12); do
  pending=0
  for f in UAV-benchmark-M.zip; do
    if [ -f "$f" ] && [ "$(head -c 2 "$f")" = "PK" ] && unzip -tq "$f" >/dev/null 2>&1; then continue; fi
    rm -f "$f"
    echo "[attempt $attempt] $f $(date)"
    bash "$S" "${IDS[$f]}" "$f"
    if [ ! -f "$f" ] || [ "$(head -c 2 "$f")" != "PK" ]; then rm -f "$f"; pending=1; fi
  done
  [ $pending -eq 0 ] && { echo "ALL_OK $(date)"; exit 0; }
  sleep 1200
done
echo "GAVE_UP $(date)"; exit 3
