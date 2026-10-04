#!/usr/bin/env bash
# Thử lại link Google Drive chính thức của VisDrone2019-VID (quota Drive reset theo thời gian).
# Không dùng mirror, không form. Mỗi 20 phút, tối đa 12 lần.
CODE="$(cd "$(dirname "$0")" && pwd)"
DEST="${THS_DATASETS:-$THS_DATASETS}/VisDrone2019-VID/_zips"  # luật dataset 28-09-2026: zip ngoài Google Drive
mkdir -p "$DEST" && cd "$DEST" || exit 1
S="$CODE/dl_gdrive.sh"
declare -A IDS=( [VisDrone2019-VID-val.zip]=1xuG7Z3IhVfGGKMe3Yj6RnrFHqo_d2a1B
                 [VisDrone2019-VID-test-dev.zip]=1-BEq--FcjshTF1UwUabby_LHhYj41os5
                 [VisDrone2019-VID-train.zip]=1NSNapZQHar22OYzQYuXCugA3QlMndzvw )
for attempt in $(seq 1 12); do
  pending=0
  for f in VisDrone2019-VID-val.zip VisDrone2019-VID-test-dev.zip VisDrone2019-VID-train.zip; do
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
