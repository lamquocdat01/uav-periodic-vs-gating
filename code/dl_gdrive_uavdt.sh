#!/usr/bin/env bash
# Tải file Google Drive công khai qua trang xác nhận virus-scan (link chính thức, không form xin quyền).
# Dùng: dl_gdrive.sh <file_id> <out_name>
set -u
id="$1"; out="$2"
html=$(curl -sL "https://drive.usercontent.google.com/download?id=$id&export=download")
uuid=$(echo "$html" | grep -oE 'name="uuid" value="[^"]*"' | sed -E 's/.*value="([^"]*)"/\1/')
confirm=$(echo "$html" | grep -oE 'name="confirm" value="[^"]*"' | sed -E 's/.*value="([^"]*)"/\1/')
if [ -z "$confirm" ]; then
  if echo "$html" | head -c 200 | grep -qi '<html\|<!DOCTYPE'; then
    echo "BLOCKED $id: $(echo "$html" | sed -e 's/<[^>]*>/ /g' | tr -s ' \n' ' ' | grep -oE 'Quota exceeded|Too many users[^.]*|Sign in|Request access' | head -1)"
    exit 2
  fi
  # file nhỏ: Drive trả thẳng nội dung, không có trang xác nhận
  curl -sL --fail -o "$out" "https://drive.usercontent.google.com/download?id=$id&export=download"
  echo "rc=$? size=$(stat -c %s "$out") (direct)"; exit 0
fi
echo "downloading $out (id=$id) $(date)"
curl -L --fail --retry 5 --retry-delay 10 -C - -o "$out" \
  "https://drive.usercontent.google.com/download?id=$id&export=download&confirm=$confirm&uuid=$uuid" -s -S
rc=$?
echo "rc=$rc size=$(stat -c %s "$out" 2>/dev/null) $(date)"
head -c 4 "$out" | od -An -c
exit $rc
