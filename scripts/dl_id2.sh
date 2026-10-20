#!/bin/bash
# ID2.rar 分段并行下载 v2：每段校验大小、不足重下（Zenodo 会中途断连，curl 退出码仍 0）
set -uo pipefail
URL="https://zenodo.org/api/records/15946165/files/ID2.rar/content"
SIZE=1346665013
MD5="fa05ecfdde7661ac16568aa0dde1f4ea"
OUT=/root/autodl-tmp/ID2.rar
PARTS=/root/autodl-tmp/id2_parts
N=32
mkdir -p "$PARTS"
CHUNK=$(( (SIZE + N - 1) / N ))

dl_part() {  # $1=段号；循环到大小正确为止
  local i=$1
  local S=$((i * CHUNK))
  local E=$(( (i+1) * CHUNK - 1 ))
  [ $E -ge $SIZE ] && E=$((SIZE-1))
  [ $S -gt $E ] && return 0
  local want=$((E-S+1))
  for attempt in $(seq 1 10); do
    local cur=0
    [ -f "$PARTS/p$i" ] && cur=$(stat -c %s "$PARTS/p$i")
    [ "$cur" = "$want" ] && return 0
    if [ "$cur" -gt 0 ] && [ "$cur" -lt "$want" ]; then
      # 续传剩余段（zenodo 支持 range；截断文件从头下更稳）
      rm -f "$PARTS/p$i"
    fi
    curl -sL --max-time 900 --speed-time 30 --speed-limit 10240 -r "$S-$E" -o "$PARTS/p$i" "$URL"
    [ -f "$PARTS/p$i" ] && [ "$(stat -c %s "$PARTS/p$i")" = "$want" ] && return 0
    echo "段 $i 第 $attempt 次不完整（$(stat -c %s "$PARTS/p$i" 2>/dev/null || echo 0)/$want），重下"
    sleep 3
  done
  echo "FATAL: 段 $i 十次仍未下满"
  return 1
}

echo "总分 $SIZE 字节，$N 段并行（每段失败重下至多 10 次）..."
pids=()
for i in $(seq 0 $((N-1))); do
  dl_part "$i" &
  pids+=($!)
  if [ $(( (${#pids[@]}) % 8 )) -eq 0 ]; then wait "${pids[@]}"; pids=(); fi
done
[ ${#pids[@]} -gt 0 ] && wait "${pids[@]}"

cat "$PARTS"/p* > "$OUT"
ACTUAL_MD5=$(md5sum "$OUT" | awk '{print $1}')
if [ "$ACTUAL_MD5" = "$MD5" ]; then
  echo "DL-OK md5 校验通过: $OUT ($(stat -c %s "$OUT") 字节)"
  rm -rf "$PARTS"
else
  echo "FATAL: md5 不符 $ACTUAL_MD5 ≠ $MD5"
  exit 1
fi
