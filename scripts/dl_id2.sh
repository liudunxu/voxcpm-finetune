#!/bin/bash
# ID2.rar 分段并行下载（Zenodo 单连接 ~27KB/s，32 段并行约 26 分钟）
# 用法：bash /tmp/dl_id2.sh；完成自动 md5 校验
set -euo pipefail
URL="https://zenodo.org/api/records/15946165/files/ID2.rar/content"
SIZE=1346665013
MD5="fa05ecfdde7661ac16568aa0dde1f4ea"
OUT=/root/autodl-tmp/ID2.rar
PARTS=/root/autodl-tmp/id2_parts
N=32
mkdir -p "$PARTS"
CHUNK=$(( (SIZE + N - 1) / N ))
echo "总分 $SIZE 字节，$N 段并行..."
for i in $(seq 0 $((N-1))); do
  S=$((i * CHUNK))
  E=$(( (i+1) * CHUNK - 1 ))
  [ $E -ge $SIZE ] && E=$((SIZE-1))
  [ $S -gt $E ] && break
  [ -s "$PARTS/p$i" ] && [ "$(stat -c %s "$PARTS/p$i")" = "$((E-S+1))" ] && continue
  curl -sL --retry 3 -r "$S-$E" -o "$PARTS/p$i" "$URL" &
  # 每 8 段一波，避免瞬时连接洪峰
  if [ $(( (i+1) % 8 )) -eq 0 ]; then wait; fi
done
wait
# 校验每段大小
for i in $(seq 0 $((N-1))); do
  S=$((i * CHUNK)); E=$(( (i+1) * CHUNK - 1 )); [ $E -ge $SIZE ] && E=$((SIZE-1))
  [ $S -gt $E ] && break
  ACTUAL=$(stat -c %s "$PARTS/p$i" 2>/dev/null || echo 0)
  if [ "$ACTUAL" != "$((E-S+1))" ]; then echo "FATAL: 段 $i 大小 $ACTUAL ≠ $((E-S+1))"; exit 1; fi
done
cat "$PARTS"/p* > "$OUT"
ACTUAL_MD5=$(md5sum "$OUT" | awk '{print $1}')
if [ "$ACTUAL_MD5" = "$MD5" ]; then
  echo "DL-OK md5 校验通过: $OUT ($(stat -c %s $OUT) 字节)"
  rm -rf "$PARTS"
else
  echo "FATAL: md5 不符 $ACTUAL_MD5 ≠ $MD5"
  exit 1
fi
