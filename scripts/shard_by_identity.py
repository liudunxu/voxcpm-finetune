# -*- coding: utf-8 -*-
"""按身份连通分量重切 thai_ser 分片（speaker ∪ session 并查集），并回填已完成的转写。

背景（2026-10-21）：首次按 speaker 切 4 片导致 session（studioNNN 多人同棚）
跨 train/val 泄漏，混合器身份检查拒绝。同一 speaker 可能出现在多 session、
同一 session 有多人 ⇒ 只有连通分量级分组才能同时保证 speaker/session 两片间不相交。

步骤：① 从已加工产物（processed/thai_ser_s*/）把 whisper 转写文本回填进原始清单
（_transcribe_rows 跳过已有 text 的行，重加工不重转写）；② 并查集重切片；
③ 旧分片产物由调用方清理后重跑加工。

用法：uv run python scripts/shard_by_identity.py  （在远端、仓库根下运行）
"""
import json
import sys
from pathlib import Path

RAW = Path("/root/autodl-tmp/voxft_data/raw/thai_ser")
PROCESSED = Path("/root/autodl-tmp/voxft_data/processed")
N_SHARDS = 4


def main():
    rows = [json.loads(l) for l in open(RAW / "manifest.jsonl", encoding="utf-8")]

    # ① 回填转写文本（origin_audio 指向 raw 音频）；加工产物里 text 已被
    # apply_control_prefixes 改成 "(控制)正文"，回填前必须剥掉前缀，恢复裸台词。
    text_by_origin = {}
    for p in sorted(PROCESSED.glob("thai_ser_s*/[tv]*.jsonl")):
        for line in open(p, encoding="utf-8"):
            r = json.loads(line)
            if r.get("text") and r.get("origin_audio"):
                text = r["text"]
                if r.get("control") and text.startswith(f"({r['control']})"):
                    text = text[len(r["control"]) + 2:]
                text_by_origin[r["origin_audio"]] = text
    filled = 0
    for r in rows:
        if not r.get("text") and r["audio"] in text_by_origin:
            r["text"] = text_by_origin[r["audio"]]
            r["transcript_source"] = "whisper-large-v3"
            filled += 1
    print(f"回填转写 {filled}/{len(rows)} 行")

    # ② 并查集：speaker 或 session 相同 ⇒ 同一片
    parent = {}

    def find(x):
        while parent.setdefault(x, x) != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a, b):
        parent[find(a)] = find(b)

    for r in rows:
        keys = []
        if r.get("speaker"):
            keys.append(("spk", r["speaker"]))
        if r.get("session"):
            keys.append(("ses", r["session"]))
        for k in keys[1:]:
            union(keys[0], k)
    comps = {}
    for r in rows:
        anchor = ("spk", r.get("speaker", r["audio"]))
        comps.setdefault(find(anchor), []).append(r)
    groups = sorted(comps.values(), key=len, reverse=True)
    print(f"连通分量 {len(groups)} 个（最大 {len(groups[0])} 行）")

    shards = [[] for _ in range(N_SHARDS)]
    load = [0] * N_SHARDS
    for g in groups:
        i = load.index(min(load))
        shards[i].extend(g)
        load[i] += len(g)
    for i, shard in enumerate(shards):
        with open(RAW / f"shard_{i}.jsonl", "w", encoding="utf-8") as f:
            for r in shard:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        n_ses = len({r.get("session") for r in shard})
        n_spk = len({r.get("speaker") for r in shard})
        n_emo = sum(1 for r in shard if r.get("emotion") not in (None, "neutral"))
        print(f"shard_{i}: {len(shard)} 行, {n_spk} speakers, {n_ses} sessions, 非neutral {n_emo}")
    # 片间不相交自检
    for i in range(N_SHARDS):
        for j in range(i + 1, N_SHARDS):
            si, sj = shards[i], shards[j]
            assert not ({r.get("speaker") for r in si} & {r.get("speaker") for r in sj})
            assert not ({r.get("session") for r in sj} & {r.get("session") for r in si})
    print("片间 speaker/session 不相交自检通过")


if __name__ == "__main__":
    sys.exit(main())
