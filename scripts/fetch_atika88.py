# -*- coding: utf-8 -*-
"""抓取 Atika88 印尼语 ASR-11 类数据集的子集（CC-BY-4.0，真人，含说话人标签）。

用法（远端）：python -X utf8 scripts/fetch_atika88.py
- 下 viewer/metadata/train.parquet（行级元数据：transcript/speaker/is_synthetic）
- 下 3 个句型 tar（Declarative/Interrogative/Exclamatory，覆盖陈述/疑问/感叹的韵律面）
- 只解出「train split 且 is_synthetic=False」的行（上游混了 132 条 Edge-TTS 合成修复行）
- 校验 tar 的 sha256（上游 checksums/audio_shards.sha256）
- 产物：data/raw/atika88_id/{manifest.jsonl, audio/}，加工走 voxft.data.pipeline --source atika88_id
"""
import hashlib
import json
import subprocess
import sys
import tarfile
from pathlib import Path

import pandas as pd

REPO = "Atika88/Indonesian-ASR-11-Class-Dataset"
# 2026-10-20 实测：hf-mirror 对该仓大文件 ~100KB/s 太慢，改直连 huggingface.co +
# network_turbo 代理（用户指示）；3 个句型包并行下载。
BASE = f"https://huggingface.co/datasets/{REPO}/resolve/main"
CATEGORIES = ["Declarative", "Interrogative", "Exclamatory"]
OUT = Path("/root/autodl-tmp/voxft_data/raw/atika88_id")
WORK = Path("/root/autodl-tmp/atika88_tars")


def fetch(path: str, dst: Path) -> Path:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists() and dst.stat().st_size > 0:
        print(f"已有 {dst.name}，跳过")
        return dst
    subprocess.run(["curl", "-sL", "--fail", "--retry", "3", "-o", str(dst),
                    f"{BASE}/{path}"], check=True)
    return dst


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    WORK.mkdir(parents=True, exist_ok=True)
    if (OUT / "manifest.jsonl").exists():
        raise SystemExit(f"{OUT}/manifest.jsonl 已存在，不覆盖旧产物")

    meta = pd.read_parquet(fetch("viewer/metadata/train.parquet", WORK / "train.parquet"))
    real = meta[~meta["is_synthetic"] & meta["category"].isin(CATEGORIES)]
    print(f"元数据 {len(meta)} 行；过滤合成+三类句型后 {len(real)} 行 "
          f"（{real['duration_sec'].sum()/3600:.1f}h）")

    checks = fetch("checksums/audio_shards.sha256", WORK / "audio_shards.sha256")
    expected = {}
    for line in checks.read_text().splitlines():
        parts = line.split()
        if len(parts) == 2:
            expected[parts[1].lstrip("*")] = parts[0]

    # 3 个句型包并行下（各自一条连接，聚合约 3 倍）
    tar_procs = []
    for cat in CATEGORIES:
        tar = WORK / f"{cat}.tar"
        if tar.exists() and tar.stat().st_size > 0:
            continue
        tar_procs.append((cat, subprocess.Popen(
            ["curl", "-sL", "--fail", "--retry", "3", "-o", str(tar),
             f"{BASE}/data/audio_shards/by_category/{cat}.tar"])))
    for cat, p in tar_procs:
        if p.wait() != 0:
            raise SystemExit(f"FATAL: {cat}.tar 下载失败")
    print("3 个句型包下载完成")

    wanted = {}  # tar member 后缀 → 行
    for _, row in real.iterrows():
        wanted[row["audio_path"]] = row

    audio_out = OUT / "audio"
    n = 0
    with (OUT / "manifest.jsonl").open("w", encoding="utf-8") as mf:
        for cat in CATEGORIES:
            tar_path = WORK / f"{cat}.tar"
            key = f"data/audio_shards/by_category/{cat}.tar"
            if key in expected:
                actual = hashlib.sha256(tar_path.read_bytes()).hexdigest()
                if actual != expected[key]:
                    raise SystemExit(f"FATAL: {cat}.tar sha256 不符，不解析")
                print(f"{cat}.tar sha256 校验通过")
            before = n
            with tarfile.open(tar_path) as tf:
                for member in tf:
                    if not member.isfile() or not member.name.endswith(".wav"):
                        continue
                    # 成员路径可能是相对/带 ./ 前缀，按路径尾部对齐元数据
                    norm = member.name.lstrip("./")
                    hit = next((r for suffix, r in wanted.items()
                                if norm.endswith(suffix.split("Dataset_Balanced19/")[-1])
                                or norm.endswith(suffix)), None)
                    if hit is None:
                        continue
                    dst = audio_out / f"{hit['speaker_id']}_{hit['take_id']}_{hit['wav_name']}"
                    dst.parent.mkdir(parents=True, exist_ok=True)
                    with tf.extractfile(member) as src, dst.open("wb") as f:
                        f.write(src.read())
                    mf.write(json.dumps({
                        "audio": str(dst), "text": hit["transcript"],
                        "speaker": f"atika88:{hit['speaker_id']}",
                        # 说话人标签是数据集作者登记的真实身份（非聚类猜测），标记可信
                        "speaker_verified": True,
                        "session": hit["take_id"], "origin_audio": str(dst),
                        "lang": "id", "category": hit["category"],
                        "source_dataset": REPO, "source_license": "CC-BY-4.0",
                    }, ensure_ascii=False) + "\n")
                    n += 1
                    if n % 1000 == 0:
                        print(f"已解出 {n} 条")
            print(f"{cat}: 解出 {n - before} 条（该包元数据 {sum(1 for r in wanted.values() if r['category'] == cat)} 条）")
    print(f"完成：{n} 条 → {OUT}/manifest.jsonl")


if __name__ == "__main__":
    sys.exit(main())
