# -*- coding: utf-8 -*-
"""短 cue 门恢复：shard1 的 r12a 因 auditions 目录误删中断，补齐后按 target 合并。

策略：
- shard0/shard2 的三目标报告已在（按「含 sc_ case + shard 标签 + 时间窗」识别）；
- shard1 的 base/r8 报告已在；只重跑 r12a 的 shard1（117 次生成）；
- 合并后校验每目标 351 条（117 case × 3 seed）、sc case 39 条/片。
用法：python3 /tmp/short_gate_rescue.py  （在远端跑，会自动调 voxft.eval）
"""
import json
import os
import subprocess
import sys
import time
from pathlib import Path

EVAL_DIR = Path("/root/autodl-tmp/voxft_ckpt/eval")
WORK = Path("/root/autodl-tmp/refrob_20261020")
GATE_START = 1792476480  # 2026-10-20 14:08 +0800
TARGETS = {  # label 前缀 -> shard 报告路径（随进程补全）
    "base": {},
    "lora_omni5_r8_latest": {},
    "lora_omni5_r12a_refrob_clean_latest": {},
}
R12A = "/root/autodl-tmp/voxft_ckpt/lora_omni5_r12a_refrob_clean/latest"


def is_gate_shard(name: str, k: int) -> bool:
    return f"_shard{k}of3_" in name


def inventory():
    found = {t: {} for t in TARGETS}
    for f in EVAL_DIR.glob("*.json"):
        for t in TARGETS:
            if not f.name.startswith(t + "_shard"):
                continue
            if f.stat().st_mtime < GATE_START:
                continue
            try:
                r = json.loads(f.read_text(encoding="utf-8"))
            except Exception:
                continue
            if not any(i["case_id"].startswith("sc_") for i in r["items"]):
                continue
            k = int(f.name.split("_shard")[1].split("of")[0])
            if len(r["items"]) != 117:
                raise SystemExit(f"片报告条数异常: {f.name} = {len(r['items'])}")
            found[t][k] = str(f)
    return found


def merge(tag, paths):
    log = WORK / f"logs/merge_short_rescue_{tag}.log"
    cmd = ["uv", "run", "python", "-X", "utf8", "-u", "-m", "voxft.eval", "--merge", *paths]
    with log.open("w") as lf:
        subprocess.run(cmd, cwd="/root/voxcpm-finetune", check=True, stdout=lf)
    out = None
    for line in log.read_text(encoding="utf-8").splitlines()[::-1]:
        if ".json" in line and "/root/" in line:
            out = line.split()[-1]
            break
    return out


def main():
    found = inventory()
    print("现状：", {t: sorted(v) for t, v in found.items()})
    # 1) 缺 r12a 的 shard1 → 只补它
    if 1 not in found["lora_omni5_r12a_refrob_clean_latest"]:
        print("补跑 r12a shard1/3 ...")
        cmd = ["uv", "run", "python", "-X", "utf8", "-u", "-m", "voxft.eval", R12A,
               "--lang", "th", "--texts-file", "eval_cases/short_cue_probe_20261020.jsonl",
               "--seeds", "42", "43", "44", "--cfg-value", "1.8",
               "--inference-timesteps", "20", "--shard", "1/3"]
        with (WORK / "logs/short_gate_shard1_rescue.log").open("w") as lf:
            subprocess.run(cmd, cwd="/root/voxcpm-finetune", check=True, stdout=lf)
        found = inventory()
    missing = {t: sorted(set(range(3)) - set(v)) for t, v in found.items()}
    missing = {t: m for t, m in missing.items() if m}
    if missing:
        raise SystemExit(f"仍缺分片: {missing}")
    # 2) 按 target 合并
    for t, shards in found.items():
        paths = [shards[k] for k in sorted(shards)]
        merged = merge(t, paths)
        n = len(json.loads(Path(merged).read_text(encoding="utf-8"))["items"])
        if n != 351:
            raise SystemExit(f"{t} 合并 {n} ≠ 351")
        with (WORK / "report_map_short_gate.jsonl").open("a", encoding="utf-8") as mf:
            mf.write(json.dumps({"tag": "short_gate", "target": t, "merged": merged,
                                 "rescue": True}) + "\n")
        print(f"合并完成 {t}: {merged} ({n} 条)")
    print("=== 短 cue 门恢复合并完成 ===")


if __name__ == "__main__":
    main()
