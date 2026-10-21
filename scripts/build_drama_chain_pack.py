# -*- coding: utf-8 -*-
"""drama_chain 盲听包：按用户优先级（badcase→音色一致→情绪）组织 base/r18 对照音频。

选取 seed=42 的全部 2×2 条件（28 文本 × 4 条件 × 2 模型 = 224 条 + 6 ref），
从远端 auditions rsync 回本地，按 case 目录归组，附 LISTENING.md 听评指引。

用法：uv run python scripts/build_drama_chain_pack.py <base报告> <r18报告> [输出目录]
远端 auditions 路径从报告的 wav 字段自动提取（默认 ssh -p 51348 root@connect.bjb1.seetacloud.com）。
"""
import json
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REMOTE = "ssh -p 51348 -o StrictHostKeyChecking=no root@connect.bjb1.seetacloud.com"
COND = re.compile(r"_(nref|sref)_(plain|emo)$")


def main():
    base_rep = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
    r18_rep = json.loads(Path(sys.argv[2]).read_text(encoding="utf-8"))
    out = Path(sys.argv[3]) if len(sys.argv) > 3 else ROOT / "work/audio_ab/drama_chain_20261021"
    out.mkdir(parents=True, exist_ok=True)

    def pick(rep):
        return {i["case_id"]: i for i in rep["items"] if i["seed"] == 42}

    base, r18 = pick(base_rep), pick(r18_rep)
    case_ids = sorted(set(base) & set(r18))
    if not case_ids:
        raise SystemExit("两份报告没有共同 case")

    # 收集远端 wav 清单（含 ref）
    remote_wavs = set()
    for cid in case_ids:
        for it in (base[cid], r18[cid]):
            remote_wavs.add(it["wav"])
            remote_wavs.add(it["ref_audio"])
    list_file = out / ".rsync_list"
    list_file.write_text("\n".join(sorted(remote_wavs)) + "\n")
    subprocess.run(["rsync", "-aq", "--files-from", str(list_file), "-e",
                    "ssh -p 51348 -o StrictHostKeyChecking=no",
                    "root@connect.bjb1.seetacloud.com:/", str(out / "remote")], check=True)
    list_file.unlink()

    manifest = []
    for cid in case_ids:
        m = COND.search(cid)
        ref_cond, ctrl_cond = m.group(1), m.group(2)
        text_key = COND.sub("", cid)
        d = out / text_key / f"{ref_cond}"
        d.mkdir(parents=True, exist_ok=True)
        ref_local = out / "remote" / base[cid]["ref_audio"].lstrip("/")
        ref_dst = out / "refs" / Path(base[cid]["ref_audio"]).name
        ref_dst.parent.mkdir(parents=True, exist_ok=True)
        if not ref_dst.exists():
            ref_dst.write_bytes(ref_local.read_bytes())
        for tag, it in (("base", base[cid]), ("r18", r18[cid])):
            src = out / "remote" / it["wav"].lstrip("/")
            dst = d / f"{tag}_{ctrl_cond}.wav"
            dst.write_bytes(src.read_bytes())
            manifest.append({"wav": str(dst.relative_to(out)), "case_id": cid, "lang": it["lang"],
                             "model": tag, "ref_cond": ref_cond, "ctrl_cond": ctrl_cond,
                             "control": it["control"], "text": it["text"],
                             "cer": it["cer"], "speaker_sim": it["speaker_sim"],
                             "ref_audio": str(ref_dst.relative_to(out))})
    (out / "manifest.jsonl").write_text(
        "\n".join(json.dumps(r, ensure_ascii=False) for r in manifest) + "\n", encoding="utf-8")
    subprocess.run(["rm", "-rf", str(out / "remote")], check=True)
    n_dirs = len({(COND.sub("", r["case_id"]), r["ref_cond"]) for r in manifest})
    print(f"{len(manifest)} 条 → {out}")
    print(f"case 目录 {n_dirs} 个（每目录 4 条：base/r18 × plain/emo；refs/ 为 6 条参考音）")


if __name__ == "__main__":
    main()
