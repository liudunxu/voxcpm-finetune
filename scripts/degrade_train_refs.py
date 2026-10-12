"""Build the single-variable "50% train refs degraded" dataset copy (差参考音鲁棒性 B 臂).

唯一数据变量：train.jsonl 里带 ref_audio 的行按 seed 选 share（默认 50%），把 ref 换成
确定性降质副本（写在 <out>/ref_audio_degraded/<profile>/）；行集、split、文本、target
音频、配比、曝光次数全部不变。降质保持长度不变，ref_duration 不动。
原数据集目录一个字节都不改；val.jsonl 逐字节复制；干扰池只取同一 train split 内
speaker_verified=true 的行（防 val 泄漏），并按 ref_speaker/ref_origin_audio 逐行过滤。
"""
import argparse
import hashlib
import json
import random
import shutil
from collections import Counter
from pathlib import Path

from voxft.data.pipeline import _read_manifest, _write_jsonl
from voxft.data.ref_degrade import PROFILES, append_sidecars, degrade_file, derive_seed
from voxft.paths import DATA_PROCESSED

# 按行确定性分配 profile：sha256 取模，五档各约 20%；顺序即 PROFILES 声明顺序
_PROFILE_ORDER = list(PROFILES)


def _row_profile(seed: int, row: dict) -> str:
    digest = hashlib.sha256(f"profile:{seed}:{row['audio']}".encode("utf-8")).digest()
    return _PROFILE_ORDER[int.from_bytes(digest[:8], "big") % len(_PROFILE_ORDER)]


def _resolve_input(dataset: str) -> Path:
    p = Path(dataset)
    if p.is_dir():
        return p
    candidate = DATA_PROCESSED / dataset
    if candidate.is_dir():
        return candidate
    raise FileNotFoundError(f"数据集目录不存在（按路径与 processed 目录名都试过）: {dataset}")


def _resolve_output(out: str | None, src: Path) -> Path:
    if out is None:
        return src.parent / f"{src.name}_refrob"
    p = Path(out)
    return p if p.is_absolute() or p.parent != Path(".") else DATA_PROCESSED / p


def degrade_dataset(dataset: str, out: str | None = None, *, share: float = 0.5,
                    seed: int = 42, progress=None) -> dict:
    src = _resolve_input(dataset)
    out_dir = _resolve_output(out, src)
    if src.resolve() == out_dir.resolve():
        raise ValueError("输出目录不能与源数据集相同")
    if not (0 < share <= 1):
        raise ValueError("share 必须在 (0, 1] 区间")
    train_path = src / "train.jsonl"
    if not train_path.exists():
        raise FileNotFoundError(train_path)
    if (out_dir / "train.jsonl").exists() or (out_dir / "val.jsonl").exists():
        raise ValueError(f"输出已存在，不覆盖旧产物: {out_dir}")
    rows = _read_manifest(train_path)
    # 干扰池只取同一 train split 内已验证说话人的行；val 行绝不入池
    pool = [{"audio": r["audio"], "speaker": r["speaker"],
             "origin_audio": r.get("origin_audio", r["audio"])}
            for r in rows if r.get("speaker_verified") is True]
    ref_rows = [i for i, r in enumerate(rows) if r.get("ref_audio")]
    rng = random.Random(seed)
    selected = set(rng.sample(ref_rows, round(len(ref_rows) * share)))

    out_dir.mkdir(parents=True, exist_ok=True)
    degraded_dir = out_dir / "ref_audio_degraded"
    records: list[dict] = []
    cache: dict[tuple[str, str], dict] = {}  # (源 ref sha, profile) → sidecar，重复曝光不重复降质
    profile_counts: Counter = Counter()
    new_rows = []
    for i, row in enumerate(rows):
        row = dict(row)
        if i in selected:
            profile = _row_profile(seed, row)
            ref_src = Path(row["ref_audio"])
            with ref_src.open("rb") as f:
                src_sha = hashlib.file_digest(f, "sha256").hexdigest()
            key = (src_sha, profile)
            rec = cache.get(key)
            if rec is None:
                dst = degraded_dir / profile / f"{src_sha[:12]}_{ref_src.name}"
                try:
                    rec = degrade_file(ref_src, dst, profile,
                                       seed=derive_seed(seed, src_sha, profile),
                                       interferers=pool,
                                       ref_speaker=row.get("ref_speaker"),
                                       ref_origin=row.get("ref_origin_audio"))
                except ValueError as exc:
                    raise ValueError(f"行 {row['audio']} 的 ref 降质失败: {exc}") from exc
                rec["dst"] = str(dst)
                cache[key] = rec
                records.append(rec)
            row["ref_audio"] = rec["dst"]
            row["ref_audio_original"] = str(ref_src)
            row["ref_degradation"] = {"profile": profile,
                                      "severity": PROFILES[profile]["severity"]}
            profile_counts[profile] += 1
        new_rows.append(row)
        if progress and (i + 1) % 500 == 0:
            progress(f"处理 {i + 1}/{len(rows)} 行，已降质 {sum(profile_counts.values())} 行")

    _write_jsonl(new_rows, out_dir / "train.jsonl")
    val_src = src / "val.jsonl"
    if val_src.exists():
        shutil.copyfile(val_src, out_dir / "val.jsonl")  # 逐字节复制，一行都不动
    mix_src = src / "mix.json"
    if mix_src.exists():
        shutil.copyfile(mix_src, out_dir / "mix.json")
    append_sidecars(records, out_dir / "degradations.jsonl")
    stats = {"source": str(src), "output": str(out_dir), "share": share, "seed": seed,
             "train_rows": len(rows), "ref_rows": len(ref_rows),
             "replaced": sum(profile_counts.values()),
             "profiles": dict(profile_counts), "unique_degraded_refs": len(records),
             "interferer_pool": len(pool), "degraded_dir": "ref_audio_degraded"}
    (out_dir / "refrob_stats.json").write_text(
        json.dumps(stats, ensure_ascii=False, indent=2), encoding="utf-8")
    if progress:
        progress(f"完成：{len(rows)} 行 train，{len(ref_rows)} 行带 ref，"
                 f"替换 {stats['replaced']} 行（{dict(profile_counts)}）；输出 {out_dir}")
    return stats


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dataset", required=True, help="processed 数据集目录名或路径")
    ap.add_argument("--out", default=None, help="新数据集目录名，默认 <name>_refrob")
    ap.add_argument("--share", type=float, default=0.5, help="带 ref 行中降质的比例")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    print(json.dumps(degrade_dataset(args.dataset, args.out, share=args.share,
                                     seed=args.seed, progress=print),
                     ensure_ascii=False, indent=2))
