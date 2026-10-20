# -*- coding: utf-8 -*-
"""构建短 cue 探针：tl_short_holdout 的 14 个 FLEURS 句 id，补齐 th/id 同句译文。

FLEURS 跨语种句级平行（同一 source_sentence_id 是同一句子的不同语言版），
所以从同 pinned revision 的 th/id test.tsv 取同 id 句子，得到可跨语种对照的短文本探针。
每条文本 × 3 种 ref_lang（zh/en/tl），ref 沿用对应 tl_short 行同位置的 ref 文件
（改写为相对路径，远端靠 eval_cases/refs 软链与 prod_ref_fil.wav 软链解析）。
共 14 × 3 语种 × 3 ref = 126 行。

是否见过训练集：按 r12a_refrob_clean/train.jsonl 归一化文本逐条标记
cohort = r12_train_seen / r12_train_unseen（base/r8 的可见性另算；r8 的语料已删，
只标 r12a 口径，配对时按 cohort 分组读）。
输出写本仓 eval_cases/short_cue_probe_20261020.jsonl（远端生成后回收进 git）。
"""
import json
import sys
from pathlib import Path

from voxft.data.pipeline import _read_manifest, _write_jsonl
from voxft.eval import _norm
from voxft.paths import DATA_PROCESSED, ROOT

from holdout_texts import parse_tsv, REVISION  # 同一份 FLEURS 固定 revision

SENT_IDS = None  # 从 tl_short_holdout 读
TL_SHORT = ROOT / "eval_cases" / "tl_short_holdout_20260918.jsonl"
OUT = ROOT / "eval_cases" / "short_cue_probe_20261020.jsonl"
CONFIGS = {"th": "th_th", "tl": "fil_ph", "id": "id_id"}


def fetch_test_rows(lang: str):
    from huggingface_hub import hf_hub_download
    path = hf_hub_download(repo_id="google/fleurs", repo_type="dataset",
                           revision=REVISION, filename=f"data/{CONFIGS[lang]}/test.tsv")
    return {r["source_sentence_id"]: r for r in parse_tsv(Path(path).read_text(encoding="utf-8"))}


def main():
    tl_rows = _read_manifest(TL_SHORT)
    sent_ids = sorted({r["source_sentence_id"] for r in tl_rows}, key=int)
    print(f"14 句 id（实际 {len(sent_ids)}）：{sent_ids}")
    # 每个 (句id, ref_lang) 在 tl 行里用哪条 ref（取相对路径）
    ref_by_slot = {}
    for r in tl_rows:
        reflang = r["case_id"].rsplit("_", 1)[1]
        ref = r["ref_audio"].replace("/root/autodl-tmp/eval_cases/", "")
        ref_by_slot[(r["source_sentence_id"], reflang)] = ref

    r12a_texts = {_norm(r["text"]) for r in
                  _read_manifest(DATA_PROCESSED / "r12a_refrob_clean" / "train.jsonl")}

    rows, stats = [], {"seen": 0, "unseen": 0}
    texts_by_lang = {}
    for lang in ("tl", "th", "id"):
        table = fetch_test_rows(lang)
        missing = [s for s in sent_ids if s not in table]
        if missing:
            # FLEURS 各语种 test 集不完全对齐；缺句整句剔除，保住跨语种平行对照
            print(f"⚠️ fleurs {lang} test 缺句 id {missing}，整句剔除")
            sent_ids = [s for s in sent_ids if s not in missing]
        texts_by_lang[lang] = table
    for sid in sent_ids:
        for lang in ("th", "tl", "id"):
            text = texts_by_lang[lang][sid]["text"]
            cohort = "r12_train_seen" if _norm(text) in r12a_texts else "r12_train_unseen"
            stats["seen" if cohort == "r12_train_seen" else "unseen"] += 1
            for reflang in ("zh", "en", "tl"):
                rows.append({
                    "case_id": f"sc_{lang}_{sid}_{reflang}",
                    "lang": lang, "text": text,
                    "ref_audio": ref_by_slot[(sid, reflang)], "ref_lang": reflang,
                    "numeric": False, "cohort": cohort,
                    "source_sentence_id": sid,
                    "source_dataset": "google/fleurs", "source_revision": REVISION,
                    "source_split": "test", "source_license": "CC-BY-4.0",
                    "note": "短 cue 探针：FLEURS 平行句库同句 id 跨语种对照；tl 行文本与 tl_short_holdout 同源",
                })
    _write_jsonl(rows, OUT)
    print(f"写出 {len(rows)} 行 -> {OUT}；r12a 训练集已见 {stats['seen']} / 未见 {stats['unseen']}")


if __name__ == "__main__":
    sys.exit(main())
