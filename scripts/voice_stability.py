"""已有报告的生成音频两两音色一致性；同 case 跨 seed、同 ref/seed 跨 cue 分开。"""
import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from voxft import eval as evaluation
from voxft.data.pipeline import load_wav_mono
from voxft.qc.audio import speaker_embedding


def summarize_embeddings(items, embeddings):
    evaluation.seed_stability(items)
    out = {}
    for mode in ("cross_seed", "cross_cue"):
        groups = defaultdict(list)
        for index, item in enumerate(items):
            if mode == "cross_cue" and not item.get("ref_audio"):
                continue
            key = (item["lang"], item.get("ref_audio"),
                   item["case_id"] if mode == "cross_seed" else item["seed"])
            groups[key].append(index)
        rows = []
        for (lang, reference, unit), indices in groups.items():
            if len(indices) < 2:
                continue
            vectors = embeddings[indices]
            matrix = vectors @ vectors.T
            values = matrix[np.triu_indices(len(indices), 1)]
            rows.append({"lang": lang, "ref_audio": reference, "unit": unit,
                         "items": len(indices), "pairs": len(values),
                         "mean_cosine": round(float(values.mean()), 6),
                         "min_cosine": round(float(values.min()), 6)})
        out[mode] = {
            "groups": rows,
            "by_lang": {lang: {
                "groups": len(group),
                "mean_cosine": round(float(np.mean([row["mean_cosine"] for row in group])), 6),
                "min_cosine": min(row["min_cosine"] for row in group)}
                for lang in sorted({row["lang"] for row in rows})
                if (group := [row for row in rows if row["lang"] == lang])},
        }
    out["note"] = ("WavLM 诊断，不等于人工娃娃音/音色判定。按组等权，"
                   "cross_cue 只合并同 ref 文件/同 seed/同语种；缺 ref 或单条组不当作稳定。")
    return out


def analyze(report_path):
    report_path = Path(report_path)
    report = json.loads(report_path.read_text())
    items = report["items"]
    vectors = []
    for index, item in enumerate(items):
        wav, sample_rate = load_wav_mono(item["wav"])
        vector = speaker_embedding(wav, sample_rate)
        if vector is None or not np.isfinite(vector).all():
            raise RuntimeError(f"Missing/invalid speaker embedding: {item['wav']}")
        vectors.append(vector)
        if (index + 1) % 25 == 0:
            print(f"{report_path.name}: {index + 1}/{len(items)} embeddings", flush=True)
    result = summarize_embeddings(items, np.stack(vectors))
    result["source_report"] = str(report_path)
    output = report_path.parent / "voice_stability" / report_path.name
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2))
    print(f"Saved {output}", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("reports", nargs="+")
    args = parser.parse_args()
    for report_path in args.reports:
        analyze(report_path)
