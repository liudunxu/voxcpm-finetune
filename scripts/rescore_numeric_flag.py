# -*- coding: utf-8 -*-
"""用修复后的 numeric 口径（含泰文数字词）重算存量 eval 报告的聚合指标。

不改原报告文件：逐 item 重算 numeric 标记（存量标记 OR 泰文数字词命中，
等价于新代码在 v2 上的判定——v2 里所有显式 numeric=true 的 case 本来就含
阿拉伯数字），再用 voxft.eval._agg 重新聚合，打印新旧对照。

用法：uv run python scripts/rescore_numeric_flag.py <报告.json> [更多报告...]
"""
import json
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from voxft.eval import _THAI_NUM_WORDS, _agg  # noqa: E402


def rescore(path: Path) -> dict:
    report = json.loads(path.read_text(encoding="utf-8"))
    items = [dict(i, numeric=bool(i["numeric"]) or bool(_THAI_NUM_WORDS.search(i["text"] or "")))
             for i in report["items"]]
    by_lang = defaultdict(list)
    for item in items:
        by_lang[item["lang"]].append(item)
    return {
        "label": report["label"],
        "overall": _agg(items),
        "by_lang": {lang: _agg(g) for lang, g in sorted(by_lang.items())},
        "old_overall": {k: report.get(k) for k in ("mean_cer", "mean_cer_non_numeric")},
        "old_by_lang": {lang: {k: v.get(k) for k in ("mean_cer", "mean_cer_non_numeric")}
                        for lang, v in (report.get("by_lang") or {}).items()},
    }


def main():
    for arg in sys.argv[1:]:
        r = rescore(Path(arg))
        print(f"\n=== {r['label']} ===")
        old, new = r["old_overall"], r["overall"]
        print(f"总体  mean_cer {old['mean_cer']} -> {new['mean_cer']} | "
              f"non_numeric {old['mean_cer_non_numeric']} -> {new.get('mean_cer_non_numeric')} "
              f"(n={new.get('non_numeric_cases')})")
        for lang, agg in r["by_lang"].items():
            o = r["old_by_lang"].get(lang, {})
            print(f"  {lang}: cer {o.get('mean_cer')} -> {agg['mean_cer']} | "
                  f"non_numeric {o.get('mean_cer_non_numeric')} -> {agg.get('mean_cer_non_numeric')} "
                  f"(n={agg.get('non_numeric_cases')}/{agg['cases']})")


if __name__ == "__main__":
    main()
