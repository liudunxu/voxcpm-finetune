#!/usr/bin/env python3
"""构建 id 口音地道性的母语者盲听包（复用已生成的评测音频，不新增任何合成/ASR）。

为什么要有这个包：口音与语调地道性**没有任何自动指标能判**（f0_std_st 含泰语声调与清浊音误差、
CER 只测音段、speaker_sim 只测音色、词表扫描只验正字法），从 2026-09-18 用户第一次提"id 像 MY 腔"
到今天一直是"未验证"。这个包是唯一能把该维度变成有结论的路径，也决定 `enable_proj` 值不值得开：
听感若指向"输出带中文/英文节奏"⇒ proj 是机制对症的杠杆（跨语言 ref 的韵律泄漏通路）；
若指向"印尼语内部口音像马来"⇒ 属基座预训练先验，proj 无效。

选取：id 韵律探针的 **zh 参考**变体（线上真实形态就是中文 ref → id 输出）、6 条诊断文本、
固定 seed（默认 42）、7 个模型（base 与 6 个候选）。口音是系统性属性、不像跑飞那样依赖 seed，
所以单 seed 足够；每条另设"明显生成异常"标记，异常条不作口音证据。

两阶段：
  plan  —— 从本地报告里选出条目，写 fetch_list.json（远端 WAV 路径清单）
  build —— 从已下载的 WAV 目录构建盲听包（public/ 可外发，private/ 不得外发）

盲化与完整性守卫（任一不满足即失败退出，不写半个产物）：
- 每组内模型顺序随机化，公开侧只出现组号+字母代码，**public/ 里不得出现任何模型 label**（构建后扫描断言）；
- 每条 WAV 的实际时长必须与报告里的 audio_sec 相符（±0.05s），否则拒绝——防止拿错文件；
- 拒绝覆盖已存在的包目录；映射只写进 private/。
"""
import argparse
import json
import random
import shutil
import sys
import wave
from pathlib import Path

RESCORE = Path("work/rescore")
DEFAULT_MODELS = {
    "base": "base_c529323baaf34af3a90845fa87ad36bb.json",
    "r18": "lora_omni5_r18_idaccent_3ep_latest_9385e915bf4d46e2b640b549446e5fe8.json",
    "r23": "lora_omni5_r23_thtlboost_cont_latest_dd5f5c56755149bebed5e4921e98559f.json",
    "r24": "lora_omni5_r24_coldstart_latest_bec165693e9b4edea8ae40d8c425b5b9.json",
    "r25": "lora_omni5_r25_r18_prefix_graft_latest_f6da45869f394e019b4fb6362aefa35d.json",
    "r26": "lora_omni5_r26_r18_inoc10_latest_561b62ba50d24ee291bc5c9f1acfeb4e.json",
    "r27": "lora_omni5_r27_scale25h_latest_bd51fff6a9454293b010c0653fb6606a.json",
}
CODES = "ABCDEFG"
DUR_TOL = 0.05


def resolve_models(names: str | None) -> dict[str, Path]:
    """--models 逗号分隔；缺省用全部。名字必须在 DEFAULT_MODELS 里，避免打错字静默少一个模型。"""
    if not names:
        return {k: RESCORE / v for k, v in DEFAULT_MODELS.items()}
    want = [n.strip() for n in names.split(",") if n.strip()]
    unknown = [n for n in want if n not in DEFAULT_MODELS]
    if unknown:
        raise SystemExit(f"未知模型 {unknown}；可选：{sorted(DEFAULT_MODELS)}")
    if len(want) > len(CODES):
        raise SystemExit(f"最多 {len(CODES)} 个模型（代码只有 {CODES}）")
    return {n: RESCORE / DEFAULT_MODELS[n] for n in want}


def select(reports: dict[str, Path], ref_lang: str, seed: int) -> list[dict]:
    """按 (case_id) 对齐所有模型，返回每条待取音频的描述。"""
    per_model = {}
    for name, path in reports.items():
        doc = json.loads(Path(path).read_text(encoding="utf-8"))
        idx = {(i["case_id"], i["seed"]): i for i in doc.get("items") or []}
        per_model[name] = idx
    common = None
    for name, idx in per_model.items():
        keys = {k[0] for k, v in idx.items()
                if k[1] == seed and (v.get("ref_lang") or "") == ref_lang}
        common = keys if common is None else (common & keys)
    if not common:
        raise SystemExit(f"没有所有模型都具备的 (case, seed={seed}, ref_lang={ref_lang}) 条目")
    out = []
    for case_id in sorted(common):
        ref = per_model[next(iter(per_model))][(case_id, seed)]
        out.append({
            "case_id": case_id, "seed": seed, "lang": ref.get("lang"),
            "text": ref.get("text"), "ref_lang": ref.get("ref_lang"),
            "ref_audio": ref.get("ref_audio"),
            "models": {name: {"wav": idx[(case_id, seed)].get("wav"),
                              "audio_sec": idx[(case_id, seed)].get("audio_sec"),
                              "cer": idx[(case_id, seed)].get("cer"),
                              "len_ratio": idx[(case_id, seed)].get("len_ratio"),
                              "over_read": idx[(case_id, seed)].get("over_read")}
                       for name, idx in per_model.items()},
        })
    return out


def wav_duration(path: Path) -> float:
    with wave.open(str(path), "rb") as w:
        return w.getnframes() / float(w.getframerate())


def apply_overrides(reports: dict[str, Path], overrides: list[str]) -> dict[str, Path]:
    """--report NAME=PATH 覆盖某个模型用的报告。

    需要它是因为 auditions 会轮换清理：base/r23 的 id 探针音频（10-21 生成）已被清掉，
    只能用同口径补生成的新报告；补生成结果已与冻结报告逐条核对（audio_sec/CER/speaker_sim 全等）。
    """
    out = dict(reports)
    for item in overrides or []:
        if "=" not in item:
            raise SystemExit(f"--report 需要 NAME=PATH 格式，收到 {item!r}")
        name, path = item.split("=", 1)
        if name not in out:
            raise SystemExit(f"--report 的 {name} 不在本次模型列表 {sorted(out)} 里")
        if not Path(path).is_file():
            raise SystemExit(f"--report 指向的报告不存在：{path}")
        out[name] = Path(path)
    return out


def cmd_plan(args) -> int:
    reports = apply_overrides(resolve_models(args.models), args.report)
    entries = select(reports, args.ref_lang, args.seed)
    rows = []
    for e in entries:
        for model, info in e["models"].items():
            if not info["wav"]:
                raise SystemExit(f"{model}/{e['case_id']} 的报告里没有 wav 路径")
            rows.append({"key": f"{e['case_id']}__{model}", "remote": info["wav"],
                         "model": model, "case_id": e["case_id"], "seed": e["seed"],
                         "audio_sec": info["audio_sec"], "text": e["text"],
                         "ref_lang": e["ref_lang"], "ref_audio": e["ref_audio"]})
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"ref_lang": args.ref_lang, "seed": args.seed,
                               "models": sorted(reports), "cases": entries, "fetch": rows},
                              ensure_ascii=False, indent=2), encoding="utf-8")
    paths = Path(str(out) + ".paths.txt")
    paths.write_text("".join(r["remote"] + "\n" for r in rows), encoding="utf-8")
    print(f"选出 {len(entries)} 条文本 × {len(reports)} 个模型（{', '.join(sorted(reports))}）= {len(rows)} 条音频")
    print(f"  plan → {out}")
    print(f"  远端路径清单 → {paths}（用 tar -T 取回）")
    for e in entries:
        print(f"  - {e['case_id']:<24} {e['text'][:52]!r}")
    return 0


def cmd_build(args) -> int:
    plan_path = Path(args.plan)
    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    raw = Path(args.raw)
    dst = Path(args.dst)
    if dst.exists():
        print(f"拒绝覆盖已存在的包目录：{dst}", file=sys.stderr)
        return 1

    # 1) 校验每条音频存在且时长与报告一致
    missing, mismatch = [], []
    for r in plan["fetch"]:
        p = raw / f"{r['key']}.wav"
        if not p.exists():
            missing.append(r["key"])
            continue
        d = wav_duration(p)
        if abs(d - float(r["audio_sec"])) > DUR_TOL:
            mismatch.append((r["key"], d, r["audio_sec"]))
    if missing or mismatch:
        print(f"拒绝构建：缺失 {len(missing)} 条 {missing[:5]}", file=sys.stderr)
        for k, d, exp in mismatch[:5]:
            print(f"  时长不符 {k}: 实际 {d:.3f}s vs 报告 {exp}s", file=sys.stderr)
        return 1

    # 2) 组内随机化（固定 seed，可复现；映射只进 private/）
    rng = random.Random(args.shuffle_seed)
    models = plan.get("models") or sorted(DEFAULT_MODELS)
    codes = CODES[:len(models)]
    groups = []
    for gi, case in enumerate(plan["cases"], 1):
        order = sorted(case["models"])
        rng.shuffle(order)
        groups.append({"group": gi, "case_id": case["case_id"], "text": case["text"],
                       "seed": case["seed"], "ref_lang": case["ref_lang"],
                       "ref_audio": case["ref_audio"],
                       "codes": {code: m for code, m in zip(codes, order)}})

    # 3) 写产物
    pub = dst / "public"
    (pub / "audio").mkdir(parents=True)
    priv = dst / "private"
    priv.mkdir(parents=True)
    mapping = []
    for g in groups:
        for code, model in g["codes"].items():
            src = raw / f"{g['case_id']}__{model}.wav"
            name = f"g{g['group']}_{code.lower()}.wav"
            shutil.copyfile(src, pub / "audio" / name)
            info = plan["cases"][g["group"] - 1]["models"][model]
            mapping.append({"group": g["group"], "code": code, "file": f"audio/{name}",
                            "model": model, "case_id": g["case_id"], "seed": g["seed"],
                            "ref_lang": g["ref_lang"], "ref_audio": g["ref_audio"],
                            "text": g["text"], "source_wav": src.name,
                            "reported_audio_sec": info["audio_sec"], "reported_cer": info["cer"],
                            "reported_len_ratio": info["len_ratio"],
                            "reported_over_read": info["over_read"],
                            "actual_duration_sec": round(wav_duration(pub / "audio" / name), 3)})
    (priv / "mapping.json").write_text(json.dumps(
        {"shuffle_seed": args.shuffle_seed, "models": models,
         "note": "私有映射，不得与 public/ 一起外发；含模型身份与自动分数。",
         "entries": mapping}, ensure_ascii=False, indent=2), encoding="utf-8")
    (pub / "index.html").write_text(render_html(groups, codes), encoding="utf-8")

    # 4) 泄漏守卫：public/ 里不得出现模型身份。
    #    case_id 不在禁词里——它在同一组内对所有 clip 都相同，不泄漏"哪条来自哪个模型"，
    #    而台词本身必须显示给听者（禁它会因为台词里恰好含 case_id 字样而误报）。
    banned = set(models) | {"lora_omni5", "coldstart", "inoc10", "prefix_graft", "scale25h",
                            "idaccent", "thtlboost", "nothaiser", "emofix"}
    leaks = []
    for p in pub.rglob("*"):
        if not p.is_file():
            continue
        blob = p.read_bytes()
        text = blob.decode("utf-8", "ignore") if p.suffix in (".html", ".json", ".txt") else ""
        for b in banned:
            if b and (b in text or b.encode() in p.name.encode()):
                leaks.append((str(p.relative_to(dst)), b))
    if leaks:
        shutil.rmtree(dst)
        print(f"拒绝：public/ 泄漏了身份信息 {leaks[:6]}（已回滚删除整个包）", file=sys.stderr)
        return 1

    print(f"盲听包已生成 → {dst}")
    print(f"  public/  可外发：index.html + audio/{len(mapping)} 个 wav（组内顺序已随机化）")
    print(f"  private/ 不得外发：mapping.json（代码 → 模型 / 自动分数）")
    print(f"  {len(groups)} 组 × {len(codes)} 条 = {len(mapping)} 条；shuffle_seed={args.shuffle_seed}")
    return 0


def render_html(groups, codes=CODES) -> str:
    """公开页面文案用英文：听者是印尼语母语者，中文说明等于没说明。

    data-* 属性与导出文本的字段名保持不变（accent/defect/best/rank/note），
    这样换语言不影响回流的反馈解析。
    """
    cards = []
    for g in groups:
        rows = "".join(
            f"""      <div class="clip">
        <span class="code">{g['group']}{c}</span>
        <audio controls preload="none" src="audio/g{g['group']}_{c.lower()}.wav"></audio>
        <label>Non-Indonesian rhythm/accent
          <select data-g="{g['group']}" data-c="{c}" data-q="accent">
            <option value="">—</option><option value="no">No</option>
            <option value="yes">Yes</option><option value="unsure">Not sure</option>
          </select>
        </label>
        <label>Obvious generation defect (runaway / repetition / silence / cut-off)
          <select data-g="{g['group']}" data-c="{c}" data-q="defect">
            <option value="">—</option><option value="no">No</option><option value="yes">Yes</option>
          </select>
        </label>
      </div>""" for c in codes)
        opts = "".join(f'<label class="pick"><input type="checkbox" data-g="{g["group"]}" '
                       f'data-c="{c}" data-q="best"> {g["group"]}{c}</label>' for c in codes)
        cards.append(f"""  <section class="group">
    <h2>Group {g['group']}</h2>
    <p class="text">Line spoken in every clip of this group: <em>{g['text']}</em></p>
{rows}
    <div class="q">
      <p><strong>Q1. Which clip sounds most like a native Indonesian speaker?</strong>
         (tick as many as you like; leave all unticked if none does)</p>
      <div class="picks">{opts}</div>
      <p><strong>Q2. Overall ranking for this group (optional)</strong>
        <input type="text" data-g="{g['group']}" data-q="rank" placeholder="e.g. C &gt; A &gt; D &gt; B" size="34"></p>
      <p><strong>Comments</strong>
        <input type="text" data-g="{g['group']}" data-q="note"
               placeholder="which clip, what sounds wrong, which language it reminds you of" size="64"></p>
    </div>
  </section>""")
    return f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<title>Indonesian accent blind listening ({len(groups)} groups × {len(codes)} clips)</title>
<style>
 body{{font-family:system-ui,-apple-system,"Segoe UI",sans-serif;margin:24px;max-width:940px;line-height:1.5}}
 .group{{border:1px solid #ddd;border-radius:8px;padding:12px 16px;margin:18px 0}}
 h1{{font-size:20px}} h2{{font-size:16px;margin:0 0 6px}}
 .text{{color:#444;margin:0 0 10px}} .text em{{font-style:normal;color:#111}}
 .clip{{display:flex;align-items:center;gap:10px;margin:6px 0;flex-wrap:wrap}}
 .code{{display:inline-block;min-width:26px;font-weight:600}}
 audio{{height:32px}}
 label{{font-size:13px;color:#333}} select{{font-size:13px}}
 .q{{margin-top:12px;padding-top:10px;border-top:1px dashed #ccc}}
 .picks{{display:flex;gap:12px;flex-wrap:wrap;margin:4px 0 8px}}
 .pick{{font-weight:600}}
 #out{{width:100%;height:230px;font-family:ui-monospace,Menlo,monospace;font-size:12px}}
 button{{font-size:14px;padding:8px 14px;margin:10px 0}}
 .hint{{background:#fff8e1;border:1px solid #eed;border-radius:6px;padding:10px 14px;font-size:13.5px}}
 .hint li{{margin:4px 0}}
</style></head><body>
<h1>Indonesian accent &amp; intonation — blind listening test</h1>
<div class="hint">
 <p>Each group contains {len(codes)} clips. <strong>Within a group, every clip speaks the same Indonesian
 sentence, was conditioned on the same reference voice, and used the same random seed.</strong>
 Only the underlying system differs — the order is shuffled and nothing is labelled.</p>
 <p>Please judge by ear only, and answer three things:</p>
 <ol>
  <li><strong>Which clip sounds most like a native Indonesian speaker?</strong>
      Ties are fine; if none does, tick nothing.</li>
  <li>For each clip: <strong>does it carry a non-Indonesian rhythm or accent</strong>
      (Chinese, English, or Malay)? If you cannot tell, choose “Not sure” — please do not guess.</li>
  <li>For each clip: <strong>is there an obvious generation defect</strong>
      (keeps talking past the sentence, repeats itself, silence, cut-off words)?
      If yes, mark it as a defect and <strong>do not treat that clip as accent evidence</strong>.</li>
 </ol>
 <p>When finished, press <strong>Export feedback</strong> at the bottom and send back the whole text block.
 Unanswered items stay unanswered — they are <strong>not</strong> counted as “no problem” or as a tie.
 <strong>The browser does not save anything automatically</strong>, so please do not refresh halfway.</p>
 <p>Terima kasih! Roughly 10–15 minutes for all {len(groups)} groups.</p>
</div>
{chr(10).join(cards)}
<button id="exp">Export feedback</button>
<textarea id="out" readonly placeholder="Press “Export feedback” above, then select and copy everything from this box."></textarea>
<script>
document.getElementById('exp').addEventListener('click', function () {{
  var lines = ['ACCENT_BLIND_FEEDBACK v1', 'groups={len(groups)} codes_per_group={len(codes)}',
               'filled_at=' + new Date().toISOString()];
  document.querySelectorAll('.group').forEach(function (sec) {{
    var g = sec.querySelector('select').dataset.g;
    var best = [];
    sec.querySelectorAll('input[data-q="best"]:checked').forEach(function (x) {{ best.push(x.dataset.c); }});
    lines.push('group=' + g + ' best=' + (best.join('+') || 'none'));
    sec.querySelectorAll('select[data-q="accent"]').forEach(function (x) {{
      lines.push('  ' + g + x.dataset.c + ' accent=' + (x.value || 'unrated'));
    }});
    sec.querySelectorAll('select[data-q="defect"]').forEach(function (x) {{
      if (x.value) lines.push('  ' + g + x.dataset.c + ' defect=' + x.value);
    }});
    ['rank', 'note'].forEach(function (q) {{
      var el = sec.querySelector('[data-q="' + q + '"]');
      if (el && el.value.trim()) lines.push('  ' + q + '=' + el.value.trim());
    }});
  }});
  var rated = lines.filter(function (l) {{ return /accent=(no|yes|unsure)/.test(l); }}).length;
  lines.push('rated_clips=' + rated + '/' + ({len(groups)} * {len(codes)}));
  lines.push('note: unrated clips stay unrated; they are not treated as clear or as a tie.');
  document.getElementById('out').value = lines.join('\n');
}});
</script>
</body></html>
"""


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("plan", help="从本地报告选出条目并写远端取件清单")
    p.add_argument("--ref-lang", default="zh")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--models", default=None,
                   help=f"逗号分隔的模型名，缺省全部：{','.join(sorted(DEFAULT_MODELS))}")
    p.add_argument("--report", action="append", default=[], metavar="NAME=PATH",
                   help="覆盖某个模型用的报告路径（可重复）；用于原音频被清理后按同口径补生成的报告")
    p.add_argument("--out", default="work/audio_ab/20261023_id_accent/plan.json")
    p.set_defaults(func=cmd_plan)
    b = sub.add_parser("build", help="从已下载的 WAV 构建盲听包")
    b.add_argument("--plan", required=True)
    b.add_argument("--raw", required=True, help="含 <case_id>__<model>.wav 的目录")
    b.add_argument("--dst", required=True)
    b.add_argument("--shuffle-seed", type=int, default=20261023)
    b.set_defaults(func=cmd_build)
    a = ap.parse_args()
    return a.func(a)


if __name__ == "__main__":
    sys.exit(main())
