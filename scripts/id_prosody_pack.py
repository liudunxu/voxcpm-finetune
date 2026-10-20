# -*- coding: utf-8 -*-
"""id 口音试听包：6 条探针文本 × 4 模型（base/r8/r12a/r13），同 ref（zh）同 seed（42）。

从远端 4 份探针报告取 wav 路径，scp 回本地，按文本分组、组内模型顺序随机（固定 seed），
盲标甲乙丙丁 + 离线 HTML（原生 <audio>，无外部依赖），映射存 key.json。
用法：python3 scripts/id_prosody_pack.py  （在本机跑，经 ssh 取音频）
"""
import json
import random
import subprocess
from pathlib import Path

EVAL_DIR = "/root/autodl-tmp/voxft_ckpt/eval"
SSH = ["ssh", "-p", "51348", "-o", "StrictHostKeyChecking=no",
       "root@connect.bjb1.seetacloud.com"]
WORK_LOCAL = Path("work/audio_ab/20261020_id_prosody")
MODELS = ["base", "r8", "r12a", "r13x"]


def find_probe_reports():
    """远端找 4 个模型各自的 id 探针报告（items 的 case_id 以 id_ms_ 开头）。"""
    cmd = ("python3 -c \"import json,glob,os;"
           "fs=sorted(glob.glob('/root/autodl-tmp/voxft_ckpt/eval/*.json'),key=os.path.getmtime);"
           "out={};"
           "[out.setdefault(json.load(open(f))['label'],f) for f in fs "
           " if json.load(open(f))['items'] and json.load(open(f))['items'][0]['case_id'].startswith('id_ms_')];"
           "print(json.dumps(out))\"")
    raw = subprocess.run(["bash", "/tmp/rsh", cmd], capture_output=True, text=True,
                         check=True).stdout.strip()
    by_label = json.loads(raw)
    label_map = {"base": "base", "r8": "lora_omni5_r8_latest",
                 "r12a": "lora_omni5_r12a_refrob_clean_latest",
                 "r13x": "lora_omni5_r13x_idemph_3ep_latest"}
    return {m: by_label[l] for m, l in label_map.items()}


def main():
    reports = find_probe_reports()
    print("报告：", json.dumps(reports, indent=1))
    items = {}
    texts = {}
    for model, rpath in reports.items():
        raw = subprocess.run(["bash", "/tmp/rsh", f"cat {rpath}"],
                             capture_output=True, text=True, check=True).stdout
        rep = json.loads(raw)
        for it in rep["items"]:
            if it["seed"] != 42 or it.get("ref_lang") != "zh":
                continue
            text_id = it["case_id"].rsplit("_", 1)[0]  # 只去 ref_lang 后缀
            texts[text_id] = it["text"]
            items[(text_id, model)] = it["wav"]

    rng = random.Random(20261020)
    audio_dir = WORK_LOCAL / "audio"
    audio_dir.mkdir(parents=True, exist_ok=True)
    key = {}
    for tid in sorted(texts):
        order = MODELS[:]
        rng.shuffle(order)
        key[tid] = {label: model for label, model in zip("甲乙丙丁", order)}
        for label, model in key[tid].items():
            src = items[(tid, model)]
            dst = audio_dir / f"{tid}_{label}.wav"
            if not dst.exists():
                subprocess.run(["scp", "-P", "51348", "-q",
                                f"root@connect.bjb1.seetacloud.com:{src}", str(dst)],
                               check=True)

    rows = []
    for tid in sorted(texts):
        cells = "".join(
            f'<td><b>{label}</b><br><audio controls preload="none" '
            f'src="audio/{tid}_{label}.wav"></audio></td>'
            for label in "甲乙丙丁")
        rows.append(f'<tr><td style="max-width:380px">{texts[tid]}</td>{cells}</tr>')
    html = f"""<!doctype html><meta charset="utf-8"><title>id 口音试听 20261020</title>
<p>6 条印尼语文本 × 4 个模型（甲乙丙丁盲标）。请母语者只听「像不像地道印尼语 /
有没有马来腔」，每组选一个最地道的，听不出就选「持平」。映射在 key.json。</p>
<table border="1" cellpadding="6">{''.join(rows)}</table>"""
    (WORK_LOCAL / "index.html").write_text(html, encoding="utf-8")
    (WORK_LOCAL / "key.json").write_text(json.dumps(key, ensure_ascii=False, indent=1),
                                         encoding="utf-8")
    print(f"试听包：{WORK_LOCAL}/index.html（{len(texts)} 组 × 4 模型）")


if __name__ == "__main__":
    main()
