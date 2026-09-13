from __future__ import annotations

import threading
import time

import gradio as gr

from ..paths import DATA_PROCESSED, CHECKPOINT_DIR, env, load_dotenv
from ..data.registry import TARGET_LANGS, sources_by_quality
from ..data import download, ingest, pipeline
from ..train import launcher, yaml_builder
from ..lora.merge import merge_lora
from ..hub.sync import upload_folder
from ..log import file_tail, get_log
from .. import infer

PORT = int(env("VOXFT_UI_PORT", "6006"))


def _processed_datasets() -> list[str]:
    if not DATA_PROCESSED.exists():
        return []
    return sorted(d.name for d in DATA_PROCESSED.iterdir()
                  if (d / "train.jsonl").exists())


def _dataset_table() -> str:
    rows = []
    for name in _processed_datasets():
        stats = DATA_PROCESSED / name / "stats.json"
        if stats.exists():
            import json
            s = json.loads(stats.read_text())
            rows.append(f"{name}: train={s.get('train')} val={s.get('val')} "
                        f"speakers={s.get('speakers')} ref_audio={s.get('with_ref_audio')}")
        else:
            rows.append(name)
    return "\n".join(rows) or "（暂无，先下载并加工）"


def _stream(log_name: str, fn):
    """后台线程执行 fn(log)，流式把日志文本刷到页面。"""
    log = get_log(log_name)
    result: dict = {}

    def worker():
        try:
            result["msg"] = fn(log)
        except Exception as exc:
            log(f"失败: {exc}")
            result["msg"] = None

    t = threading.Thread(target=worker, daemon=True)
    t.start()
    while t.is_alive():
        yield log.text()
        time.sleep(0.8)
    yield log.text()


def _ds_choices_update():
    """加工后刷新各处数据集下拉框（混合页 + 训练页）。"""
    return gr.update(choices=_processed_datasets())


def do_preview(name, idx):
    try:
        import json
        p = DATA_PROCESSED / name / "train.jsonl"
        if not p.exists():
            return "（该数据集不存在）", None
        rows = [json.loads(l) for l in p.open(encoding="utf-8") if l.strip()]
        if not rows:
            return "（空数据集）", None
        i = int(idx or 0) % len(rows)
        rec = rows[i]
        info = (json.dumps(rec, ensure_ascii=False, indent=2)
                + f"\n（train 共 {len(rows)} 条，当前第 {i} 条）")
        return info, rec.get("audio")
    except Exception as exc:
        return f"预览失败: {exc}", None


def _config_files() -> list[str]:
    """configs/ 下已生成的训练配置（按时间从新到旧）。"""
    from ..paths import CONFIG_DIR
    if not CONFIG_DIR.exists():
        return []
    files = sorted(CONFIG_DIR.glob("*.yaml"),
                   key=lambda p: p.stat().st_mtime, reverse=True)
    return [str(p) for p in files]


def _upload_choices() -> list[str]:
    """可上传的本地目录：合并输出目录 + 各 LoRA run 的 latest（按时间从新到旧）。"""
    from ..lora.merge import is_lora_dir
    dirs: list = []
    merged = CHECKPOINT_DIR / "merged"
    if merged.exists():
        dirs.append(merged)
    if CHECKPOINT_DIR.exists():
        for d in CHECKPOINT_DIR.iterdir():
            if d.is_dir() and is_lora_dir(d / "latest"):
                dirs.append(d / "latest")
    dirs.sort(key=lambda d: d.stat().st_mtime, reverse=True)
    return [str(d) for d in dirs]


# ---------------- Tab 1: 数据集 ----------------

def do_download(source_id, max_samples):
    if not source_id:
        yield "请先选择数据源"
        return

    def fn(log):
        from ..data.registry import source_id_from_display
        sid = source_id_from_display(source_id)
        log(f"开始下载 {sid}（max_samples={max_samples or '全量'}）")
        dest = download.download_source(sid,
                                        int(max_samples) if max_samples else None,
                                        progress=log)
        log(f"完成 → {dest}/manifest.jsonl")
        return None

    yield from _stream("download", fn)


def do_process(source_id, min_dur, max_dur, val_ratio,
               control_ratio=None, manifest_path=""):
    if not source_id:
        yield ("请先选择原始数据源", _dataset_table(), _ds_choices_update(),
               _ds_choices_update(), _ds_choices_update())
        return

    def fn(log):
        import json
        from ..data.registry import get_source
        src = get_source(source_id)
        opts = pipeline.options_for(
            source_id, min_dur=float(min_dur), max_dur=float(max_dur),
            val_ratio=float(val_ratio),
            control_ratio=float(control_ratio) if control_ratio is not None else None,
        )
        log(f"开始加工 {source_id}，按数据源自动配置: "
            f"ref_audio={opts.ref_audio_ratio}"
            f"（仅配已核验同说话人；源身份={'可信' if src.has_speaker else '未核验'}），"
            f"UTMOS={opts.utmos_min or '关'}，Whisper={opts.whisper_lang or '关'}，"
            f"首尾裁切={'VAD 定边界' if opts.edge_vad else f'RMS 门限 {opts.edge_trim_ratio}'}，"
            f"不拼接短句；有可信标签的控制前缀目标={opts.control_ratio}")
        stats = pipeline.process_dataset(source_id, opts=opts, progress=log,
                                         manifest_path=manifest_path or None)
        log("加工完成，统计:\n" + json.dumps(stats, ensure_ascii=False, indent=2))
        return None

    for text in _stream("process", fn):
        yield text, _dataset_table(), _ds_choices_update(), _ds_choices_update(), \
            _ds_choices_update()


def do_mix(target_ds, target_w, zh_ds, zh_w, out_name, recipe=""):
    if not target_ds and not recipe.strip():
        yield ("请选择目标语言数据集", _dataset_table(), _ds_choices_update(),
               _ds_choices_update(), _ds_choices_update())
        return

    def fn(log):
        import json
        if recipe.strip():
            parts = [(name.strip(), float(weight)) for name, weight in
                     (line.split("=", 1) for line in recipe.splitlines() if line.strip())]
        else:
            parts = [(target_ds, float(target_w))]
            if zh_ds:
                parts.append((zh_ds, float(zh_w)))
        log(f"混合 {parts} → {out_name}")
        res = pipeline.mix_manifests(parts, out_name, progress=log)
        log("混合完成:\n" + json.dumps(res, ensure_ascii=False, indent=2))
        return None

    for text in _stream("mix", fn):
        yield text, _dataset_table(), _ds_choices_update(), _ds_choices_update(), \
            _ds_choices_update()


# ---------------- Tab 1.5: 素材导入 ----------------

def _ingest_sources() -> list[str]:
    from ..data.registry import SOURCES
    return [s.id for s in SOURCES if s.kind == "local"]


def _known_speakers(source_id: str) -> list[str]:
    """原始清单里已用过的说话人 ID；同一演员跨素材必须复用同一个 ID。"""
    p = pipeline.DATA_RAW / source_id / "manifest.jsonl"
    if not p.exists():
        return []
    return sorted({r["speaker"] for r in pipeline._read_manifest(p) if r.get("speaker")})


def _clip_info(rows: list[dict]) -> str:
    from collections import Counter, defaultdict
    if not rows:
        return "（暂无候选；先在上方切分素材）"
    by_vid: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        by_vid[r.get("ingest_video", "?")].append(r)
    lines = []
    for vid, rs in sorted(by_vid.items()):
        n = Counter(r.get("verdict") or "待定" for r in rs)
        lines.append(f"{vid}: {len(rs)} 条 | 保留 {n['保留']} / 丢弃 {n['丢弃']} / 待定 {n['待定']}"
                     f" | 听不清 {sum(1 for r in rs if r.get('transcribe_error'))}"
                     f" | 已标说话人 {len({r['speaker'] for r in rs if r.get('speaker')})}")
    return "\n".join(lines)


def _clip_panel(source_id: str, keep: int | None = None):
    """片段下拉框 + 批次统计。keep 是行号，保存后停在原位/下一条而不是跳回开头。"""
    rows = ingest.load_candidates(source_id)
    labels = [ingest.clip_label(i, r) for i, r in enumerate(rows)]
    value = labels[keep] if keep is not None and keep < len(labels) \
        else (labels[0] if labels else None)
    return gr.update(choices=labels, value=value), _clip_info(rows)


def _row_of(source_id: str, label: str) -> tuple[list[dict], int]:
    """按 label 反查行号；label 由 clip_label 生成，别自己 split。"""
    rows = ingest.load_candidates(source_id)
    labels = [ingest.clip_label(i, r) for i, r in enumerate(rows)]
    if not label or label not in labels:
        raise ValueError("请先在片段列表里选一条（列表过期就点「刷新列表」）")
    return rows, labels.index(label)


def do_refresh_ingest(source_id):
    panel, info = _clip_panel(source_id)
    return panel, info, gr.update(choices=_known_speakers(source_id))


def do_ingest(source_id, paths, video_id, min_dur, max_dur, max_items):
    if not source_id or not (paths or "").strip():
        yield "请填写数据源与远程素材路径（一行一个）", gr.update(), gr.update()
        return

    def fn(log):
        import json
        inputs = [p.strip() for p in paths.splitlines() if p.strip()]
        vid = (video_id or "").strip() or None
        if vid and len(inputs) > 1:
            raise ValueError("素材 ID 只能在单个文件时指定；多个文件请用文件名作为 ID")
        log(f"切分 {len(inputs)} 个素材 → {source_id}（{min_dur}-{max_dur}s；"
            f"PyAV 解码 → Whisper VAD 定边界 → large-v3 逐条转写 + 语种过滤）")
        summaries = [ingest.ingest_file(p, source_id, vid, float(min_dur), float(max_dur),
                                        int(max_items) if max_items else None, progress=log)
                     for p in inputs]
        log("切分完成:\n" + json.dumps(summaries, ensure_ascii=False, indent=2))
        log("下一步：选片段试听 → 标说话人/情绪 → 判定保留或丢弃 → 追加")
        return None

    text = ""
    for text in _stream("ingest", fn):
        yield text, gr.update(), gr.update()
    panel, info = _clip_panel(source_id)
    yield text, panel, info


def _clip_empty():
    return (None, "", "", gr.update(), False, "", False, "", "", False, "待定")


def do_load_clip(source_id, label):
    """选中片段 → 播放 + 回填已有标注（可反复修改）。"""
    if not label:
        return _clip_empty()
    try:
        rows, i = _row_of(source_id, label)
    except ValueError as exc:
        return (None, "", str(exc), *_clip_empty()[3:])
    r = rows[i]
    meta = (f"素材 {r.get('ingest_video')} | {r.get('start')}-{r.get('end')}s"
            f"（{float(r.get('end', 0)) - float(r.get('start', 0)):.2f}s）\n"
            f"语种 {r.get('lang')} | 转写 {r.get('transcript_source') or '—'}"
            f" | 判定 {r.get('verdict') or '待定'}\n"
            f"状态 {r.get('transcribe_error') or '通过'}\n{r.get('audio')}")
    return (r["audio"], r.get("text", ""), meta,
            gr.update(choices=_known_speakers(source_id), value=r.get("speaker") or None),
            bool(r.get("speaker_verified")), r.get("emotion", ""),
            bool(r.get("emotion_verified")), r.get("control_zh", ""), r.get("control_en", ""),
            bool(r.get("control_verified")), r.get("verdict") or "待定")


def do_save_clip(source_id, label, text, speaker, speaker_ok, emotion, emotion_ok,
                 ctrl_zh, ctrl_en, ctrl_ok, verdict):
    try:
        rows, i = _row_of(source_id, label)
        rows[i].update(ingest.validate_annotation(text, speaker, speaker_ok, emotion,
                                                  emotion_ok, ctrl_zh, ctrl_en, ctrl_ok,
                                                  verdict))
        ingest.save_candidates(source_id, rows)
        msg = f"已保存第 {i} 条（{verdict}）" + ("" if speaker_ok else "；未核实身份 → 不配 ref")
    except ValueError as exc:
        return f"标注未保存: {exc}", gr.update(), gr.update()
    panel, info = _clip_panel(source_id, keep=min(i + 1, len(rows) - 1))
    return msg, panel, info


def do_append(source_id, pin, out_name, auto, min_dur, max_dur):
    if not source_id:
        yield "请选择目标数据源", gr.update(), gr.update(), gr.update(), gr.update(), gr.update()
        return

    def fn(log):
        import json
        rows = ingest.load_candidates(source_id)
        holdout = (pin or "").replace("，", " ").replace(",", " ").split()
        res = ingest.append_to_source(source_id, rows, holdout, progress=log)
        log("追加结果:\n" + json.dumps(res, ensure_ascii=False, indent=2))
        if auto and res.get("appended"):
            name = (out_name or "").strip() or None
            log(f"重新加工 {source_id} → {name or source_id}"
                f"（钉住的素材只进验证集，不会因清单变长被重排进训练）")
            stats = pipeline.process_dataset(
                source_id, name,
                pipeline.options_for(source_id, min_dur=float(min_dur), max_dur=float(max_dur)),
                progress=log)
            log("加工完成，统计:\n" + json.dumps(stats, ensure_ascii=False, indent=2))
        return None

    for text in _stream("append", fn):
        yield text, _dataset_table(), _ds_choices_update(), _ds_choices_update(), \
            gr.update(), gr.update()
    panel, info = _clip_panel(source_id)
    yield text, _dataset_table(), _ds_choices_update(), _ds_choices_update(), panel, info


# ---------------- Tab 2: 训练 ----------------

def do_build_yaml(ftype, ds_name, r, alpha, lr, num_iters, batch_size,
                  grad_accum, save_interval, run_name, epochs=1.0, gpus=1):
    if not ds_name:
        yield "请先选择训练数据集", "", gr.update()
        return

    log = get_log("train")
    res: dict = {}

    def worker():
        try:
            ds = DATA_PROCESSED / ds_name
            rn = run_name or yaml_builder.default_run_name(ftype, str(ds / "train.jsonl"))
            base = launcher.resolve_base_path(
                env("VOXCPM_BASE_PATH") or "openbmb/VoxCPM2", progress=log)
            overrides = {
                "batch_size": int(batch_size),
                "grad_accum_steps": int(grad_accum), "save_interval": int(save_interval),
                "valid_interval": int(save_interval),
            }
            if num_iters:
                overrides.update(num_iters=int(num_iters),
                                 warmup_steps=max(1, int(num_iters) // 10))
            if ftype == "lora":
                overrides["learning_rate"] = float(lr)
                overrides["lora"] = {"r": int(r), "alpha": int(alpha)}
            else:
                overrides["learning_rate"] = float(lr)
            path = yaml_builder.build_yaml(
                rn, base, str(ds / "train.jsonl"),
                str(ds / "val.jsonl"), ftype, overrides,
                epochs=None if num_iters else float(epochs), gpus=int(gpus))
            cmd = launcher.gpu_command(path, gpus=int(gpus))
            log(f"训练配置已生成: {path}")
            res["msg"] = (
                f"run: {rn}\n基座: {base}\n配置: {path}\n\nGPU 机器上执行（可复制到远程）:\n"
                f"cd <项目路径> && {cmd}\n\n"
                f"更改 GPU 数或训练清单后请重新生成配置。", rn)
        except Exception as exc:
            log(f"生成配置失败: {exc}")
            res["msg"] = (f"失败: {exc}", "")

    t = threading.Thread(target=worker, daemon=True)
    t.start()
    while t.is_alive():
        yield log.text(), "", gr.update()
        time.sleep(0.8)
    msg, rn = res.get("msg", ("失败: 未知错误", ""))
    yield msg, rn, gr.update(choices=_config_files())


def do_start(config_path, gpus):
    tlog = get_log("train")
    try:
        if not config_path:
            yield "请先填写配置路径", ""
            return
        issues = launcher.preflight(config_path, int(gpus))
        if issues:
            for i in issues:
                tlog(f"预检: {i}")
            note = "\n".join(f"- {i}" for i in issues)
            if any(not i.startswith("警告") for i in issues):
                yield f"**预检未通过，未启动：**\n{note}", ""
                return
        log = launcher.start_local(config_path, int(gpus), progress=tlog)
        msg = f"已启动，日志实时刷新（进程退出后自动停止）: {log}"
        tlog(f"训练已启动: {config_path}（gpus={gpus}），日志 {log}")
        # 训练进程加载模型/数据需要时间，日志会稍后才出现
        while launcher.status()["running"]:
            yield msg, launcher.tail_log(log, 20)
            time.sleep(2)
        removed = launcher.cleanup_lora_runs(keep=5)
        if removed:
            tlog(f"只保留最新 5 次 LoRA 运行，已清理: {', '.join(removed)}")
            msg += f"\n\n已清理旧 LoRA 运行: {', '.join(removed)}"
        yield msg, launcher.tail_log(log, 20)
    except Exception as exc:
        tlog(f"启动失败: {exc}")
        yield f"失败: {exc}", ""


def do_refresh_log(config_path):
    if not config_path:
        return "", "未选择配置"
    import yaml
    from pathlib import Path
    cfg = yaml.safe_load(Path(config_path).read_text(encoding="utf-8"))
    return launcher.tail_log(Path(cfg["save_path"]) / "train.log", 40), \
        str(launcher.status())


def do_stop():
    get_log("train")("收到停止信号")
    return "已发送停止信号" if launcher.stop_local() else "无运行中任务"


# ---------------- Tab 3: 试听 ----------------

def do_synthesize(text, base, lora_dir, ref_audio, ref_text, cfg, steps,
                  control, seed):
    try:
        wav, secs = infer.synthesize(
            text, base or None, lora_dir if lora_dir != "（无 LoRA）" else None,
            ref_audio, ref_text, float(cfg), int(steps),
            int(seed) if seed not in (None, "") else None, control)
        note = f"（前缀 ({infer.clean_control(control)})，走 reference-only 模式）" \
            if infer.clean_control(control) else ""
        return wav, f"耗时 {secs}s {note}"
    except Exception as exc:
        return None, f"失败: {exc}"


def do_ab(text, lora_dir, ref_audio, ref_text, cfg, steps, control, seed, base=None):
    if not lora_dir or lora_dir == "（无 LoRA）":
        return None, None, "请先在上面选择要对比的 LoRA"
    try:
        (wb, sb), (wl, sl), status = infer.synthesize_ab(
            text, base or None, lora_dir, ref_audio, ref_text, float(cfg), int(steps),
            int(seed) if seed not in (None, "") else 42, control)
        return wb, wl, f"基座 {sb}s ｜ LoRA {sl}s ｜ {status}（同一文本/参考音频/种子/前缀）"
    except Exception as exc:
        return None, None, f"失败: {exc}"


# ---------------- Tab 4: 模型管理 ----------------

def do_merge(base, lora_dir, out):
    if not lora_dir:
        yield "请选择 LoRA checkpoint"
        return

    def fn(log):
        base_path = launcher.resolve_base_path(base or env("VOXCPM_BASE_PATH")
                                               or "openbmb/VoxCPM2", progress=log)
        log(f"开始合并: 基座={base_path} lora={lora_dir} → {out}")
        p = merge_lora(base_path, lora_dir, out, progress=log)
        log(f"合并完成 → {p}")
        return None

    yield from _stream("merge", fn)


def do_upload(local_dir, repo_id, kind):
    if not local_dir or not repo_id:
        yield "请填写本地目录与仓库 ID"
        return

    def fn(log):
        from pathlib import Path
        files = [f for f in Path(local_dir).rglob("*") if f.is_file()] \
            if Path(local_dir).is_dir() else []
        size_gb = sum(f.stat().st_size for f in files) / 1024**3
        log(f"上传 {local_dir} → {repo_id}（{kind}）：{len(files)} 个文件，共 {size_gb:.2f}GB")
        if not files:
            log("警告: 目录为空，请先确认路径（合并产物在「合并」的输出目录）")
        url = upload_folder(local_dir, repo_id, kind)
        log(f"上传完成: {url}")
        return None

    yield from _stream("upload", fn)


def _ckpt_choices() -> list[str]:
    out = ["（无 LoRA）"]
    out.extend(infer.list_lora_dirs())
    return out


def build_ui() -> gr.Blocks:
    _sorted_sources = sources_by_quality()
    source_choices = [s.display() for s in _sorted_sources if s.kind != "local"]

    with gr.Blocks(title="VoxCPM 微调工作台") as demo:
        gr.Markdown("## VoxCPM 2 微调工作台（端口 6006）")

        with gr.Tab("数据集") as tab_data:
            gr.Markdown("**已加工数据集**")
            ds_table = gr.Textbox(_dataset_table(), label="data/processed",
                                  lines=6, interactive=False)
            with gr.Row():
                ds_pick = gr.Dropdown(_processed_datasets(), label="预览数据集")
                ds_idx = gr.Number(0, label="样本序号", precision=0)
                ds_prev_btn = gr.Button("预览该样本（含音频播放）")
            ds_info = gr.Textbox(label="样本内容", lines=5, interactive=False)
            ds_audio = gr.Audio(label="样本音频", type="filepath")
            ds_prev_btn.click(do_preview, [ds_pick, ds_idx], [ds_info, ds_audio])
            tab_data.select(lambda: gr.update(choices=_processed_datasets()),
                            outputs=ds_pick)
            with gr.Row():
                src = gr.Dropdown(source_choices, label="数据源（含许可）")
                max_n = gr.Number(label="最大样本数（空=全量）", precision=0)
                dl_btn = gr.Button("下载", variant="primary")
            dl_out = gr.Textbox(label="下载日志（实时）", lines=10, interactive=False)
            dl_btn.click(do_download, [src, max_n], dl_out)

            gr.Markdown("---\n**加工**（16k → 裁静音 → 时长过滤 → 质检 → 表现力指标 → "
                        "已核验说话人响度对齐 → 按身份/会话切分 → 可信控制前缀 → 集合内 ref 配对；"
                        "各项按数据源自动配置，日志里可见）")
            p_manifest = gr.Textbox("", label="远程原始 JSONL 路径（可空；drama_tl/th/vi/id/ms、replay_en 在此导入）")
            with gr.Row():
                p_src = gr.Dropdown([s.id for s in _sorted_sources], label="原始数据源")
                p_min = gr.Number(3.0, label="最短时长(s)")
                p_max = gr.Number(30.0, label="最长时长(s)")
                p_val = gr.Slider(0, 0.2, 0.02, step=0.01, label="val 比例")
            with gr.Row():
                p_ctrl = gr.Slider(0, 1.0, 0.5, step=0.05,
                                   label="控制前缀目标比例（仅可信标签；其余保持裸文本）")
                p_btn = gr.Button("开始加工", variant="primary")
            p_out = gr.Textbox(label="加工日志（实时）", lines=10, interactive=False)

            gr.Markdown(
                "泰语：`thai_ser` 仅 impro + 审核后的 `yodas_th`；"
                "Tagalog：真人短剧 `drama_tl` 优先，`filipino_emotion` 仅待审候选；"
                "`filswitch` 是新闻朗读，低比例补发音。中英回放：`aishell3` / `replay_en`。\n\n"
                "越南语 / 印尼语：表现力只能靠自建 `drama_vi` / `drama_id`（无已核实的开源真人情感语料）；"
                "公开源里 `gigaspeech2_vi/id` 许可最干净（Apache-2.0）但**字段形态未核实，先 "
                "`--max-samples 20` 试跑**，`fleurs_*` / `cv22_*` 只当发音锚点。详见 `docs/vi_id_support.md`。\n\n"
                "马来语：表现力同样只能自建 `drama_ms`；`yodas2_ms`（CC-BY-3.0，YouTube 自发口语）"
                "是自然口语首选，`fleurs_ms` 的 config 名是 `ms_my`（结尾 my 是**马来西亚国家码**，"
                "不是缅甸语）。两者形态未核实，先 `--max-samples 20` 试跑。"
                "**Common Voice 22 与 gigaspeech2 都没有 ms**；`mesolitica/Malaysian-TTS` 是 "
                "F5-TTS 合成的，禁用。详见 `docs/ms_support.md`。")
            gr.Markdown("---\n**跨语言混合**（按音频时长采样；联合微调建议每个目标语种各 17% + "
                        "中文 10% + 英文 5%；训练重复上限 3×，验证集不重复；"
                        "实际占比、分语种「请求 vs 实际」对照与曝光记录在 mix.json 的 `language_shares`）")
            with gr.Row():
                m_target = gr.Dropdown(_processed_datasets(), label="目标语言数据集")
                m_tw = gr.Number(0.85, label="权重")
                m_zh = gr.Dropdown(_processed_datasets(), label="中文数据集（可空）")
                m_zw = gr.Number(0.15, label="权重")
                m_name = gr.Textbox("joint_v1", label="输出名称")
                m_btn = gr.Button("混合", variant="primary")
            m_recipe = gr.Textbox("", lines=8,
                                 label="多源时长配比（联合微调主路径；填写后替代上方两源设置）",
                                 placeholder="thai_ser_v1=6\ndrama_th_v1=3\nyodas_th_v1=7\n"
                                             "fleurs_th_v1=1\ndrama_tl_v1=9\nfilswitch_v1=5\n"
                                             "fleurs_tl_v1=3\ngigaspeech2_vi_v1=15\nfleurs_vi_v1=1\n"
                                             "cv22_vi_v1=1\ngigaspeech2_id_v1=15\nfleurs_id_v1=1\n"
                                             "cv22_id_v1=1\nyodas2_ms_v1=15\nfleurs_ms_v1=2\n"
                                             "aishell3_v2=10\nreplay_en_v2=5")
            m_out = gr.Textbox(label="混合日志", lines=6, interactive=False)

        with gr.Tab("素材导入") as tab_ingest:
            gr.Markdown("**成片 → 切分 → 转写 → 追加**（Tagalog / 越南语 / 印尼语 / 马来语都没有"
                        "已核实的可商用开源真人表演语料：Common Voice tl 官方 0 小时、"
                        "Common Voice 22 与 gigaspeech2 都**没有 ms**、YODAS 无 tl 子集、"
                        "OpenSLR 无菲律宾语资源，vi/id/ms 侧的情感语料也未能核实到任何现货，"
                        "短剧素材只能自备。目标源要按语种显式选（`drama_tl` / `drama_vi` / "
                        "`drama_id` / `drama_ms`）。有对白轨就喂对白轨；"
                        "成片混音轨靠试听淘汰 BGM 重的条目）")
            with gr.Row():
                ig_source = gr.Dropdown(_ingest_sources(),
                                        label="目标数据源（自备语料，按语种显式选）")
                ig_vid = gr.Textbox("", label="素材 ID（空=文件名；同时作为 session 与 holdout 键）")
            ig_input = gr.Textbox("", lines=3,
                                  label="远程视频/音频路径（一行一个；mp4/mkv/wav/flac）",
                                  placeholder="/root/autodl-tmp/drama/ep01.mp4")
            with gr.Row():
                ig_min = gr.Number(3.0, label="最短时长(s)")
                ig_max = gr.Number(30.0, label="最长时长(s)")
                ig_n = gr.Number(label="试跑条数（空=全部；试跑不追加）", precision=0)
                ig_run = gr.Button("切分并转写", variant="primary")
            ig_log = gr.Textbox(label="导入日志（实时）", lines=10, interactive=False)

            gr.Markdown("---\n**试听与标注**（说话人只有勾选「已核实是本人」才会写 "
                        "`speaker_verified=true`，那是 ref 配对与响度对齐的前提；"
                        "情绪与控制描述只用中英文，没核实就别勾）")
            with gr.Row():
                ig_pick = gr.Dropdown([], label="片段（素材:序号 | 时长 | 判定 | 说话人）", scale=5)
                ig_refresh = gr.Button("刷新列表", scale=1)
            ig_info = gr.Textbox(label="批次统计", lines=3, interactive=False)
            ig_audio = gr.Audio(label="片段音频", type="filepath")
            ig_text = gr.Textbox("", lines=2, label="台词（转写结果，可直接改；必须是裸台词）")
            ig_meta = gr.Textbox(label="片段信息", lines=4, interactive=False)
            with gr.Row():
                ig_spk = gr.Dropdown([], label="说话人（可手填新 ID；选源后自动加载）",
                                     allow_custom_value=True)
                ig_spk_ok = gr.Checkbox(False, label="已核实是本人")
            with gr.Row():
                ig_emo = gr.Textbox("", label="情绪标签（可空，如 surprised）")
                ig_emo_ok = gr.Checkbox(False, label="情绪已核实")
            with gr.Row():
                ig_ctrl_zh = gr.Textbox("", label="控制描述（中文，可空）")
                ig_ctrl_en = gr.Textbox("", label="控制描述（英文，可空）")
                ig_ctrl_ok = gr.Checkbox(False, label="控制描述已核实")
            with gr.Row():
                ig_verdict = gr.Radio(list(ingest.VERDICTS), value="待定", label="判定")
                ig_save = gr.Button("保存并下一条", variant="primary")

            gr.Markdown("---\n**追加到训练集**（只追加判定为「保留」且转写通过的条目；"
                        "同一素材重切会替换它上次追加的行。钉住的素材永远只进验证集，"
                        "否则追加后清单变长，随机分组会把旧验证集重排进训练集）")
            with gr.Row():
                ig_pin = gr.Textbox("", label="钉进验证集的素材 ID（空格/逗号分隔，通常只挑一集）")
                ig_out = gr.Textbox("", label="加工输出名（空=与数据源同名）")
                ig_auto = gr.Checkbox(True, label="追加后自动加工")
            ig_append = gr.Button("追加（并按需加工）", variant="primary")

            ig_run.click(do_ingest, [ig_source, ig_input, ig_vid, ig_min, ig_max, ig_n],
                         [ig_log, ig_pick, ig_info])
            ig_pick.change(do_load_clip, [ig_source, ig_pick],
                           [ig_audio, ig_text, ig_meta, ig_spk, ig_spk_ok, ig_emo,
                            ig_emo_ok, ig_ctrl_zh, ig_ctrl_en, ig_ctrl_ok, ig_verdict])
            ig_save.click(do_save_clip,
                          [ig_source, ig_pick, ig_text, ig_spk, ig_spk_ok, ig_emo, ig_emo_ok,
                           ig_ctrl_zh, ig_ctrl_en, ig_ctrl_ok, ig_verdict],
                          [ig_log, ig_pick, ig_info])
            ig_refresh.click(do_refresh_ingest, ig_source, [ig_pick, ig_info, ig_spk])
            ig_source.change(do_refresh_ingest, ig_source, [ig_pick, ig_info, ig_spk])
            ig_append.click(do_append, [ig_source, ig_pin, ig_out, ig_auto, ig_min, ig_max],
                            [ig_log, ds_table, m_target, m_zh, ig_pick, ig_info])
            tab_ingest.select(do_refresh_ingest, ig_source, [ig_pick, ig_info, ig_spk])

        with gr.Tab("训练") as tab_train:
            with gr.Row():
                ft_type = gr.Radio(["lora", "full"], value="lora", label="微调方式（推荐 LoRA）")
                ft_ds = gr.Dropdown(_processed_datasets(), label="训练数据集")
                ft_name = gr.Textbox("", label="run 名称（空=自动）")
            with gr.Row():
                ft_r = gr.Slider(8, 128, 64, step=8,
                                 label="LoRA r（64=语言风格适配，32=纯说话人适配）")
                ft_alpha = gr.Number(64, label="LoRA alpha（= r）")
                ft_lr = gr.Number(1e-4, label="学习率（LoRA=1e-4 / 全量=1e-5）")
            ft_type.change(lambda t: 1e-4 if t == "lora" else 1e-5, ft_type, ft_lr)
            p_btn.click(do_process, [p_src, p_min, p_max, p_val, p_ctrl, p_manifest],
                        [p_out, ds_table, m_target, m_zh, ft_ds])
            m_btn.click(do_mix, [m_target, m_tw, m_zh, m_zw, m_name, m_recipe],
                        [m_out, ds_table, m_target, m_zh, ft_ds])
            with gr.Row():
                ft_epochs = gr.Number(1.0, minimum=0.1, maximum=3, label="Epoch（首轮 1，最多 3）")
                ft_iters = gr.Number(0, minimum=0, precision=0, label="手动步数（0=按 epoch 自动计算）")
                ft_bs = gr.Number(2, minimum=1, precision=0, label="batch_size（音频序列长，勿调大）")
                ft_ga = gr.Number(8, minimum=1, precision=0, label="梯度累积（等效batch=bs×累积×GPU数）")
                ft_save = gr.Number(250, minimum=1, precision=0, label="保存间隔（LoRA 存档小，留多点做 A/B）")
            build_btn = gr.Button("生成训练配置", variant="primary")
            ft_out = gr.Markdown()
            run_state = gr.Textbox(visible=False)
            gr.Markdown("---\n**本机启动（需 GPU；Mac 上请复制命令到远程执行）**")
            with gr.Row():
                cfg_path = gr.Dropdown(_config_files(), label="训练配置（生成后自动出现，也可粘贴路径）",
                                       allow_custom_value=True)
                gpus = gr.Number(1, minimum=1, precision=0, label="GPU 数（生成配置/启动共用）")
                start_btn = gr.Button("启动训练")
                stop_btn = gr.Button("停止", variant="stop")
                refresh_btn = gr.Button("刷新日志")
            st_out = gr.Markdown()
            log_out = gr.Textbox(label="train.log 尾部", lines=12)
            start_btn.click(do_start, [cfg_path, gpus], [st_out, log_out])
            stop_btn.click(do_stop, outputs=st_out)
            refresh_btn.click(do_refresh_log, cfg_path, [log_out, st_out])
            build_btn.click(do_build_yaml,
                            [ft_type, ft_ds, ft_r, ft_alpha, ft_lr, ft_iters,
                             ft_bs, ft_ga, ft_save, ft_name, ft_epochs, gpus],
                            [ft_out, run_state, cfg_path])
            tab_train.select(lambda: (gr.update(choices=_config_files()),
                                      gr.update(choices=_processed_datasets())),
                             outputs=[cfg_path, ft_ds])
            gr.Markdown(f"""---
**续训**：官方脚本自动从 `save_path` 的 `latest/` 断点恢复（权重+优化器+调度器）；
重启后用同一配置重新启动即可，无需任何额外参数。SIGTERM/SIGINT 会自动保存。

**效果验证**：
1. 看曲线：wandb（已配 token 自动桥接）或 `tensorboard --logdir <save_path>/logs` ——
   `loss/diff` 应持续下降后趋平、`val/loss` 不与训练损失背离
2. 听验证音频：TensorBoard 会按 `valid_interval` 生成样本音频；每个 `save_interval`
   都保留 checkpoint，在「试听」页用不同 step 逐一 A/B 对比
3. 过拟合信号（立即回退到更早 checkpoint）：生成忽略输入文本、无论输什么都相似、
   生成停不下来（检查数据尾静音是否 >0.5s）
4. 客观对比：`uv run python -m voxft.eval base <lora_dir> --lang th`
   （`--lang` 支持 {"/".join((*TARGET_LANGS, "zh", "en"))}；th 词间无空格只有 CER，
   vi 的 WER 是音节级口径，都不与词级横向比。需 qc 组）——
   固定 case/ref/control/seed → ASR 内容误差与疑似漏尾诊断；
   **联合模型必须看 report 的 `by_lang`**：逐语种与 `eval base` 的同一份 case 对比，
   任一语种退化即算失败（归因与回退条件见 playbook 决策点 3）；
   自然度、情绪和音色由母语盲听验收，F0 起伏不是越高越好""")

        with gr.Tab("试听") as tab_listen:
            with gr.Row():
                a_base = gr.Textbox(env("VOXCPM_BASE_PATH"),
                                    label="基座（空=默认 openbmb/VoxCPM2）")
                a_lora = gr.Dropdown(_ckpt_choices(), value="（无 LoRA）", label="LoRA")
            a_text = gr.Textbox(infer.SAMPLE_TEXTS["th"], label="合成文本")
            gr.Examples(list(infer.SAMPLE_TEXTS.values()), a_text)
            with gr.Row():
                a_ref = gr.Audio(label="参考音频（可选，零样本克隆）", type="filepath")
                a_ref_text = gr.Textbox("", label="参考音频转写（可选）")
            a_ctrl = gr.Textbox(
                "", label="情绪/语气 prompt（中英文，如「愤怒地，语速快」/「sad, slow」）",
                placeholder="留空=裸文本。填了就自动走 reference-only 模式，参考音频转写会被忽略")
            gr.Examples(["愤怒地，语速快", "伤心地，轻声", "开心地，语调上扬",
                         "frustrated, holding back anger", "surprised, in disbelief"],
                        a_ctrl)
            with gr.Row():
                a_cfg = gr.Slider(1.0, 4.0, 2.0, step=0.1,
                                  label="cfg_value（偏高更贴文本但更僵，去念稿感试 1.2-1.6）")
                a_steps = gr.Slider(4, 32, 20, step=1, label="inference_timesteps")
                a_seed = gr.Number(42, label="seed（固定才可比）", precision=0)
            a_btn = gr.Button("合成", variant="primary")
            a_out = gr.Audio(label="输出（48kHz）")
            a_info = gr.Markdown()
            a_btn.click(do_synthesize,
                        [a_text, a_base, a_lora, a_ref, a_ref_text,
                         a_cfg, a_steps, a_ctrl, a_seed], [a_out, a_info])
            tab_listen.select(lambda: gr.update(choices=_ckpt_choices()),
                              outputs=a_lora)

            gr.Markdown("---\n**A/B 对比**（同一文本 + 同一参考音频，基座 vs 所选 LoRA，种子固定；验收克隆音色是否受损就用它）")
            ab_btn = gr.Button("生成 A/B 对比", variant="primary")
            with gr.Row():
                ab_base_out = gr.Audio(label="基座")
                ab_lora_out = gr.Audio(label="LoRA")
            ab_info = gr.Markdown()
            ab_btn.click(do_ab,
                         [a_text, a_lora, a_ref, a_ref_text, a_cfg, a_steps,
                          a_ctrl, a_seed, a_base],
                         [ab_base_out, ab_lora_out, ab_info])

        with gr.Tab("模型管理") as tab_mgmt:
            gr.Markdown("**Merge LoRA** → 导出完整模型目录")
            with gr.Row():
                mg_base = gr.Textbox(env("VOXCPM_BASE_PATH"), label="基座目录")
                mg_lora = gr.Dropdown(choices=_ckpt_choices()[1:], label="LoRA checkpoint")
                mg_out = gr.Textbox(str(CHECKPOINT_DIR / "merged"), label="输出目录")
            mg_btn = gr.Button("合并", variant="primary")
            mg_res = gr.Textbox(label="合并日志", lines=8, interactive=False)
            mg_btn.click(do_merge, [mg_base, mg_lora, mg_out], mg_res)

            gr.Markdown("---\n**同步到 HuggingFace**（merged 完整模型先点上面「合并」；LoRA 目录可直接上传）")
            with gr.Row():
                up_dir = gr.Dropdown(_upload_choices(), label="本地目录（选合并产物或 LoRA latest）",
                                     allow_custom_value=True)
                up_repo = gr.Textbox("FrankLiuDundun/voxcpm-finetune-lora",
                                     label="仓库 ID")
                up_kind = gr.Radio(["model", "dataset"], value="model", label="类型")
            up_btn = gr.Button("上传", variant="primary")
            up_res = gr.Textbox(label="上传日志", lines=6, interactive=False)
            up_btn.click(do_upload, [up_dir, up_repo, up_kind], up_res)
            tab_mgmt.select(lambda: (gr.update(choices=_ckpt_choices()[1:]),
                                     gr.update(choices=_upload_choices())),
                            outputs=[mg_lora, up_dir])

        with gr.Tab("日志"):
            gr.Markdown("统一运行日志 `logs/voxft.log`（下载 / 加工 / 混合 / 训练 / 合并 / 上传）")
            g_log = gr.Textbox(file_tail(), label="voxft.log 尾部 200 行",
                               lines=24, interactive=False)
            g_btn = gr.Button("刷新")
            g_btn.click(lambda: file_tail(), outputs=g_log)

    return demo


def main() -> None:
    load_dotenv()
    from ..paths import DATA_RAW
    demo = build_ui()
    demo.launch(server_port=PORT, server_name="0.0.0.0",
                allowed_paths=[str(DATA_PROCESSED), str(DATA_RAW),
                               str(CHECKPOINT_DIR)])


if __name__ == "__main__":
    main()
