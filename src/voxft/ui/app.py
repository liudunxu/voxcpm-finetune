from __future__ import annotations

import threading
import time

import gradio as gr

from ..paths import DATA_PROCESSED, CHECKPOINT_DIR, env, load_dotenv
from ..data.registry import SOURCES
from ..data import download, pipeline
from ..train import launcher, yaml_builder
from ..lora.merge import merge_lora, find_checkpoints, is_lora_dir
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
        sid = source_id.split(" — ")[0]
        log(f"开始下载 {sid}（max_samples={max_samples or '全量'}）")
        dest = download.download_source(sid,
                                        int(max_samples) if max_samples else None,
                                        progress=log)
        log(f"完成 → {dest}/manifest.jsonl")
        return None

    yield from _stream("download", fn)


def do_process(source_id, min_dur, max_dur, val_ratio,
               control_ratio=None, min_snr=None):
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
            min_snr_db=float(min_snr) if min_snr else None,
        )
        log(f"开始加工 {source_id}，按数据源自动配置: "
            f"ref_audio={opts.ref_audio_ratio}"
            f"（{'有说话人列' if src.has_speaker else ('聚类伪说话人' if src.pseudo_speaker else '无说话人')}），"
            f"UTMOS={opts.utmos_min or '关'}，Whisper={opts.whisper_lang or '关'}，"
            f"短句拼接={'~' + str(opts.concat_target) + 's' if opts.concat_target else '关'}，"
            f"控制前缀={opts.control_ratio}，SNR 门限={opts.min_snr_db or '关'}")
        stats = pipeline.process_dataset(source_id, opts=opts, progress=log)
        log("加工完成，统计:\n" + json.dumps(stats, ensure_ascii=False, indent=2))
        return None

    for text in _stream("process", fn):
        yield text, _dataset_table(), _ds_choices_update(), _ds_choices_update(), \
            _ds_choices_update()


def do_mix(target_ds, target_w, zh_ds, zh_w, out_name):
    if not target_ds:
        yield ("请选择目标语言数据集", _dataset_table(), _ds_choices_update(),
               _ds_choices_update(), _ds_choices_update())
        return

    def fn(log):
        import json
        parts = [(target_ds, float(target_w))]
        if zh_ds:
            parts.append((zh_ds, float(zh_w)))
        log(f"混合 {parts} → {out_name}")
        res = pipeline.mix_manifests(parts, out_name)
        log("混合完成:\n" + json.dumps(res, ensure_ascii=False, indent=2))
        return None

    for text in _stream("mix", fn):
        yield text, _dataset_table(), _ds_choices_update(), _ds_choices_update(), \
            _ds_choices_update()


# ---------------- Tab 2: 训练 ----------------

def do_build_yaml(ftype, ds_name, r, alpha, lr, num_iters, batch_size,
                  grad_accum, save_interval, run_name):
    if not ds_name:
        yield "请先选择训练数据集", "", gr.update()
        return

    log = get_log("train")
    res: dict = {}

    def worker():
        try:
            ds = DATA_PROCESSED / ds_name
            rn = run_name or yaml_builder.default_run_name(ftype)
            base = launcher.resolve_base_path(
                env("VOXCPM_BASE_PATH") or "openbmb/VoxCPM2", progress=log)
            overrides = {
                "num_iters": int(num_iters), "batch_size": int(batch_size),
                "grad_accum_steps": int(grad_accum), "save_interval": int(save_interval),
                "valid_interval": int(save_interval),
                "warmup_steps": max(10, int(num_iters) // 10),
            }
            if ftype == "lora":
                overrides["learning_rate"] = float(lr)
                overrides["lora"] = {"r": int(r), "alpha": int(alpha)}
            else:
                overrides["learning_rate"] = float(lr)
            path = yaml_builder.build_yaml(
                rn, base, str(ds / "train.jsonl"),
                str(ds / "val.jsonl"), ftype, overrides)
            cmd = launcher.gpu_command(path, gpus=1)
            log(f"训练配置已生成: {path}")
            res["msg"] = (
                f"run: {rn}\n基座: {base}\n配置: {path}\n\nGPU 机器上执行（可复制到远程）:\n"
                f"cd <项目路径> && {cmd}\n\n"
                f"多卡: 把 python 换成 torchrun --nproc_per_node=N", rn)
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
        issues = launcher.preflight(config_path)
        if issues:
            for i in issues:
                tlog(f"预检: {i}")
            note = "\n".join(f"- {i}" for i in issues)
            if any(not i.startswith("警告") for i in issues):
                yield f"**预检未通过，未启动：**\n{note}", ""
                return
        log = launcher.start_local(config_path, int(gpus))
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


def do_ab(text, lora_dir, ref_audio, ref_text, cfg, steps, control, seed):
    if not lora_dir or lora_dir == "（无 LoRA）":
        return None, None, "请先在上面选择要对比的 LoRA"
    try:
        (wb, sb), (wl, sl), status = infer.synthesize_ab(
            text, None, lora_dir, ref_audio, ref_text, float(cfg), int(steps),
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
        p = merge_lora(base_path, lora_dir, out)
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
    _role_order = {"expressive": 0, "anchor": 1, "antiforget": 2}
    _sorted_sources = sorted(
        SOURCES, key=lambda s: (s.lang, _role_order.get(s.role, 9), not s.preferred))
    source_choices = [s.display() for s in _sorted_sources]

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
                        "伪说话人聚类 → 按说话人响度对齐 → 控制前缀 → ref_audio 配对 → 切分；"
                        "各项按数据源自动配置，日志里可见）")
            with gr.Row():
                p_src = gr.Dropdown([s.id for s in _sorted_sources], label="原始数据源")
                p_min = gr.Number(3.0, label="最短时长(s)")
                p_max = gr.Number(30.0, label="最长时长(s)")
                p_val = gr.Slider(0, 0.2, 0.02, step=0.01, label="val 比例")
            with gr.Row():
                p_ctrl = gr.Slider(0, 1.0, 0.5, step=0.05,
                                   label="控制前缀比例（0=全裸文本，会冲掉基座的情绪 prompt 能力）")
                p_snr = gr.Number(0, label="SNR 门限 dB（0=关，建议 12）")
                p_btn = gr.Button("开始加工", variant="primary")
            p_out = gr.Textbox(label="加工日志（实时）", lines=10, interactive=False)

            gr.Markdown(
                "首选源：泰语表现力 `thai_ser` / 泰语口语锚点 `yodas_th`；"
                "Tagalog 表现力 `filipino_emotion` / Taglish 锚点 `filswitch`；"
                "中文防遗忘 `aishell3`。表现力语料占目标语言的 30-50%。")
            gr.Markdown("---\n**跨语言混合**（目标语言为主 + 中文 10-20% 防遗忘；"
                        "小语料重复上限 3×，超出会自动缩水并记在 mix.json）")
            with gr.Row():
                m_target = gr.Dropdown(_processed_datasets(), label="目标语言数据集")
                m_tw = gr.Number(0.85, label="权重")
                m_zh = gr.Dropdown(_processed_datasets(), label="中文数据集（可空）")
                m_zw = gr.Number(0.15, label="权重")
                m_name = gr.Textbox("mixed_th_zh", label="输出名称")
                m_btn = gr.Button("混合", variant="primary")
            m_out = gr.Textbox(label="混合日志", lines=6, interactive=False)

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
            p_btn.click(do_process, [p_src, p_min, p_max, p_val, p_ctrl, p_snr],
                        [p_out, ds_table, m_target, m_zh, ft_ds])
            m_btn.click(do_mix, [m_target, m_tw, m_zh, m_zw, m_name],
                        [m_out, ds_table, m_target, m_zh, ft_ds])
            with gr.Row():
                ft_iters = gr.Number(1000, label="训练步数")
                ft_bs = gr.Number(2, label="batch_size（音频序列长，勿调大）")
                ft_ga = gr.Number(8, label="梯度累积（等效batch=bs×累积）")
                ft_save = gr.Number(250, label="保存间隔（LoRA 存档小，留多点做 A/B）")
            build_btn = gr.Button("生成训练配置", variant="primary")
            ft_out = gr.Markdown()
            run_state = gr.Textbox(visible=False)
            gr.Markdown("---\n**本机启动（需 GPU；Mac 上请复制命令到远程执行）**")
            with gr.Row():
                cfg_path = gr.Dropdown(_config_files(), label="训练配置（生成后自动出现，也可粘贴路径）",
                                       allow_custom_value=True)
                gpus = gr.Number(1, label="GPU 数")
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
                             ft_bs, ft_ga, ft_save, ft_name],
                            [ft_out, run_state, cfg_path])
            tab_train.select(lambda: gr.update(choices=_config_files()),
                             outputs=cfg_path)
            gr.Markdown("""---
**续训**：官方脚本自动从 `save_path` 的 `latest/` 断点恢复（权重+优化器+调度器）；
重启后用同一配置重新启动即可，无需任何额外参数。SIGTERM/SIGINT 会自动保存。

**效果验证**：
1. 看曲线：wandb（已配 token 自动桥接）或 `tensorboard --logdir <save_path>/logs` ——
   `loss/diff` 应持续下降后趋平、`val/loss` 不与训练损失背离
2. 听验证音频：TensorBoard 会按 `valid_interval` 生成样本音频；每个 `save_interval`
   都保留 checkpoint，在「试听」页用不同 step 逐一 A/B 对比
3. 过拟合信号（立即回退到更早 checkpoint）：生成忽略输入文本、无论输什么都相似、
   生成停不下来（检查数据尾静音是否 >0.5s）
4. 客观对比：`uv run python -m voxft.eval base <lora_dir> --lang th`（需 qc 组）——
   批量合成→Whisper 转写→输出各 checkpoint 的平均文本贴合度排名""")

        with gr.Tab("试听") as tab_listen:
            with gr.Row():
                a_base = gr.Textbox(env("VOXCPM_BASE_PATH"),
                                    label="基座（空=默认 openbmb/VoxCPM2）")
                a_lora = gr.Dropdown(_ckpt_choices(), value="（无 LoRA）", label="LoRA")
            a_text = gr.Textbox(infer.SAMPLE_TEXTS["泰语"], label="合成文本")
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
                          a_ctrl, a_seed],
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
