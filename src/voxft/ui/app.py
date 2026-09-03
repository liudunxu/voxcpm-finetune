from __future__ import annotations

import gradio as gr

from ..paths import DATA_PROCESSED, CHECKPOINT_DIR, env, load_dotenv
from ..data.registry import SOURCES
from ..data import download, pipeline
from ..train import launcher, yaml_builder
from ..lora.merge import merge_lora, find_checkpoints, is_lora_dir
from ..hub.sync import upload_folder
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


# ---------------- Tab 1: 数据集 ----------------

def do_download(source_id, max_samples):
    try:
        dest = download.download_source(source_id,
                                        int(max_samples) if max_samples else None)
        return f"完成 → {dest}/manifest.jsonl"
    except Exception as exc:
        return f"失败: {exc}"


def do_process(source_id, min_dur, max_dur, ref_ratio, val_ratio,
               use_utmos, utmos_min, whisper_lang):
    try:
        opts = pipeline.Options(
            min_dur=float(min_dur), max_dur=float(max_dur),
            ref_audio_ratio=float(ref_ratio), val_ratio=float(val_ratio),
            utmos_min=float(utmos_min) if use_utmos else None,
            whisper_lang=whisper_lang or None,
        )
        import json
        return json.dumps(pipeline.process_dataset(source_id, opts=opts),
                          ensure_ascii=False, indent=2), _dataset_table()
    except Exception as exc:
        return f"失败: {exc}", _dataset_table()


def do_mix(target_ds, target_w, zh_ds, zh_w, out_name):
    try:
        parts = [(target_ds, float(target_w))]
        if zh_ds:
            parts.append((zh_ds, float(zh_w)))
        import json
        return json.dumps(pipeline.mix_manifests(parts, out_name),
                          ensure_ascii=False, indent=2), _dataset_table()
    except Exception as exc:
        return f"失败: {exc}", _dataset_table()


# ---------------- Tab 2: 训练 ----------------

def do_build_yaml(ftype, ds_name, r, alpha, lr, num_iters, batch_size,
                  grad_accum, save_interval, run_name):
    try:
        ds = DATA_PROCESSED / ds_name
        rn = run_name or yaml_builder.default_run_name(ftype)
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
            rn, env("VOXCPM_BASE_PATH"), str(ds / "train.jsonl"),
            str(ds / "val.jsonl"), ftype, overrides)
        cmd = launcher.gpu_command(path, gpus=1)
        return (f"run: {rn}\n配置: {path}\n\nGPU 机器上执行（可复制到远程）:\n"
                f"cd <项目路径> && {cmd}\n\n"
                f"多卡: 把 python 换成 torchrun --nproc_per_node=N"), rn
    except Exception as exc:
        return f"失败: {exc}", ""


def do_start(config_path, gpus):
    try:
        log = launcher.start_local(config_path, int(gpus))
        return f"已启动，日志: {log}", launcher.tail_log(log, 20)
    except Exception as exc:
        return f"失败: {exc}", ""


def do_refresh_log(config_path):
    if not config_path:
        return "", "未选择配置"
    import yaml
    from pathlib import Path
    cfg = yaml.safe_load(Path(config_path).read_text(encoding="utf-8"))
    return launcher.tail_log(Path(cfg["save_path"]) / "train.log", 40), \
        str(launcher.status())


def do_stop():
    return "已发送停止信号" if launcher.stop_local() else "无运行中任务"


# ---------------- Tab 3: 试听 ----------------

def do_synthesize(text, base, lora_dir, ref_audio, ref_text, cfg, steps):
    try:
        wav, secs = infer.synthesize(
            text, base or None, lora_dir if lora_dir != "（无 LoRA）" else None,
            ref_audio, ref_text, float(cfg), int(steps))
        return wav, f"耗时 {secs}s"
    except Exception as exc:
        return None, f"失败: {exc}"


# ---------------- Tab 4: 模型管理 ----------------

def do_merge(base, lora_dir, out):
    try:
        p = merge_lora(base or env("VOXCPM_BASE_PATH"), lora_dir, out)
        return f"完成 → {p}"
    except Exception as exc:
        return f"失败: {exc}"


def do_upload(local_dir, repo_id, kind):
    try:
        return upload_folder(local_dir, repo_id, kind)
    except Exception as exc:
        return f"失败: {exc}"


def _ckpt_choices() -> list[str]:
    out = ["（无 LoRA）"]
    out.extend(infer.list_lora_dirs())
    return out


def build_ui() -> gr.Blocks:
    source_choices = [f"{s.id} — {s.label} [{s.license}]" for s in SOURCES]

    with gr.Blocks(title="VoxCPM 微调工作台") as demo:
        gr.Markdown("## VoxCPM 2 微调工作台（端口 6006）")

        with gr.Tab("数据集"):
            gr.Markdown("**已加工数据集**")
            ds_table = gr.Textbox(_dataset_table(), label="data/processed",
                                  lines=6, interactive=False)
            with gr.Row():
                src = gr.Dropdown(source_choices, label="数据源（含许可）")
                max_n = gr.Number(label="最大样本数（空=全量）", precision=0)
                dl_btn = gr.Button("下载", variant="primary")
            dl_out = gr.Textbox(label="下载结果")
            dl_btn.click(lambda s, n: do_download(s.split(" — ")[0], n),
                         [src, max_n], dl_out)

            gr.Markdown("---\n**加工**（16k 重采样 → 裁尾静音 → 响度归一化 → 时长过滤 → 可选质检 → ref_audio 配对 → 切分）")
            with gr.Row():
                p_src = gr.Dropdown([s.id for s in SOURCES], label="原始数据源")
                p_min = gr.Number(3.0, label="最短时长(s)")
                p_max = gr.Number(30.0, label="最长时长(s)")
                p_ref = gr.Slider(0, 1, 0.4, step=0.05, label="ref_audio 比例")
                p_val = gr.Slider(0, 0.2, 0.02, step=0.01, label="val 比例")
            with gr.Row():
                p_utmos_on = gr.Checkbox(False, label="UTMOS 质检")
                p_utmos = gr.Slider(2.5, 4.5, 3.5, step=0.1, label="UTMOS 阈值")
                p_wlang = gr.Dropdown(["", "th", "tl", "zh"], value="",
                                      label="Whisper 转写校验语言（空=关闭，需 --group qc）")
                p_btn = gr.Button("开始加工", variant="primary")
            p_out = gr.Textbox(label="加工统计", lines=8)
            p_btn.click(do_process,
                        [p_src, p_min, p_max, p_ref, p_val,
                         p_utmos_on, p_utmos, p_wlang],
                        [p_out, ds_table])

            gr.Markdown("---\n**跨语言混合**（目标语言为主 + 中文 10-20% 防遗忘）")
            with gr.Row():
                m_target = gr.Dropdown(_processed_datasets(), label="目标语言数据集")
                m_tw = gr.Number(0.85, label="权重")
                m_zh = gr.Dropdown(_processed_datasets(), label="中文数据集（可空）")
                m_zw = gr.Number(0.15, label="权重")
                m_name = gr.Textbox("mixed_th_zh", label="输出名称")
                m_btn = gr.Button("混合", variant="primary")
            m_out = gr.Textbox(label="混合结果", lines=5)
            m_btn.click(do_mix, [m_target, m_tw, m_zh, m_zw, m_name],
                        [m_out, ds_table])

        with gr.Tab("训练"):
            with gr.Row():
                ft_type = gr.Radio(["lora", "full"], value="lora", label="微调方式（推荐 LoRA）")
                ft_ds = gr.Dropdown(_processed_datasets(), label="训练数据集")
                ft_name = gr.Textbox("", label="run 名称（空=自动）")
            with gr.Row():
                ft_r = gr.Slider(8, 128, 32, step=8, label="LoRA r（语言适配建议 64）")
                ft_alpha = gr.Number(32, label="LoRA alpha（一般 = r）")
                ft_lr = gr.Number(1e-4, label="学习率（全量建议 1e-5）")
            with gr.Row():
                ft_iters = gr.Number(1000, label="训练步数")
                ft_bs = gr.Number(16, label="batch_size")
                ft_ga = gr.Number(1, label="梯度累积")
                ft_save = gr.Number(500, label="保存间隔")
            build_btn = gr.Button("生成训练配置", variant="primary")
            ft_out = gr.Markdown()
            run_state = gr.Textbox(visible=False)
            build_btn.click(do_build_yaml,
                            [ft_type, ft_ds, ft_r, ft_alpha, ft_lr, ft_iters,
                             ft_bs, ft_ga, ft_save, ft_name],
                            [ft_out, run_state])
            gr.Markdown("---\n**本机启动（需 GPU；Mac 上请复制命令到远程执行）**")
            with gr.Row():
                cfg_path = gr.Textbox(label="配置路径（粘贴上面生成的）")
                gpus = gr.Number(1, label="GPU 数")
                start_btn = gr.Button("启动训练")
                stop_btn = gr.Button("停止", variant="stop")
                refresh_btn = gr.Button("刷新日志")
            st_out = gr.Markdown()
            log_out = gr.Textbox(label="train.log 尾部", lines=12)
            start_btn.click(do_start, [cfg_path, gpus], [st_out, log_out])
            stop_btn.click(do_stop, outputs=st_out)
            refresh_btn.click(do_refresh_log, cfg_path, [log_out, st_out])
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
4. 最终客观复核：用 Whisper 对生成音频转写，与输入文本算相似度（文本贴合度）""")

        with gr.Tab("试听"):
            with gr.Row():
                a_base = gr.Textbox(env("VOXCPM_BASE_PATH"),
                                    label="基座（空=默认 openbmb/VoxCPM2）")
                a_lora = gr.Dropdown(_ckpt_choices(), value="（无 LoRA）", label="LoRA")
            a_text = gr.Textbox(infer.SAMPLE_TEXTS["泰语"], label="合成文本")
            gr.Examples(list(infer.SAMPLE_TEXTS.values()), a_text)
            with gr.Row():
                a_ref = gr.Audio(label="参考音频（可选，零样本克隆）", type="filepath")
                a_ref_text = gr.Textbox("", label="参考音频转写（可选）")
            with gr.Row():
                a_cfg = gr.Slider(1.0, 4.0, 2.0, step=0.1, label="cfg_value")
                a_steps = gr.Slider(4, 32, 10, step=1, label="inference_timesteps")
            a_btn = gr.Button("合成", variant="primary")
            a_out = gr.Audio(label="输出（48kHz）")
            a_info = gr.Markdown()
            a_btn.click(do_synthesize,
                        [a_text, a_base, a_lora, a_ref, a_ref_text,
                         a_cfg, a_steps], [a_out, a_info])

        with gr.Tab("模型管理"):
            gr.Markdown("**Merge LoRA** → 导出完整模型目录")
            with gr.Row():
                mg_base = gr.Textbox(env("VOXCPM_BASE_PATH"), label="基座目录")
                mg_lora = gr.Dropdown(choices=_ckpt_choices()[1:], label="LoRA checkpoint")
                mg_out = gr.Textbox("checkpoints/merged", label="输出目录")
            mg_btn = gr.Button("合并", variant="primary")
            mg_res = gr.Markdown()
            mg_btn.click(do_merge, [mg_base, mg_lora, mg_out], mg_res)

            gr.Markdown("---\n**同步到 HuggingFace**")
            with gr.Row():
                up_dir = gr.Textbox(label="本地目录")
                up_repo = gr.Textbox("FrankLiuDundun/voxcpm-finetune-lora",
                                     label="仓库 ID")
                up_kind = gr.Radio(["model", "dataset"], value="model", label="类型")
            up_btn = gr.Button("上传", variant="primary")
            up_res = gr.Markdown()
            up_btn.click(do_upload, [up_dir, up_repo, up_kind], up_res)

    return demo


def main() -> None:
    load_dotenv()
    demo = build_ui()
    demo.launch(server_port=PORT, server_name="0.0.0.0")


if __name__ == "__main__":
    main()
