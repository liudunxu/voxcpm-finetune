from __future__ import annotations

import os
import shlex
import subprocess
import sys
from pathlib import Path

from ..paths import CHECKPOINT_DIR, VOXCPM_REPO

TRAIN_SCRIPT = VOXCPM_REPO / "scripts" / "train_voxcpm_finetune.py"


def gpu_command(config_path: str | Path, gpus: int = 1,
                cuda_devices: str | None = None) -> str:
    """生成 GPU 机器上的训练命令（本地无 CUDA 时复制到远程执行）。"""
    if not TRAIN_SCRIPT.exists():
        raise FileNotFoundError(f"官方训练脚本不存在: {TRAIN_SCRIPT}（submodule 未初始化？）")
    if not isinstance(gpus, int) or gpus < 1:
        raise ValueError("GPU 数必须为正整数")
    script_args = f"{shlex.quote(str(TRAIN_SCRIPT))} --config_path {shlex.quote(str(Path(config_path).resolve()))}"
    if gpus > 1:
        cmd = f"torchrun --nproc_per_node={gpus} {script_args}"
    else:
        cmd = f"python {script_args}"
    cmd = f"PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True {cmd}"
    if cuda_devices:
        cmd = f"CUDA_VISIBLE_DEVICES={shlex.quote(cuda_devices)} {cmd}"
    return cmd


def resolve_base_path(path: str, progress=None) -> str:
    """基座路径归一化：本地目录直接用；HF 仓库 ID 则下载快照后返回本地目录。

    官方训练脚本要求 pretrained_path 是含 config.json 的本地目录。
    """
    if not path or Path(path).is_dir():
        return path
    import threading

    from huggingface_hub import HfApi, snapshot_download
    from huggingface_hub.constants import HF_HUB_CACHE

    from ..log import _fmt_size
    from ..paths import env
    token = env("HF_TOKEN") or None
    if progress:
        import os
        progress(f"基座模型本地不存在，开始下载 {path}（模型较大，请耐心等待；"
                 f"endpoint={os.environ.get('HF_ENDPOINT') or 'https://huggingface.co'}）")
    cache_dir = Path(HF_HUB_CACHE) / f"models--{path.replace('/', '--')}"

    def _dir_size(d: Path) -> int:
        return sum(f.stat().st_size for f in d.rglob("*") if f.is_file()) \
            if d.exists() else 0

    stop = threading.Event()
    total = 0

    def watcher():
        # tqdm_class 不会传给单文件下载，聚合条对超大文件无输出，直接轮询目录大小
        while not stop.wait(5):
            try:
                done = _dir_size(cache_dir)
                if total:
                    progress(f"基座下载中: {_fmt_size(done)}/{_fmt_size(total)}"
                             f" ({min(100, 100 * done / total):.0f}%)")
                else:
                    progress(f"基座下载中: {_fmt_size(done)}")
            except Exception as exc:
                progress(f"进度监控异常（不影响下载）: {exc}")

    t = threading.Thread(target=watcher, daemon=True) if progress else None
    if t:
        progress("下载进度监控已启动（每 5 秒更新）")
        t.start()
    try:
        if progress:
            try:
                info = HfApi(token=token).repo_info(path, files_metadata=True)
                total = sum(s.size or 0 for s in info.siblings)
                progress(f"基座总大小: {_fmt_size(total)}")
            except Exception:
                pass
        local = snapshot_download(path, token=token)
    finally:
        stop.set()
    if progress:
        progress(f"基座模型就绪 → {local}")
    return local


def preflight(config_path: str | Path, gpus: int | None = None) -> list[str]:
    """训练前预检；返回问题列表（空 = 通过）。把报错提前到启动前。"""
    import json
    import yaml

    issues: list[str] = []
    cfg_path = Path(config_path)
    if not cfg_path.exists():
        return [f"配置文件不存在: {cfg_path}"]
    try:
        cfg = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))
        if not isinstance(cfg, dict):
            raise ValueError("YAML 必须为对象")
    except (ValueError, yaml.YAMLError) as exc:
        return [f"训练配置无效: {exc}"]

    if not TRAIN_SCRIPT.exists():
        issues.append(f"官方训练脚本不存在: {TRAIN_SCRIPT}（执行过 git submodule update --init？）")

    memberships = {}
    counts = {}
    controls = refs = 0
    for key, required in (("train_manifest", True), ("val_manifest", False)):
        members = memberships[key] = set()
        counts[key] = 0
        p = str(cfg.get(key, "") or "")
        if not p:
            if required:
                issues.append(f"{key} 未配置")
            continue
        mp = Path(p)
        if not mp.exists():
            issues.append(f"{key} 不存在: {mp}")
            continue
        with mp.open(encoding="utf-8") as f:
            for i, line in enumerate(f):
                if not line.strip():
                    continue
                counts[key] += 1
                try:
                    rec = json.loads(line)
                    if not isinstance(rec, dict):
                        raise ValueError("每行必须为对象")
                except Exception:
                    issues.append(f"{key} 第 {i + 1} 行不是合法 JSON")
                    continue
                if not isinstance(rec.get("audio"), str) or not Path(rec["audio"]).is_file():
                    issues.append(f"{key} 第 {i + 1} 行音频不存在: {rec.get('audio')}")
                    continue
                if not isinstance(rec.get("text"), str) or not rec["text"].strip():
                    issues.append(f"{key} 第 {i + 1} 行缺少训练文本")
                if key == "train_manifest":
                    controls += isinstance(rec.get("text"), str) and rec["text"].startswith("(")
                    refs += bool(rec.get("ref_audio"))
                for field in ("audio", "ref_audio", "origin_audio", "ref_origin_audio"):
                    if rec.get(field):
                        if isinstance(rec[field], str):
                            members.add(("audio", str(Path(rec[field]).resolve())))
                        else:
                            issues.append(f"{key} 第 {i + 1} 行 {field} 必须为路径字符串")
                if rec.get("speaker_verified") is True:
                    members.add(("speaker", rec.get("speaker")))
                if rec.get("session"):
                    members.add(("session", rec.get("source_id", ""), rec["session"]))
                if rec.get("ref_audio"):
                    if not isinstance(rec["ref_audio"], str):
                        continue
                    if not Path(rec["ref_audio"]).is_file():
                        issues.append(f"{key} 第 {i + 1} 行参考音频不存在")
                    if (rec.get("speaker_verified") is not True or not rec.get("speaker")
                            or rec.get("ref_speaker") != rec["speaker"]):
                        issues.append(f"{key} 第 {i + 1} 行 ref 缺少已验证的同人身份，请重新加工")
                    if rec["ref_audio"] == rec.get("audio"):
                        issues.append(f"{key} 第 {i + 1} 行 ref 与目标音频相同")
        if not counts[key]:
            issues.append(f"{key} 为空；无验证集时应将 val_manifest 留空")
    if memberships["train_manifest"] & memberships["val_manifest"]:
        issues.append("训练/验证集共享音频、ref、说话人或会话，请先按组重新加工")
    if counts["train_manifest"]:
        if controls / counts["train_manifest"] < 0.25:
            issues.append("警告：带前缀目标不足 25%，请检查真实标签覆盖，勿用猜测标签补齐")
        if refs / counts["train_manifest"] < 0.3:
            issues.append("警告：同人 ref 覆盖不足 30%，请补已核验同人录音；不要强配未知身份")

    pre = str(cfg.get("pretrained_path", "") or "")
    if not pre:
        issues.append("pretrained_path 未配置")
    if pre and not Path(pre).is_dir():
        if pre.count("/") == 1:
            issues.append(f"pretrained_path 是 HF 仓库 ID（{pre}），训练脚本要求本地目录："
                          "请在页面重新「生成训练配置」（会自动下载基座），"
                          "或在 .env 设置 VOXCPM_BASE_PATH 指向本地基座目录")
        else:
            issues.append(f"pretrained_path 既非本地目录也非 HF 仓库 ID（owner/name）: {pre}")
    elif pre:
        model_config = Path(pre) / "config.json"
        if not model_config.exists():
            issues.append("基座缺少 config.json")
        else:
            model_cfg = json.loads(model_config.read_text(encoding="utf-8"))
            rate = model_cfg.get("dit_config", {}).get("cfm_config", {}).get("training_cfg_rate", 0.1)
            if rate != 0.1:
                issues.append(f"基座 training_cfg_rate={rate}，本项目要求保留 0.1")
    if "training_cfg_rate" in cfg:
        issues.append("training_cfg_rate 不能放在训练 YAML 顶层，请检查基座模型配置")
    plan_path = cfg_path.with_suffix(".plan.json")
    if plan_path.exists() and gpus is not None:
        plan = json.loads(plan_path.read_text(encoding="utf-8"))
        if plan.get("epochs") is not None and plan.get("gpus") != gpus:
            issues.append("GPU 数与按 epoch 生成配置时不同，请按实际 GPU 数重新生成配置")
        if plan.get("epochs") is not None and plan.get("train_samples") != counts["train_manifest"]:
            issues.append("训练条数与生成计划时不同，请重新生成配置")
    effective_batch = cfg.get("batch_size", 2) * cfg.get("grad_accum_steps", 8) * (gpus or 1)
    if counts["train_manifest"] and cfg.get("num_iters", 0) * effective_batch > 3 * counts["train_manifest"]:
        issues.append("警告：预计训练超过 3 epoch，小语料还需计入混合重复曝光")

    save = Path(cfg.get("save_path", ""))
    try:
        save.mkdir(parents=True, exist_ok=True)
    except Exception as exc:
        issues.append(f"save_path 不可写: {save} ({exc})")

    try:
        import torch
        if not torch.cuda.is_available():
            issues.append("警告：本机无 CUDA，训练请在 GPU 服务器执行（本机仅生成命令）")
    except Exception:
        pass
    return issues


_PROC: subprocess.Popen | None = None


def start_local(config_path: str | Path, gpus: int = 1, progress=None) -> Path:
    """在本机（需有 GPU）以后台子进程启动训练，日志写入 run 目录下 train.log。"""
    global _PROC
    if _PROC is not None and _PROC.poll() is None:
        raise RuntimeError("已有训练任务在运行；先停止或等待其结束")
    issues = preflight(config_path, gpus)
    fatal = [i for i in issues if not i.startswith("警告")]
    if fatal:
        raise RuntimeError("预检未通过：\n" + "\n".join(issues))
    import yaml
    cfg = yaml.safe_load(Path(config_path).read_text(encoding="utf-8"))
    log_path = Path(cfg["save_path"]) / "train.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    cmd = gpu_command(config_path, gpus)
    # ponytail: 单实例全局锁（_PROC），多任务并行时改为按 run 字典管理
    _PROC = subprocess.Popen(
        cmd, shell=True, cwd=VOXCPM_REPO,
        stdout=log_path.open("a"), stderr=subprocess.STDOUT,
        start_new_session=True,
        env={**os.environ, "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True"},
    )
    from .tb_wandb_bridge import start_bridge
    start_bridge(cfg.get("tensorboard", ""), Path(config_path).stem, progress=progress)
    return log_path


def stop_local() -> bool:
    global _PROC
    if _PROC is None or _PROC.poll() is not None:
        return False
    _PROC.terminate()
    try:
        _PROC.wait(timeout=30)
    except subprocess.TimeoutExpired:
        _PROC.kill()
    _PROC = None
    return True


def status() -> dict:
    running = _PROC is not None and _PROC.poll() is None
    return {"running": running, "returncode": None if running else
            (_PROC.returncode if _PROC is not None else None)}


def tail_log(log_path: str | Path, lines: int = 30) -> str:
    p = Path(log_path)
    if not p.exists():
        return "（日志尚未生成）"
    return "".join(p.read_text(encoding="utf-8", errors="replace")
                   .splitlines(keepends=True)[-lines:])


def list_runs() -> list[str]:
    if not CHECKPOINT_DIR.exists():
        return []
    return sorted(d.name for d in CHECKPOINT_DIR.iterdir() if d.is_dir())


def cleanup_lora_runs(keep: int = 5) -> list[str]:
    """只保留最近 keep 次 LoRA 运行，删除更早的；返回被删目录。

    按 latest/ 的修改时间排序；全量微调目录（无 lora 配置）不受影响。
    """
    import shutil

    from ..lora.merge import is_lora_dir
    runs = [d for d in (CHECKPOINT_DIR.iterdir() if CHECKPOINT_DIR.exists() else [])
            if d.is_dir() and is_lora_dir(d / "latest")]
    runs.sort(key=lambda d: (d / "latest").stat().st_mtime, reverse=True)
    removed = []
    for d in runs[keep:]:
        shutil.rmtree(d, ignore_errors=True)
        removed.append(str(d))
    return removed


if __name__ == "__main__":
    if len(sys.argv) > 1:
        print(gpu_command(sys.argv[1], int(sys.argv[2]) if len(sys.argv) > 2 else 1))
    else:
        print("用法: python -m voxft.train.launcher <config.yaml> [gpus]")
