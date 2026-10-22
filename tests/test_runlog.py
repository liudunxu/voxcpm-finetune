"""runlog 运行记录的输入守卫自测。"""
import pytest
import yaml

from voxft.train import runlog


def test_build_record_refuses_missing_config(tmp_path, monkeypatch):
    """配置缺失必须硬失败：静默填默认值会把 LoRA 轮记成 finetune=full、超参全记成 None。"""
    monkeypatch.setattr(runlog, "CONFIG_DIR", tmp_path)
    monkeypatch.setattr(runlog, "CHECKPOINT_DIR", tmp_path)
    with pytest.raises(FileNotFoundError, match="缺少训练配置"):
        runlog.build_record("no_such_run", [])


def _write_cfg(tmp_path, run, extra=None):
    cfg = {"pretrained_path": "/base", "train_manifest": "", "num_iters": 10,
           "learning_rate": 1e-4}
    cfg.update(extra or {})
    (tmp_path / f"{run}.yaml").write_text(yaml.safe_dump(cfg), encoding="utf-8")


def test_finetune_label_follows_lora_block(tmp_path, monkeypatch):
    monkeypatch.setattr(runlog, "CONFIG_DIR", tmp_path)
    monkeypatch.setattr(runlog, "CHECKPOINT_DIR", tmp_path)
    _write_cfg(tmp_path, "with_lora", {"lora": {"r": 64, "alpha": 64, "dropout": 0.05}})
    _write_cfg(tmp_path, "without_lora")
    assert "| finetune | lora |" in runlog.build_record("with_lora", [])
    assert "| finetune | full |" in runlog.build_record("without_lora", [])
