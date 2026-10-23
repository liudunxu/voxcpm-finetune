"""runlog 运行记录的输入守卫自测。"""
import sys

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


def test_print_mode_announces_it_did_not_write(tmp_path, monkeypatch, capsys):
    """--print 的输出长得和落盘记录一模一样，必须自己说清"没写"，否则会把预览当成已归档。"""
    monkeypatch.setattr(runlog, "CONFIG_DIR", tmp_path)
    monkeypatch.setattr(runlog, "CHECKPOINT_DIR", tmp_path)
    monkeypatch.setattr(runlog, "RUNLOG", tmp_path / "runs.md")
    _write_cfg(tmp_path, "printed")
    monkeypatch.setattr(sys, "argv", ["runlog", "--run", "printed", "--print"])
    runlog.main()
    out = capsys.readouterr().out
    assert "## printed" in out
    assert "未写入 docs/runs.md" in out
    assert not (tmp_path / "runs.md").exists()


def test_val_loss_line_carries_subset_caveat(tmp_path, monkeypatch):
    """官方 validate() 只跑前 10 个 batch（max_val_batches）且 val_loader 也 shuffle=True，
    这个数是约 20 条随机样本的估计——记录里必须自带"跨轮不可比"的告示，否则会被当收敛曲线读。"""
    monkeypatch.setattr(runlog, "CONFIG_DIR", tmp_path)
    monkeypatch.setattr(runlog, "CHECKPOINT_DIR", tmp_path)
    _write_cfg(tmp_path, "with_val")
    log = tmp_path / "with_val" / "train.log"
    log.parent.mkdir(parents=True)
    log.write_text("[val] step 0: loss/total: 1.0416, loss/diff: 0.9151\n"
                   "[val] step 1389: loss/total: 0.9273, loss/diff: 0.9000\n", encoding="utf-8")
    text = runlog.build_record("with_val", [])
    assert "1.0416" in text and "0.9273" in text
    assert "跨轮不可比" in text and "不作选点依据" in text
