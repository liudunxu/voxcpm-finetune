"""runlog 运行记录的输入守卫自测。"""
import json
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


def test_seed_severity_breakdown_splits_moderate_and_catastrophic():
    """r26 的教训：按多读条数它比 r25 差（4/96 vs 2/96），按严重度它更好（全 ≤2.00× vs 一条 7.97×）。
    分解只是参考输出，不改任何门禁的语义与阈值。"""
    rep = {"label": "ck", "items": [
        {"lang": "id", "seed": 42, "len_ratio": 1.0, "over_read": False, "tail_silence_sec": 0.1},
        {"lang": "id", "seed": 44, "len_ratio": 1.6, "over_read": True, "tail_silence_sec": 0.2},
        {"lang": "id", "seed": 44, "len_ratio": 7.97, "over_read": True, "tail_silence_sec": 0.3},
        {"lang": "th", "seed": 42, "len_ratio": 1.0, "over_read": False, "tail_silence_sec": 0.1},
    ]}
    text = "\n".join(runlog._seed_severity_lines([rep]))
    assert "不作门禁" in text
    assert "id（3 条）中等 1 / 灾难 1" in text
    assert "th（1 条）中等 0 / 灾难 0" in text
    assert "s42 多读0/1 max1.00×" in text
    assert "s44 多读2/2 max7.97×" in text


def test_record_includes_seed_breakdown(tmp_path, monkeypatch):
    """分解必须真的出现在生成的记录里，否则等于没接上。"""
    monkeypatch.setattr(runlog, "CONFIG_DIR", tmp_path)
    monkeypatch.setattr(runlog, "CHECKPOINT_DIR", tmp_path)
    _write_cfg(tmp_path, "with_eval")
    report = tmp_path / "ck.json"
    report.write_text(json.dumps({
        "label": "ck", "cfg_value": 1.8, "inference_timesteps": 20, "retry_badcase": False,
        "asr_model": "large-v3", "mean_cer": 0.05, "suspected_truncation_rate": 0.0,
        "by_lang": {"id": {"mean_cer": 0.05}},
        "items": [{"case_id": "c1", "lang": "id", "seed": 44, "len_ratio": 3.0,
                   "over_read": True, "tail_silence_sec": 0.4, "cer": 0.05}],
    }, ensure_ascii=False), encoding="utf-8")
    base = tmp_path / "base.json"
    base.write_text(json.dumps({
        "label": "base", "cfg_value": 1.8, "inference_timesteps": 20, "retry_badcase": False,
        "asr_model": "large-v3", "mean_cer": 0.04, "suspected_truncation_rate": 0.0,
        "by_lang": {"id": {"mean_cer": 0.04}},
        "items": [{"case_id": "c1", "lang": "id", "seed": 44, "len_ratio": 1.0,
                   "over_read": False, "tail_silence_sec": 0.2, "cer": 0.04}],
    }, ensure_ascii=False), encoding="utf-8")
    text = runlog.build_record("with_eval", [str(base), str(report)])
    assert "逐 seed 与严重度分解" in text
    assert "id（1 条）中等 0 / 灾难 1" in text


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


def test_absolute_tail_policy_keeps_legacy_and_other_gates():
    base = {"by_lang": {"id": {"p90_tail_silence": .16, "mean_speech_ratio": .95}}}
    candidate = {"label": "r30", "by_lang": {
        "id": {"p90_tail_silence": .34, "mean_speech_ratio": .89},
        "th": {"p90_tail_silence": .5}, "tl": {"p90_tail_silence": .51}}}
    legacy = runlog._duration_gates([base, candidate])["r30"]
    current = runlog._duration_gates([base, candidate], tail_policy="absolute")["r30"]
    assert len(legacy["tail_silence"]) == 2
    assert "tail_silence_observations" not in legacy
    assert len(current["tail_silence"]) == 1 and current["tail_silence"][0].startswith("tl ")
    assert len(current["tail_silence_observations"]) == 1
    assert current["tail_silence_observations"][0].startswith("id ")
    assert current["speech_ratio"] == legacy["speech_ratio"] and current["speech_ratio"]
    with pytest.raises(ValueError, match="Unknown tail policy"):
        runlog._duration_gates([], tail_policy="typo")


def test_cli_writes_explicit_tail_policy_and_observations(tmp_path, monkeypatch):
    monkeypatch.setattr(runlog, "CONFIG_DIR", tmp_path)
    monkeypatch.setattr(runlog, "CHECKPOINT_DIR", tmp_path)
    monkeypatch.setattr(runlog, "RUNLOG", tmp_path / "runs.md")
    _write_cfg(tmp_path, "relaxed")
    paths = []
    for label, tail in (("base", .16), ("candidate", .34)):
        path = tmp_path / f"{label}.json"
        path.write_text(json.dumps({"label": label, "p90_tail_silence": tail}), encoding="utf-8")
        paths.append(str(path))
    monkeypatch.setattr(sys, "argv", ["runlog", "--run", "relaxed", "--eval", *paths,
                                     "--tail-policy", "absolute"])
    runlog.main()
    text = runlog.RUNLOG.read_text(encoding="utf-8")
    assert "尾静音策略=absolute" in text
    assert "尾静音观察（不否决）" in text
    assert "- `candidate`：无" in text
