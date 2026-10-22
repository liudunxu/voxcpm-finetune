"""训练前预检逻辑自测。"""
import json

import yaml

from voxft.train.launcher import preflight


def _cfg(tmp_path, train=None, pre=None):
    if pre is None:  # 训练脚本要求本地基座目录，裸 HF 仓库 ID 本身就是致命问题
        base = tmp_path / "base"
        base.mkdir(exist_ok=True)
        (base / "config.json").write_text("{}", encoding="utf-8")
        pre = str(base)
    c = {
        "pretrained_path": pre,
        "train_manifest": train or "",
        "save_path": str(tmp_path / "ckpt"),
        "num_iters": 10,
    }
    p = tmp_path / "cfg.yaml"
    p.write_text(yaml.safe_dump(c), encoding="utf-8")
    return p


def test_preflight_missing_manifest(tmp_path):
    issues = preflight(_cfg(tmp_path))
    assert any("train_manifest" in i for i in issues)


def test_preflight_bad_audio(tmp_path):
    mf = tmp_path / "train.jsonl"
    mf.write_text(json.dumps({"audio": "/nonexistent/x.wav", "text": "hi"}) + "\n",
                  encoding="utf-8")
    issues = preflight(_cfg(tmp_path, train=str(mf)))
    assert any("音频不存在" in i for i in issues)


def test_preflight_ok_manifest(tmp_path):
    wav = tmp_path / "a.wav"
    wav.write_bytes(b"x")
    mf = tmp_path / "train.jsonl"
    mf.write_text(json.dumps({"audio": str(wav), "text": "hi"}) + "\n",
                  encoding="utf-8")
    issues = preflight(_cfg(tmp_path, train=str(mf)))
    # 仅剩可能的警告（无 CUDA），不应有致命问题
    assert all(i.startswith("警告") for i in issues)


def test_preflight_bad_pretrained(tmp_path):
    mf = tmp_path / "train.jsonl"
    mf.write_text("{}", encoding="utf-8")
    issues = preflight(_cfg(tmp_path, train=str(mf), pre="not-a-valid-path"))
    assert any("pretrained_path" in i for i in issues)


def _rows(tmp_path, n):
    wav = tmp_path / "a.wav"
    wav.write_bytes(b"x")
    mf = tmp_path / "train.jsonl"
    mf.write_text("".join(json.dumps({"audio": str(wav), "text": "hi"}) + "\n" for _ in range(n)),
                  encoding="utf-8")
    return mf


def _set_iters(path, num_iters):
    cfg = yaml.safe_load(path.read_text(encoding="utf-8"))
    cfg["num_iters"] = num_iters
    path.write_text(yaml.safe_dump(cfg), encoding="utf-8")
    return path


def test_preflight_three_epoch_ceil_slack(tmp_path):
    """steps_for_epochs 向上取整：正好 3 epoch 的配置不该被报成「超过 3 epoch」。"""
    mf = _rows(tmp_path, 8)  # 8 行 / 等效 batch 16 → 3 epoch = ceil(1.5) = 2 步
    cfg = _set_iters(_cfg(tmp_path, train=str(mf)), 2)
    assert not any("超过 3 epoch" in i for i in preflight(cfg))


def test_preflight_over_three_epoch_warns(tmp_path):
    mf = _rows(tmp_path, 8)
    cfg = _set_iters(_cfg(tmp_path, train=str(mf)), 5)  # 5×16=80 > 3×8+16=40
    assert any("超过 3 epoch" in i for i in preflight(cfg))
