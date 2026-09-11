"""不下载权重的训练计划 / LoRA 接线 / 离线验收回归。"""
import json
import sys
from types import SimpleNamespace

import pytest
import yaml

from voxft import eval as evaluation, infer
from voxft.train import yaml_builder as builder
from voxft.train.launcher import preflight


def test_epoch_plan_and_preflight(tmp_path, monkeypatch):
    monkeypatch.setattr(builder, "CONFIG_DIR", tmp_path / "configs")
    monkeypatch.setattr(builder, "CHECKPOINT_DIR", tmp_path / "ckpt")
    base = tmp_path / "base"
    base.mkdir()
    (base / "config.json").write_text("{}")
    wav = tmp_path / "a.wav"
    wav.write_bytes(b"fake")
    train = tmp_path / "train.jsonl"
    train.write_text((json.dumps({"audio": str(wav), "text": "test"}) + "\n") * 320)
    val = tmp_path / "val.jsonl"
    val.write_text("")
    config = builder.build_yaml("trial", str(base), str(train), str(val), epochs=1, gpus=2)
    cfg = yaml.safe_load(config.read_text())
    assert cfg["num_iters"] == cfg["max_steps"] == 10 and cfg["warmup_steps"] == 1
    assert cfg["val_manifest"] == "" and cfg["lora"]["enable_dit"] is True
    assert cfg["lora"]["r"] == 64 and cfg["learning_rate"] == 1e-4
    assert all(i.startswith("警告") for i in preflight(config, 2))
    assert any("GPU 数" in i for i in preflight(config, 1))
    with pytest.raises(ValueError):
        builder.steps_for_epochs(str(train), 4)
    with pytest.raises(ValueError, match="training_cfg_rate"):
        builder.build_yaml("bad", str(base), str(train), overrides={"training_cfg_rate": 0.1})
    # 检查第 4 行之后的坏 ref，不再只抽查前三行。
    with train.open("a") as f:
        f.write(json.dumps({"audio": str(wav), "text": "bad ref", "ref_audio": str(wav)}) + "\n")
    assert any("ref 缺少" in i for i in preflight(config, 2))
    (base / "config.json").write_text(json.dumps({"dit_config": {"cfm_config": {"training_cfg_rate": 0}}}))
    assert any("training_cfg_rate=0" in i for i in preflight(config, 2))


def test_plan_records_langs_and_run_name(tmp_path, monkeypatch):
    """联合 run 事后要能反查训了哪些语种；run 名也要带上，否则一堆 lora_0911 分不清。"""
    monkeypatch.setattr(builder, "CONFIG_DIR", tmp_path / "configs")
    monkeypatch.setattr(builder, "CHECKPOINT_DIR", tmp_path / "ckpt")
    wav = tmp_path / "a.wav"
    wav.write_bytes(b"fake")
    train = tmp_path / "train.jsonl"
    rows = ([{"audio": str(wav), "text": "t", "lang": "th"}] * 2
            + [{"audio": str(wav), "text": "t", "lang": lang} for lang in ("ms", "tl", "vi", "id")]
            + [{"audio": str(wav), "text": "t"}])       # 没标 lang 的行归 unknown，不许吞掉
    train.write_text("".join(json.dumps(r) + "\n" for r in rows))
    config = builder.build_yaml("joint", "", str(train), epochs=1)
    plan = json.loads(config.with_suffix(".plan.json").read_text())
    assert plan["langs"] == {"id": 1, "ms": 1, "th": 2, "tl": 1, "unknown": 1, "vi": 1}
    assert plan["train_samples"] == 7
    name = builder.default_run_name("lora", str(train))
    assert name.startswith("lora_joint6_") and "/" not in name and name not in ("", ".", "..")
    assert builder.default_run_name("lora").startswith("lora_")   # 没给清单时退回时间戳


def test_lora_load_config_and_ab_toggles(tmp_path, monkeypatch):
    lora = tmp_path / "adapter"
    lora.mkdir()
    (lora / "lora_config.json").write_text(json.dumps({"base_model": "base",
                                                       "lora_config": {"r": 64, "alpha": 64}}))
    seen, toggles = {}, []
    loaded = (["x.lora_A", "x.lora_B"], [])
    fake_model = SimpleNamespace(
        load_lora=lambda path: loaded,
        set_lora_enabled=toggles.append,
        tts_model=SimpleNamespace(named_parameters=lambda: [("x._orig_mod.lora_A", None), ("x._orig_mod.lora_B", None)]))
    monkeypatch.setitem(sys.modules, "voxcpm.model.voxcpm2", SimpleNamespace(LoRAConfig=lambda **kw: kw))
    monkeypatch.setattr(infer, "_MODEL", None)
    monkeypatch.setattr(infer, "_MODEL_KEY", None)
    monkeypatch.setattr(infer, "_import_voxcpm", lambda: SimpleNamespace(
        from_pretrained=lambda base, **kw: seen.update(kw) or fake_model))
    assert infer.get_model("base", str(lora)) is fake_model
    assert seen["lora_config"] == {"r": 64, "alpha": 64}
    generated = []
    def run(model, kwargs):
        generated.append((toggles[-1], dict(kwargs)))
        return f"{toggles[-1]}.wav", 1
    monkeypatch.setattr(infer, "_run", run)
    infer.synthesize_ab("text", "base", str(lora), control="angry")
    assert [x[0] for x in generated] == [False, True]
    assert generated[0][1] == generated[1][1] and not generated[0][1]["retry_badcase"]
    for loaded in (([], []), (["x.lora_A"], []), (["x.lora_A"], ["bad"])):
        monkeypatch.setattr(infer, "_MODEL_KEY", None)
        with pytest.raises(RuntimeError, match="加载不完整"):
            infer.get_model("base", str(lora))


def test_eval_keeps_conditions_thai_marks_and_unique_reports(tmp_path, monkeypatch):
    from voxft.data import pipeline
    monkeypatch.setattr(evaluation, "CHECKPOINT_DIR", tmp_path)
    monkeypatch.setattr(pipeline, "_whisper_model", lambda lang, size: object())
    monkeypatch.setattr(infer, "get_model", lambda *args: object())
    kwargs = []
    monkeypatch.setattr(infer, "_run", lambda model, kw: kwargs.append(kw) or ("fake.wav", 0.1))
    monkeypatch.setattr(evaluation, "_transcribe", lambda *a: "Hindi mo alam na buntis ka?")
    monkeypatch.setattr(evaluation, "_prosody", lambda *a: {"f0_std_st": 1.0})
    assert evaluation._norm("ก่ ก") == "ก่ก"
    assert evaluation._error_rate("abc", "ac") == 0.5
    assert evaluation._error_rate("", "ab") == 1
    ref = tmp_path / "ref.wav"
    ref.write_bytes(b"fake")
    cases = [{"text": "Hindi mo alam na buntis ka?", "lang": "tl", "control": "surprised",
              "ref_audio": str(ref), "ref_lang": "en", "speaker": "female_lead"}]
    report = evaluation.evaluate("base", "th", cases, seeds=[42, 43])
    again = evaluation.evaluate("base", "th", cases, seeds=[42, 43])
    assert report["report_path"] != again["report_path"]
    assert report["mean_cer"] == 0 and report["items"][0]["ref_lang"] == "en"
    assert report["items"][0]["human_review"]["emotion_fit_1_5"] is None
    assert [kw["seed"] for kw in kwargs] == [42, 43, 42, 43]
    assert all(not kw["retry_badcase"] and kw["text"].startswith("(surprised)") for kw in kwargs)


def test_every_registry_lang_has_eval_channel():
    """加语种时必须同步 SAMPLE_BY_LANG，否则 eval 直接拒绝该语种的 case。"""
    from voxft.data.registry import SOURCES, TARGET_LANGS
    missing = {s.lang for s in SOURCES} - set(evaluation.SAMPLE_BY_LANG)
    assert not missing, f"registry 里有语种没有验收通道: {missing}"
    assert set(TARGET_LANGS) <= set(evaluation.SAMPLE_BY_LANG)
    assert evaluation.AUTO_DETECT_LANGS == {"tl"}
    # ms/id/tl/en 词间有空格 → 词级 WER；vi 是音节级；th 词间无空格只有 CER
    assert set(evaluation.WER_LANGS) == {"tl", "en", "vi", "id", "ms"}


def test_eval_computes_cer_and_wer_for_vi_id_and_ms(tmp_path, monkeypatch):
    """vi 正字法按音节空格分隔，WER 有值但是音节级口径；id/ms 是词级。"""
    from voxft.data import pipeline
    monkeypatch.setattr(evaluation, "CHECKPOINT_DIR", tmp_path)
    monkeypatch.setattr(pipeline, "_whisper_model", lambda lang, size: object())
    monkeypatch.setattr(infer, "get_model", lambda *args: object())
    monkeypatch.setattr(infer, "_run", lambda model, kw: ("fake.wav", 0.1))
    monkeypatch.setattr(evaluation, "_prosody", lambda *a: {"f0_std_st": 1.0})
    for lang, text in (("vi", "Tôi không biết"), ("id", "Saya tidak tahu"),
                       ("ms", "Saya tidak tahu")):
        monkeypatch.setattr(evaluation, "_transcribe", lambda *a, text=text: text)
        report = evaluation.evaluate("base", lang, [{"text": text, "lang": lang}])
        item = report["items"][0]
        assert item["cer"] == 0 and item["wer"] == 0, f"{lang} 应算出 CER 与 WER"
        assert report["asr_auto_detect_langs"] == ["tl"]
        assert report["by_lang"][lang]["cases"] == 1
        assert report["by_lang"][lang]["mean_wer"] == 0
