"""情绪/语气 prompt 的推理侧接线自测（不加载模型）。"""
from voxft.infer import _gen_kwargs, clean_control


def test_clean_control_strips_brackets():
    assert clean_control("（愤怒地）") == "愤怒地"
    assert clean_control(None) == ""


def test_control_becomes_text_prefix():
    kw = _gen_kwargs("สวัสดี", None, None, 2.0, 20, 42, "愤怒地，语速快")
    assert kw["text"] == "(愤怒地，语速快)สวัสดี"


def test_control_forces_reference_only_mode():
    """combined 模式下 text = prompt_text + target_text，前缀会跑到句中而失效。"""
    kw = _gen_kwargs("hello", "/tmp/ref.wav", "ref transcript", 2.0, 20, 42,
                     "sad, slow")
    assert kw["reference_wav_path"] == "/tmp/ref.wav"
    assert "prompt_text" not in kw and "prompt_wav_path" not in kw

    kw = _gen_kwargs("hello", "/tmp/ref.wav", "ref transcript", 2.0, 20, 42, "")
    assert kw["prompt_text"] == "ref transcript"  # 无前缀时仍走 combined
