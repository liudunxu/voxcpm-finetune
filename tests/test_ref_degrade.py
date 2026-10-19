import hashlib
import importlib.util
import json
from collections import Counter
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

from voxft.data import ref_degrade
from voxft.data.ref_degrade import PROFILES, degrade_file, degrade_wav

SR = 16000


def _noise(dur, amp, seed):
    rs = np.random.RandomState(seed)
    return (amp * rs.standard_normal(int(SR * dur))).astype(np.float32)


def _write_wav(path, wav, sr=SR):
    path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(path, wav, sr)
    return path


def _load_script(name):
    spec = importlib.util.spec_from_file_location(
        name, Path(__file__).resolve().parents[1] / "scripts" / name)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _pool(*tags):
    """两条不同说话人/不同 origin 的白噪声干扰源。"""
    return [{"wav": _noise(4.0, 0.2, 900 + i), "sr": SR,
             "speaker": f"spk_{tag}", "origin_audio": f"origin_{tag}"}
            for i, tag in enumerate(tags)]


def _measured_sir(out, ref, sr=SR):
    """残差法测 SIR：out ≈ g·(ref + 干扰)，g 用最小二乘投影估计，残余即干扰分量。"""
    g = float(np.dot(out, ref) / np.dot(ref, ref))
    residual = out - g * ref
    ref_voiced = ref_degrade._voiced_rms(np.asarray(ref, dtype=np.float64), sr)
    rms_res = float(np.sqrt(np.mean(np.square(residual))))
    return 20.0 * np.log10(g * ref_voiced / rms_res)


def _band_energy_ratio(before, after, sr, f_lo):
    """after 在 f_lo 以上的能量 / before 在 f_lo 以上的能量。"""
    def band(w):
        psd = np.abs(np.fft.rfft(w)) ** 2
        return float(psd[np.fft.rfftfreq(len(w), 1 / sr) >= f_lo].sum())
    return band(after) / band(before)


def test_degrade_wav_deterministic_and_seed_sensitive():
    ref = _noise(3.0, 0.1, 0)
    pool = _pool("x", "y")
    out1, rec1 = degrade_wav(ref, SR, "babble_sir8", seed=7, interferers=pool,
                             ref_speaker="spk_ref", ref_origin="origin_ref")
    out2, rec2 = degrade_wav(ref, SR, "babble_sir8", seed=7, interferers=pool,
                             ref_speaker="spk_ref", ref_origin="origin_ref")
    assert np.array_equal(out1, out2) and rec1 == rec2
    out3, _ = degrade_wav(ref, SR, "babble_sir8", seed=8, interferers=pool,
                          ref_speaker="spk_ref", ref_origin="origin_ref")
    assert not np.array_equal(out1, out3)
    lp1, _ = degrade_wav(ref, SR, "lowpass4k", seed=7)
    lp2, _ = degrade_wav(ref, SR, "lowpass4k", seed=99)  # 无随机性的 profile 不受 seed 影响
    assert np.array_equal(lp1, lp2)
    assert len(out1) == len(ref)  # 降质保持长度不变，ref_duration 不失效


def test_babble_sir_within_tolerance():
    ref = _noise(3.0, 0.1, 0)
    pool = _pool("x", "y", "z")
    for profile, target in (("babble_sir15", 15.0), ("babble_sir8", 8.0)):
        out, rec = degrade_wav(ref, SR, profile, seed=123, interferers=pool,
                               ref_speaker="spk_ref", ref_origin="origin_ref")
        measured = _measured_sir(out, ref)
        assert abs(measured - target) <= 1.5, f"{profile}: 实测 SIR {measured:.2f}dB"
        assert rec["sir_db"] == target and rec["severity"] == PROFILES[profile]["severity"]
        assert 1 <= len(rec["interferer_origins"]) <= 2
        assert all(o.startswith("origin_") and o != "origin_ref"
                   for o in rec["interferer_origins"])


def test_babble_rejects_same_speaker_or_origin():
    ref = _noise(2.0, 0.1, 0)
    with pytest.raises(ValueError, match="干扰池为空"):
        degrade_wav(ref, SR, "babble_sir15", seed=1, ref_speaker="spk_x",
                    ref_origin="origin_ref",
                    interferers=_pool("x"))  # 唯一干扰源与 ref 同 speaker
    with pytest.raises(ValueError, match="干扰池为空"):
        degrade_wav(ref, SR, "babble_sir15", seed=1, ref_speaker="spk_ref",
                    ref_origin="origin_x", interferers=_pool("x"))  # 同 origin_audio
    with pytest.raises(ValueError, match="干扰池为空"):
        degrade_wav(ref, SR, "babble_sir15", seed=1, interferers=None)  # 无池
    # 混合池只滤掉违规条目，不整批失败
    pool = _pool("ref", "y")
    pool[0]["origin_audio"] = "origin_ref"
    _, rec = degrade_wav(ref, SR, "babble_sir15", seed=1, interferers=pool,
                         ref_speaker="spk_ref", ref_origin="origin_ref")
    assert rec["interferer_origins"] == ["origin_y"]


def test_lowpass_spectral_and_level():
    ref = _noise(3.0, 0.1, 10)
    for profile, cutoff in (("lowpass4k", 4000), ("lowpass3k", 3000)):
        out, rec = degrade_wav(ref, SR, profile, seed=1)
        # 5.5kHz 以上能量相对原信号下降 >95%（sr=16000 白噪声）
        assert _band_energy_ratio(ref.astype(np.float64), out, SR, 5500) < 0.05
        # RMS 对齐误差 <0.5dB
        align_db = 20.0 * np.log10(float(np.sqrt(np.mean(out ** 2)))
                                   / float(np.sqrt(np.mean(ref.astype(np.float64) ** 2))))
        assert abs(align_db) < 0.5
        assert rec["cutoff_hz"] == cutoff and rec["kind"] == "lowpass"
        assert rec["clipped"] is False


def test_combo_has_babble_and_lowpass_signatures():
    ref = _noise(3.0, 0.1, 0).astype(np.float64)
    pool = _pool("x", "y")
    out, rec = degrade_wav(ref, SR, "combo_babble8_lp3k", seed=5, interferers=pool,
                           ref_speaker="spk_ref", ref_origin="origin_ref")
    # 串音残差：ref 先过同一个 3k 低通再按残差法测 SIR
    lp_ref = ref_degrade._lowpass(ref, SR, 3000)
    assert abs(_measured_sir(out, lp_ref) - 8.0) <= 1.5
    # 低通频谱特征
    assert _band_energy_ratio(ref, out, SR, 5500) < 0.05
    assert rec["sir_db"] == 8.0 and rec["cutoff_hz"] == 3000 and rec["kind"] == "combo"


def test_degrade_file_deterministic_and_source_untouched(tmp_path):
    src = _write_wav(tmp_path / "ref.wav", _noise(2.0, 0.1, 50))
    src_bytes = src.read_bytes()
    pool = _pool("x", "y")
    rec1 = degrade_file(src, tmp_path / "o1.wav", "babble_sir15", seed=11,
                        interferers=pool, ref_speaker="spk_ref")
    rec2 = degrade_file(src, tmp_path / "o2.wav", "babble_sir15", seed=11,
                        interferers=pool, ref_speaker="spk_ref")
    assert (tmp_path / "o1.wav").read_bytes() == (tmp_path / "o2.wav").read_bytes()
    assert src.read_bytes() == src_bytes  # 源文件一个字节都不动
    assert rec1["dst"] == "o1.wav" and rec2["dst"] == "o2.wav"
    assert {k: v for k, v in rec1.items() if k != "dst"} == \
           {k: v for k, v in rec2.items() if k != "dst"}
    assert rec1["source_sha256"] == hashlib.sha256(src_bytes).hexdigest()
    assert rec1["profile"] == "babble_sir15" and rec1["seed"] == 11
    assert rec1["params"]["profile"] == "babble_sir15"


def _pool_jsonl(tmp_path):
    rows = []
    for i, spk in enumerate(("spkX", "spkY")):
        p = _write_wav(tmp_path / "pool" / f"{spk}.wav", _noise(3.0, 0.2, 30 + i))
        rows.append({"audio": str(p), "speaker": spk, "origin_audio": f"origin_{spk}"})
    path = tmp_path / "pool.jsonl"
    path.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
    return path


def test_cli_directory_manifest(tmp_path):
    refs = tmp_path / "refs"
    _write_wav(refs / "a.wav", _noise(2.0, 0.1, 20))
    _write_wav(refs / "b.wav", _noise(2.0, 0.1, 21))
    pool = _pool_jsonl(tmp_path)
    out = tmp_path / "out"
    ref_degrade.main(["--manifest", str(refs), "--out", str(out), "--pool", str(pool)])
    wavs = sorted(out.rglob("*.wav"))
    assert len(wavs) == 10  # 2 参考 × 5 profile
    records = [json.loads(line) for line in
               (out / "degradations.jsonl").read_text(encoding="utf-8").splitlines()]
    assert len(records) == 10
    for rec in records:
        assert rec["dst"] == f"{rec['profile']}/{Path(rec['source']).name}"
        assert rec["source_sha256"] == hashlib.sha256(
            Path(rec["source"]).read_bytes()).hexdigest()
        assert rec["params"]["severity"] == PROFILES[rec["profile"]]["severity"]
    # 跨进程确定性：同样输入跑到另一个输出目录，产物逐字节一致
    out2 = tmp_path / "out2"
    ref_degrade.main(["--manifest", str(refs), "--out", str(out2), "--pool", str(pool)])
    for wav in wavs:
        assert wav.read_bytes() == (out2 / wav.relative_to(out)).read_bytes()


def test_cli_pool_required_and_same_speaker_rejected(tmp_path):
    refs = tmp_path / "refs"
    _write_wav(refs / "a.wav", _noise(1.0, 0.1, 40))
    # 默认 5 个 profile 含 babble 类，缺 --pool 直接拒绝
    with pytest.raises(SystemExit):
        ref_degrade.main(["--manifest", str(refs), "--out", str(tmp_path / "o1")])
    # 只有低通类时不需要池
    ref_degrade.main(["--manifest", str(refs), "--out", str(tmp_path / "o2"),
                      "--profiles", "lowpass4k", "lowpass3k"])
    assert len(list((tmp_path / "o2").rglob("*.wav"))) == 2
    # 池里全是同 speaker 干扰源 → 不过滤出空就报错，不静默降级
    manifest = tmp_path / "m.jsonl"
    manifest.write_text(json.dumps({"audio": str(refs / "a.wav"), "speaker": "spkA",
                                    "origin_audio": "oA"}) + "\n", encoding="utf-8")
    bad = _write_wav(tmp_path / "bad.wav", _noise(1.0, 0.1, 41))
    pool = tmp_path / "pool.jsonl"
    pool.write_text(json.dumps({"audio": str(bad), "speaker": "spkA",
                                "origin_audio": "oB"}) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="干扰池为空"):
        ref_degrade.main(["--manifest", str(manifest), "--out", str(tmp_path / "o3"),
                          "--pool", str(pool)])


def test_build_degraded_ref_cases(tmp_path):
    mod = _load_script("build_degraded_ref_cases.py")
    rows = []
    for lang, n in (("th", 8), ("tl", 7), ("id", 5)):
        for i in range(n):
            rows.append({"case_id": f"{lang}_{i:02d}", "lang": lang,
                         "text": f"{lang} 文本 {i}",
                         "ref_audio": f"refs/ref_{lang}_{i % 2}.wav", "ref_lang": "zh",
                         "seed": 1000 + i, "note": "t",
                         **({"numeric": True} if i == 0 else {})})
    src = tmp_path / "cases.jsonl"
    src.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows),
                   encoding="utf-8")
    warnings = []
    summary = mod.build(src, tmp_path / "out", stamp="20261019", progress=warnings.append)
    assert set(summary) == {f"refrob_20261019_{c}.jsonl" for c in ("clean", *PROFILES)}
    assert all(n == 17 for n in summary.values())  # id 不足 6 条全取，已警告
    assert any("id" in w for w in warnings)
    out = tmp_path / "out"

    def read(name):
        return [json.loads(line) for line in
                (out / name).read_text(encoding="utf-8").splitlines()]

    orig = {r["case_id"]: r for r in rows}
    clean = read("refrob_20261019_clean.jsonl")
    assert Counter(r["lang"] for r in clean) == {"th": 6, "tl": 6, "id": 5}
    for row in clean:  # clean 与原 case 逐字段一致，仅追加两个新字段
        original = orig[row["case_id"]]
        assert row["ref_audio_original"] == original["ref_audio"]
        assert row["ref_degradation"] is None
        assert {k: v for k, v in row.items()
                if k not in ("ref_audio_original", "ref_degradation")} == original
    ids = {r["case_id"] for r in clean}
    for cond in PROFILES:
        degraded = read(f"refrob_20261019_{cond}.jsonl")
        assert {r["case_id"] for r in degraded} == ids  # case_id 集合跨条件不变
        assert any(r.get("numeric") is True for r in degraded)
        for row in degraded:
            original = orig[row["case_id"]]
            assert row["ref_audio"] == (
                f"refs_degraded/{cond}/{Path(original['ref_audio']).name}")
            assert row["ref_audio_original"] == original["ref_audio"]
            assert row["ref_degradation"] == {"profile": cond,
                                              "severity": PROFILES[cond]["severity"]}
            for key in ("text", "lang", "seed", "note", "ref_lang"):
                assert row[key] == original[key]
    # 幂等重跑（内容一致跳过）；抽样与字节跨进程确定
    assert mod.build(src, out, stamp="20261019") == summary
    mod.build(src, tmp_path / "out2", stamp="20261019")
    assert (tmp_path / "out2/refrob_20261019_clean.jsonl").read_bytes() == \
           (out / "refrob_20261019_clean.jsonl").read_bytes()
    (out / "refrob_20261019_clean.jsonl").write_text("{}\n", encoding="utf-8")
    with pytest.raises(ValueError, match="不覆盖旧产物"):
        mod.build(src, out, stamp="20261019")


def _fake_processed_dataset(tmp_path):
    """12 行 train（8 行带 ref、spkA/spkB 已验证）+ 2 行 val（独立说话人）+ mix.json。"""
    src = tmp_path / "processed" / "joint_demo"
    rows = []
    for i in range(12):
        audio = _write_wav(src / "audio" / f"{i:07d}.wav", _noise(2.0, 0.1, 100 + i))
        speaker = f"demo:spk{'AABBAABB'[i] if i < 8 else 'CDCD'[i - 8]}"
        row = {"audio": str(audio), "text": f"文本 {i}", "speaker": speaker,
               "origin_audio": f"origin_{i}", "duration": 2.0, "lang": "th",
               "speaker_verified": i < 8}
        if i < 8:
            ref = _write_wav(src / "refs" / f"ref_{i}.wav", _noise(1.5, 0.1, 200 + i))
            row.update(ref_audio=str(ref), ref_speaker=speaker,
                       ref_origin_audio=f"reforigin_{i}", ref_duration=1.5)
        rows.append(row)
    (src / "train.jsonl").write_text(
        "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8")
    val = [{"audio": f"/val/{i}.wav", "text": f"验证 {i}", "speaker": "demo:spkVal",
            "origin_audio": f"val_origin_{i}", "duration": 2.0, "lang": "th",
            "speaker_verified": True} for i in range(2)]
    (src / "val.jsonl").write_text(
        "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in val), encoding="utf-8")
    (src / "mix.json").write_text(json.dumps({"parts": [["joint_demo", 1.0]]}),
                                  encoding="utf-8")
    return src, rows


def test_degrade_train_refs(tmp_path):
    mod = _load_script("degrade_train_refs.py")
    src, rows = _fake_processed_dataset(tmp_path)
    train_sha = hashlib.sha256((src / "train.jsonl").read_bytes()).hexdigest()
    stats = mod.degrade_dataset(str(src))  # 默认 out = 同级 <name>_refrob
    out = tmp_path / "processed" / "joint_demo_refrob"

    # 源目录一个字节都不改；val/mix 逐字节复制
    assert hashlib.sha256((src / "train.jsonl").read_bytes()).hexdigest() == train_sha
    assert (out / "val.jsonl").read_bytes() == (src / "val.jsonl").read_bytes()
    assert (out / "mix.json").read_bytes() == (src / "mix.json").read_bytes()

    new_rows = [json.loads(line) for line in
                (out / "train.jsonl").read_text(encoding="utf-8").splitlines()]
    assert len(new_rows) == 12
    replaced = [r for r in new_rows if "ref_audio_original" in r]
    assert len(replaced) == 4  # 8 行带 ref × share 0.5
    originals_by_audio = {r["audio"]: r for r in rows}
    for row in new_rows:
        original = originals_by_audio[row["audio"]]
        if "ref_audio_original" not in row:
            assert row == original  # 未选中行（含全部无 ref 行）一个字段都不动
            continue
        assert original.get("ref_audio")  # 只有带 ref 的行才可能被选中
        assert row["ref_audio_original"] == original["ref_audio"]
        assert row["ref_audio"] != original["ref_audio"]
        assert row["ref_audio"].startswith(str(out))
        assert row["ref_degradation"]["profile"] in PROFILES
        assert row["ref_degradation"]["severity"] == \
            PROFILES[row["ref_degradation"]["profile"]]["severity"]
        for key in ("audio", "text", "speaker", "duration", "ref_duration", "lang"):
            assert row[key] == original[key]  # 降质保持长度不变，ref_duration 不动
        # 产物 wav 存在、长度与原 ref 一致
        degraded = sf.info(row["ref_audio"])
        assert degraded.frames == sf.info(original["ref_audio"]).frames
        assert degraded.samplerate == SR

    # sidecar：干扰源只来自同一 train split，拒绝 val 说话人与 ref 本人
    records = [json.loads(line) for line in
               (out / "degradations.jsonl").read_text(encoding="utf-8").splitlines()]
    train_origins = {f"origin_{i}" for i in range(12)}
    sha_to_row = {}
    for row in rows:
        if row.get("ref_audio"):
            sha_to_row[hashlib.sha256(Path(row["ref_audio"]).read_bytes()).hexdigest()] = row
    for rec in records:
        ref_row = sha_to_row[rec["source_sha256"]]
        for origin in rec["params"].get("interferer_origins", []):
            assert origin in train_origins and not origin.startswith("val_")
            assert origin != ref_row["ref_origin_audio"]
            speaker = next(r["speaker"] for r in rows if r["origin_audio"] == origin)
            assert speaker != ref_row["ref_speaker"]

    # 统计文件：替换数、profile 分布、share、seed、源目录
    saved = json.loads((out / "refrob_stats.json").read_text(encoding="utf-8"))
    assert saved["replaced"] == 4 and saved["ref_rows"] == 8 and saved["train_rows"] == 12
    assert saved["share"] == 0.5 and saved["seed"] == 42
    assert saved["source"] == str(src) and sum(saved["profiles"].values()) == 4
    assert set(saved["profiles"]) <= set(PROFILES)
    assert stats == saved

    # 确定性：换输出目录重跑，逐行一致（路径前缀归一后）；同目录重跑拒绝覆盖
    out2 = tmp_path / "second"
    mod.degrade_dataset(str(src), str(out2))
    rows2 = [json.loads(line) for line in
             (out2 / "train.jsonl").read_text(encoding="utf-8").splitlines()]
    for r1, r2 in zip(new_rows, rows2):
        assert {k: (v.replace(str(out), "") if k == "ref_audio" else v)
                for k, v in r1.items()} == \
               {k: (v.replace(str(out2), "") if k == "ref_audio" else v)
                for k, v in r2.items()}
    with pytest.raises(ValueError, match="不覆盖旧产物"):
        mod.degrade_dataset(str(src))

    # share 缩小到 0.25 → 恰好替换 2 行
    stats3 = mod.degrade_dataset(str(src), str(tmp_path / "third"), share=0.25)
    assert stats3["replaced"] == 2
