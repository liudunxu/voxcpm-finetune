import importlib
import json
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
ERES_DIR = ROOT.parent / "dubbing_intelligence_service" / "models" / "eres2netv2"


@pytest.fixture
def tv(monkeypatch):
    monkeypatch.syspath_prepend(str(ROOT / "scripts"))
    return importlib.import_module("timbre_v1")


@pytest.mark.skipif(not (ERES_DIR / "speaker.onnx").is_file(), reason="DIS ERes2NetV2 ONNX 不在本机")
def test_eres2net_port_matches_dis_validation_vectors(tv):
    model = tv.Eres2Net(ERES_DIR)
    with np.load(ERES_DIR / "validation.npz") as data:
        for sec in ("0.5", "1.2", "3.715", "30.0"):
            got = model.run(data[f"{sec}_features"][None].astype(np.float32))
            want = data[f"{sec}_embedding"] / np.linalg.norm(data[f"{sec}_embedding"])
            assert float(got @ want) > 0.9999


def test_f0_profile_genders_tones(tv):
    t = np.arange(16000 * 2) / 16000
    low = tv.f0_profile((0.3 * np.sin(2 * np.pi * 120 * t)).astype(np.float32))
    high = tv.f0_profile((0.3 * np.sin(2 * np.pi * 260 * t)).astype(np.float32))
    assert low["gender"] == "male" and high["gender"] == "female"
    assert abs(low["p50"] - 120) < 5 and abs(high["p50"] - 260) < 8


def test_cluster_ci_weights_clusters_equally(tv):
    ci = tv.cluster_ci({"a": [-0.1] * 50, "b": [0.0]})
    assert ci["point"] == pytest.approx(-0.05) and ci["clusters"] == 2


def _items(shift, rng):
    rows = []
    for ref in range(8):
        for text in range(9):
            for seed in range(5):
                base = 0.6 + rng.normal(0, 0.01)
                rows.append({"case_id": f"r{ref}|t{text}|plain", "seed": seed, "ref": f"r{ref}",
                             "text": f"t{text}", "control": "", "sim_eres": base + shift,
                             "sim_wavlm": base + shift,
                             "emb_eres": None, "emb_wavlm": None})
    return rows


def test_regression_gate_fires_only_on_clear_drop(tv):
    rng = np.random.default_rng(0)
    base = _items(0.0, rng)
    assert tv.regression_gate(_items(-0.05, np.random.default_rng(0)), base)["fires"] is True
    assert tv.regression_gate(_items(-0.005, np.random.default_rng(0)), base)["fires"] is False


def test_built_cases_are_paired_and_frozen(tv):
    if not tv.CASES.is_file():
        pytest.skip("尚未 build")
    plan = json.loads(tv.PLAN.read_text())
    assert plan["cases"]["main"]["sha256"] == tv.sha256(tv.CASES)
    rows = [json.loads(line) for line in tv.CASES.read_text().splitlines()]
    by_ref = {}
    for r in rows:
        by_ref.setdefault(r["ref"], []).append(r)
        assert r["seed"] in plan["refs"][r["ref"]]["seeds"]
    for ref_rows in by_ref.values():
        assert len(ref_rows) == (9 + len(tv.CONTROLS)) * tv.SEEDS_PER_REF
        plain = {(r["lang"], r["text"]) for r in ref_rows if r["control"] == "" and r["bucket"] == "medium"}
        # 情绪句与同参考 plain medium 同句，slice_gap 才能逐句配对
        assert all((r["lang"], r["text"]) in plain for r in ref_rows if r["control"])
