import importlib
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch


@pytest.fixture
def module(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / 'scripts'))
    return importlib.import_module('r31_component_probe')


def test_component_scales_are_disjoint_and_restored_after_failure(module):
    names = ['base_lm._orig_mod.layer', 'residual_lm.layer', 'feat_decoder.estimator.layer']
    layers = [SimpleNamespace(scaling=torch.tensor(v), _base_scaling=v) for v in (1., 2., 3.)]
    model = SimpleNamespace(tts_model=SimpleNamespace(
        _iter_lora_modules=lambda: iter(layers), named_modules=lambda: iter(zip(names, layers))))
    with module.component_profile(model, 'dit_only') as actual:
        assert actual['group_counts'] == {'lm': 2, 'dit': 1}
        assert [m.scaling.item() for m in layers] == [0, 0, 3]
    with pytest.raises(RuntimeError):
        with module.component_profile(model, 'lm_only'):
            assert [m.scaling.item() for m in layers] == [1, 2, 0]
            raise RuntimeError('generation failed')
    assert [m.scaling.item() for m in layers] == [1, 2, 3]
    names[0] = 'unknown.adapter'
    with pytest.raises(ValueError, match='Unknown'):
        with module.component_profile(model, 'lm_only'):
            pass


def row(seed=42, **extra):
    return dict(case_id='th_control', seed=seed, lang='th', hyp='spoken', cer=.02,
                rms_dbfs=-25, speech_ratio=.9, over_read=False, **extra)


def test_fail_fast_does_not_wait_for_full_budget(module):
    base = {('th_control', s): row(s) for s in (42, 43)}
    near_silent = {**row(), 'rms_dbfs': -100, 'speech_ratio': 0, 'hyp': ''}
    assert module.reject_reason([near_silent], base) == 'near_silence'
    assert module.reject_reason([{**row(), 'over_read': True}], base) == 'new_suspected_overread'
    assert module.reject_reason([{**row(), 'hyp': ''}], base) == 'new_empty_asr'
    large = [{**row(s), 'cer': .30} for s in (42, 43)]
    assert module.reject_reason(large[:1], base) is None
    assert module.reject_reason(large, base) == 'two_large_paired_cer_regressions'
    assert module.reject_reason([{**row(), 'cer': .04}], base, stage_end=True) == 'stage_mean_cer_regression_th'
    assert module.reject_reason([row()], base, stage_end=True) is None
    with pytest.raises(ValueError, match='Duplicate'):
        module.reject_reason([row(), row()], base)


def test_matching_baseline_overread_is_not_counted_as_new(module):
    base = {('th_control', 42): {**row(), 'over_read': True}}
    assert module.reject_reason([{**row(), 'over_read': True}], base) is None


def test_no_overwrite(module, tmp_path):
    with pytest.raises(FileExistsError):
        module.prepare(tmp_path)
